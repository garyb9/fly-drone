//! `SimCore` — the simulation heart. Owns all neural state and steps a LIF
//! network over the CSR graph: double-buffered synaptic input, per-neuron
//! excitatory/inhibitory sign, an EMA activity trace, and the `active_count`
//! slider (edges to targets `>= active_count` are skipped during traversal).
//! Driven by the `#[wasm_bindgen] Sim` wrapper in `lib.rs`.

use crate::core::format::{GraphFile, NeuronsFile};
use crate::core::lif::{integrate_one, LifParams, LifState};
use crate::core::rng::SplitMix64;
use crate::core::roles::Roles;

#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub params: LifParams,
    pub seed: u64,
    pub activity_tau_ticks: f32,
    /// Fixed simulation tick length in ms (the "200 Hz tick" Global Constraint).
    /// Stored so it can be read back and cross-checked; it does not feed the
    /// step math directly — `LifParams::from_ms` already folds it into `leak`
    /// and `refrac_ticks`.
    pub dt_ms: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            params: LifParams::default(),
            seed: 0x0DDB_1A5E,
            activity_tau_ticks: 40.0,
            dt_ms: 5.0,
        }
    }
}

pub struct SimCore {
    n: usize,
    core_count: usize,
    active: usize,
    params: LifParams,
    dt_ms: f32,
    activity_tau: f32,
    rng: SplitMix64,

    // CSR (owned copies from GraphFile)
    offsets: Vec<u32>,
    targets: Vec<u32>,
    weights_sim: Vec<f32>, // pre-multiplied by w_norm

    sign: Vec<f32>, // +1.0 or -1.0 per neuron
    silenced: Vec<bool>,
    bias: Vec<f32>, // tonic drive per neuron (0 for now)

    state: LifState,
    input_cur: Vec<f32>,
    input_next: Vec<f32>,
    activity: Vec<f32>,
    activity_extent: usize,

    roles: Roles,
}

impl SimCore {
    pub fn new(neurons: &NeuronsFile, graph: &GraphFile, cfg: SimConfig) -> Self {
        assert_eq!(
            neurons.count(),
            graph.n_nodes,
            "neuron/graph node count mismatch"
        );
        let n = neurons.count();
        let sign: Vec<f32> = (0..n)
            .map(|i| if neurons.is_inhibitory(i) { -1.0 } else { 1.0 })
            .collect();
        let weights_sim: Vec<f32> = graph
            .weights
            .iter()
            .map(|&w| w as f32 * graph.w_norm)
            .collect();
        Self {
            n,
            core_count: neurons.core_count as usize,
            active: n,
            params: cfg.params,
            dt_ms: cfg.dt_ms,
            activity_tau: cfg.activity_tau_ticks.max(1.0),
            rng: SplitMix64::new(cfg.seed),
            offsets: graph.offsets.clone(),
            targets: graph.targets.clone(),
            weights_sim,
            sign,
            silenced: vec![false; n],
            bias: vec![0.0; n],
            state: LifState::new(n),
            input_cur: vec![0.0; n],
            input_next: vec![0.0; n],
            activity: vec![0.0; n],
            activity_extent: 0,
            roles: Roles::new(),
        }
    }

    pub fn neuron_count(&self) -> usize {
        self.n
    }
    pub fn core_count(&self) -> usize {
        self.core_count
    }
    pub fn active_count(&self) -> usize {
        self.active
    }
    pub fn dt_ms(&self) -> f32 {
        self.dt_ms
    }
    pub fn set_params(&mut self, p: LifParams) {
        self.params = p;
    }
    pub fn set_dt_ms(&mut self, dt_ms: f32) {
        self.dt_ms = dt_ms;
    }
    pub fn set_active_count(&mut self, n: usize) {
        self.active = n.clamp(self.core_count, self.n);
    }
    pub fn add_input(&mut self, i: usize, amount: f32) {
        if i < self.n {
            self.input_cur[i] += amount;
        }
    }
    pub fn inject_cells(&mut self, ids: &[u32], value: f32) {
        if !value.is_finite() {
            return;
        }
        for &id in ids {
            self.add_input(id as usize, value.clamp(-5.0, 5.0));
        }
    }
    pub fn set_bias(&mut self, ids: &[u32], value: f32) {
        if !value.is_finite() {
            return;
        }
        for &id in ids {
            if let Some(bias) = self.bias.get_mut(id as usize) {
                *bias = value.clamp(0.0, 2.0);
            }
        }
    }
    pub fn silence_cells(&mut self, ids: &[u32], value: bool) {
        for &id in ids {
            if let Some(cell) = self.silenced.get_mut(id as usize) {
                *cell = value;
            }
        }
    }
    pub fn clear_interventions(&mut self) {
        self.silenced.fill(false);
    }
    pub fn reset(&mut self, seed: u64) {
        self.rng = SplitMix64::new(seed);
        self.state = LifState::new(self.n);
        self.input_cur.fill(0.0);
        self.input_next.fill(0.0);
        self.activity.fill(0.0);
        self.activity_extent = 0;
        self.clear_interventions();
    }

    pub fn activity(&self) -> &[f32] {
        &self.activity
    }
    pub fn spikes(&self) -> &[u8] {
        &self.state.spike
    }

    pub fn define_input_role(&mut self, name: &str, neurons: &[u32]) -> u32 {
        self.roles.define_input(name, neurons)
    }
    pub fn define_readout_role(&mut self, name: &str, neurons: &[u32]) -> u32 {
        self.roles.define_readout(name, neurons)
    }
    pub fn inject(&mut self, role_id: u32, value: f32) {
        for &i in self.roles.input_neurons(role_id) {
            let i = i as usize;
            if i < self.n {
                self.input_cur[i] += value;
            }
        }
    }
    pub fn readout(&self, role_id: u32) -> f32 {
        let ids = self.roles.readout_neurons(role_id);
        if ids.is_empty() {
            return 0.0;
        }
        // Inactive cells can retain frozen activity; they must not drive a body.
        let mut sum = 0.0;
        let mut count = 0;
        for &i in ids {
            if (i as usize) < self.active {
                sum += self.activity[i as usize];
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }
    pub fn activity_snapshot(&self) -> Vec<f32> {
        self.activity[..self.active].to_vec()
    }

    #[cfg(test)]
    pub fn debug_v(&self, i: usize) -> f32 {
        self.state.v[i]
    }

    pub fn step(&mut self, ticks: u32) {
        let inv_tau = 1.0 / self.activity_tau;
        self.activity_extent = self.activity_extent.max(self.active);
        for _ in 0..ticks {
            for i in 0..self.active {
                let noise = if self.params.noise_sigma > 0.0 {
                    self.rng.next_gaussian() * self.params.noise_sigma
                } else {
                    0.0
                };
                if self.silenced[i] {
                    self.state.v[i] = self.params.v_reset;
                    self.state.refrac[i] = 0;
                    self.state.spike[i] = 0;
                    continue;
                }
                let input = self.input_cur[i] + self.bias[i] + noise;
                let (v_new, fired, refrac_new) =
                    integrate_one(self.state.v[i], self.state.refrac[i], input, &self.params);
                self.state.v[i] = v_new;
                self.state.refrac[i] = refrac_new;
                self.state.spike[i] = fired as u8;

                if fired {
                    let s = self.offsets[i] as usize;
                    let e = self.offsets[i + 1] as usize;
                    let sgn = self.sign[i];
                    for k in s..e {
                        let t = self.targets[k] as usize;
                        if t < self.active {
                            self.input_next[t] += self.weights_sim[k] * sgn;
                        }
                    }
                }
            }
            // Preserve decay for previously active neurons; untouched zero tail needs no work.
            for i in 0..self.activity_extent {
                let sp = if i < self.active {
                    self.state.spike[i] as f32
                } else {
                    0.0
                };
                self.activity[i] += (sp - self.activity[i]) * inv_tau;
            }
            // swap buffers, clear next
            std::mem::swap(&mut self.input_cur, &mut self.input_next);
            for x in self.input_next.iter_mut() {
                *x = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::format::{GraphFile, GraphFile as GF, NeuronsFile, NeuronsFile as NF};

    // Build a tiny hand-made pair of files: 3 neurons (index 0 core), edges
    // 0->1 and 0->2. `flags0` is neuron 0's flag byte (bit1 = inhibitory);
    // `weights` are the two raw i16 edge weights (w_norm is fixed at 0.01).
    fn tiny_with(flags0: u8, weights: [i16; 2]) -> (Vec<u8>, Vec<u8>) {
        // neurons.bin: count=3, core_count=1
        let mut n = Vec::new();
        n.extend(0x4E59_4C46u32.to_le_bytes());
        n.extend(1u32.to_le_bytes());
        n.extend(3u32.to_le_bytes());
        n.extend(1u32.to_le_bytes());
        for (idx, flags) in [(1u64, flags0), (2, 0), (3, 0)] {
            n.extend(idx.to_le_bytes());
            n.extend(0f32.to_le_bytes());
            n.extend(0f32.to_le_bytes());
            n.extend(0f32.to_le_bytes());
            n.extend(0u16.to_le_bytes());
            n.push(flags);
            n.push(0);
        }
        // graph.bin: n_nodes=3, edges 0->1 and 0->2 with the given weights ; w_norm=0.01
        let mut g = Vec::new();
        g.extend(0x4759_4C46u32.to_le_bytes());
        g.extend(1u32.to_le_bytes());
        g.extend(3u32.to_le_bytes());
        g.extend(0u32.to_le_bytes());
        g.extend(2u64.to_le_bytes());
        g.extend(0.01f32.to_le_bytes());
        g.extend(0u32.to_le_bytes());
        for o in [0u32, 2, 2, 2] {
            g.extend(o.to_le_bytes());
        } // offsets[4]
        for t in [1u32, 2] {
            g.extend(t.to_le_bytes());
        }
        for w in weights {
            g.extend(w.to_le_bytes());
        }
        (n, g)
    }

    // Default fixture: neuron 0 is a non-inhibitory sensory-input neuron
    // (flags 0b0100 => sign = +1.0), edge 0->1 carries +60 and edge 0->2
    // carries -60. Exercises "a negative `weights_sim` entry propagates a
    // negative contribution"; `inhibitory_source_flips_propagation_sign`
    // covers the orthogonal "inhibitory flag flips an otherwise-positive
    // weight" path.
    fn tiny() -> (Vec<u8>, Vec<u8>) {
        tiny_with(0b0100, [60, -60])
    }

    fn core_from(nb: &[u8], gb: &[u8], seed: u64) -> SimCore {
        let nf = NeuronsFile::parse(nb).unwrap();
        let gf = GraphFile::parse(gb).unwrap();
        // Same effect as `let mut cfg = SimConfig::default(); cfg.seed = seed;
        // cfg.params.noise_sigma = 0.0;` from the task brief, written as struct
        // literals to satisfy `clippy::field_reassign_with_default` (-D warnings).
        let cfg = SimConfig {
            seed,
            params: LifParams {
                noise_sigma: 0.0,
                ..LifParams::default()
            },
            ..SimConfig::default()
        };
        SimCore::new(&nf, &gf, cfg)
    }

    #[test]
    fn injected_input_makes_source_fire_then_propagates_signed_weights() {
        // Excitatory source (sign[0] = +1.0); the downstream signs come purely
        // from the edge weights: 0->1 is +60, 0->2 is -60.
        let (nb, gb) = tiny();
        let mut s = core_from(&nb, &gb, 1);
        s.add_input(0, 2.0); // above threshold
        s.step(1); // tick A: neuron 0 fires, writes into next buffer
        assert_eq!(s.spikes()[0], 1);
        s.step(1); // tick B: neuron 1 gets +60*0.01, neuron 2 gets -60*0.01
                   // neuron 1 should have positive v, neuron 2 negative
        assert!(s.debug_v(1) > 0.0);
        assert!(s.debug_v(2) < 0.0);
    }

    #[test]
    fn inhibitory_source_flips_propagation_sign() {
        // neuron 0: core + INHIBITORY (flags 0b0110); edges 0->1 (+100), 0->2 (+100).
        // Both edge weights are POSITIVE, but sign[0] = -1.0 (built by
        // `SimCore::new` from the inhibitory flag and multiplied in by `step`),
        // so both targets must receive a NEGATIVE contribution.
        let (nb, gb) = tiny_with(0b0110, [100, 100]);
        let mut s = core_from(&nb, &gb, 1);
        s.add_input(0, 2.0);
        s.step(1); // neuron 0 fires
        assert_eq!(s.spikes()[0], 1);
        s.step(1); // propagate: 1 and 2 each get 100 * w_norm * sign[0](-1)
        assert!(s.debug_v(1) < 0.0);
        assert!(s.debug_v(2) < 0.0);
    }

    #[test]
    fn silencing_blocks_spikes_and_downstream_propagation_then_reset_replays() {
        let (nb, gb) = tiny();
        let mut s = core_from(&nb, &gb, 42);
        s.silence_cells(&[0], true);
        s.inject_cells(&[0], 5.0);
        s.step(2);
        assert_eq!(s.spikes()[0], 0);
        assert_eq!(s.debug_v(1), 0.0);
        s.reset(42);
        s.inject_cells(&[0], 2.0);
        s.step(2);
        assert!(s.debug_v(1) > 0.0);
        let expected = s.activity_snapshot();
        s.reset(42);
        s.inject_cells(&[0], 2.0);
        s.step(2);
        assert_eq!(s.activity_snapshot(), expected);
    }

    #[test]
    fn inactive_readout_does_not_reuse_frozen_activity() {
        let (nb, gb) = tiny();
        let mut s = core_from(&nb, &gb, 42);
        let role = s.define_readout_role("leg", &[1]);
        s.inject_cells(&[1], 5.0);
        s.step(1);
        assert!(s.readout(role) > 0.0);
        s.set_active_count(1);
        assert_eq!(s.readout(role), 0.0);
    }

    #[test]
    fn active_count_gates_integration_and_edges() {
        let (nb, gb) = tiny();
        let mut s = core_from(&nb, &gb, 1);
        s.set_active_count(1); // only neuron 0 active (core_count is 1)
        s.add_input(0, 2.0);
        s.step(2);
        assert_eq!(
            s.debug_v(1),
            0.0,
            "edge into inactive neuron must be skipped"
        );
        assert_eq!(s.debug_v(2), 0.0);
    }

    #[test]
    fn set_active_count_clamps_to_core_min() {
        let (nb, gb) = tiny();
        let mut s = core_from(&nb, &gb, 1);
        s.set_active_count(0);
        assert_eq!(s.active_count(), 1); // == core_count
        s.set_active_count(999);
        assert_eq!(s.active_count(), 3);
    }

    #[test]
    fn dt_ms_is_stored_and_readable() {
        let (nb, gb) = tiny();
        let mut s = core_from(&nb, &gb, 1);
        assert_eq!(s.dt_ms(), 5.0, "default dt_ms");
        s.set_dt_ms(10.0);
        assert_eq!(s.dt_ms(), 10.0);
        s.set_active_count(2);
        assert_eq!(s.active_count(), 2, "active_count round-trips");
        assert_eq!(s.dt_ms(), 10.0, "set_active_count leaves dt_ms alone");
    }

    #[test]
    fn deterministic_for_equal_seed_and_inputs() {
        let (nb, gb) = tiny();
        let run = |seed| {
            let mut s = core_from(&nb, &gb, seed);
            for t in 0..50 {
                if t % 7 == 0 {
                    s.add_input(0, 1.2);
                }
                s.step(1);
            }
            s.activity().to_vec()
        };
        assert_eq!(run(42), run(42));
    }

    #[test]
    fn deterministic_with_noise_end_to_end() {
        // Same scripted input, `noise_sigma > 0` so `step` draws from the
        // seeded SplitMix64 every tick: equal seed => bit-identical activity
        // trace; different seed => a different trace.
        let (nb, gb) = tiny();
        let run = |seed| {
            let nf = NeuronsFile::parse(&nb).unwrap();
            let gf = GraphFile::parse(&gb).unwrap();
            let cfg = SimConfig {
                seed,
                params: LifParams {
                    noise_sigma: 0.03,
                    ..LifParams::default()
                },
                ..SimConfig::default()
            };
            let mut s = SimCore::new(&nf, &gf, cfg);
            for t in 0..90 {
                if t % 3 == 0 {
                    s.add_input(0, 1.0); // parked at threshold: noise decides firing
                }
                s.step(1);
            }
            s.activity().to_vec()
        };
        assert_eq!(run(7), run(7), "equal seed + inputs must be bit-identical");
        assert_ne!(run(7), run(9), "different seed must diverge");
    }

    #[test]
    #[should_panic]
    fn new_panics_on_node_count_mismatch() {
        let (nb, _gb) = tiny();
        let bad = _gb_with_nodes(4);
        let nf = NeuronsFile::parse(&nb).unwrap();
        let gf = GraphFile::parse(&bad).unwrap();
        SimCore::new(&nf, &gf, SimConfig::default());
    }

    fn _gb_with_nodes(n: u32) -> Vec<u8> {
        let mut g = Vec::new();
        g.extend(0x4759_4C46u32.to_le_bytes());
        g.extend(1u32.to_le_bytes());
        g.extend(n.to_le_bytes());
        g.extend(0u32.to_le_bytes());
        g.extend(0u64.to_le_bytes());
        g.extend(0.01f32.to_le_bytes());
        g.extend(0u32.to_le_bytes());
        for _ in 0..=n {
            g.extend(0u32.to_le_bytes());
        }
        g
    }

    fn fixture_core() -> SimCore {
        let nb = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../pipeline/out/fixture/neurons.bin"
        ))
        .expect("run python pipeline/gen_fixture.py");
        let gb = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../pipeline/out/fixture/graph.bin"
        ))
        .unwrap();
        let nf = NF::parse(&nb).unwrap();
        let gf = GF::parse(&gb).unwrap();
        let cfg = SimConfig {
            params: LifParams {
                noise_sigma: 0.0,
                ..LifParams::default()
            },
            ..SimConfig::default()
        };
        SimCore::new(&nf, &gf, cfg)
    }

    #[test]
    fn snapshot_length_tracks_active_count() {
        let mut s = fixture_core();
        s.set_active_count(120);
        s.step(1);
        assert_eq!(s.activity_snapshot().len(), 120);
    }

    #[test]
    fn readout_is_zero_before_any_activity() {
        let mut s = fixture_core();
        let escape = s.define_readout_role("escape", &(24u32..32).collect::<Vec<_>>());
        s.step(5);
        assert_eq!(s.readout(escape), 0.0);
    }

    #[test]
    fn sustained_looming_injection_drives_escape_readout() {
        let mut s = fixture_core();
        let looming = s.define_input_role("looming", &(0u32..8).collect::<Vec<_>>());
        let escape = s.define_readout_role("escape", &(24u32..32).collect::<Vec<_>>());
        for _ in 0..400 {
            s.inject(looming, 1.5);
            s.step(1);
        }
        assert!(
            s.readout(escape) > 0.5,
            "escape readout was {}",
            s.readout(escape)
        );
    }

    #[test]
    fn injection_into_role_not_addressed_is_ignored() {
        let mut s = fixture_core();
        let _looming = s.define_input_role("looming", &(0u32..8).collect::<Vec<_>>());
        s.inject(99, 5.0); // no such role
        s.step(1);
        assert!(s.activity_snapshot().iter().all(|&a| a == 0.0));
    }
}
