//! Golden-trace determinism lock — **HOST TARGET ONLY**.
//!
//! A fixed script over the committed fixture must always produce the same trace
//! hash on this build target. Regenerate intentionally with `BLESS=1`
//! (`BLESS=1 cargo test -p fly-sim --test golden_trace golden_trace_is_stable`).
//!
//! This lock is host-target-specific: native and `wasm32` may hash differently
//! because `f32::exp` / `ln` / `cos` (in `LifParams::from_ms` and
//! `SplitMix64::next_gaussian`) resolve to different libm implementations across
//! targets. Plan 02 moves the sim into a worker; it must decide separately
//! whether it also needs a dedicated `wasm32`-target trace.
//!
//! What the hash is sensitive to (mutation-test evidence recorded in the Plan 01
//! `final-fix-report.md`): the RNG seed (membrane noise is the only seed
//! consumer and is deliberately left at its nonzero default), `noise_sigma`
//! itself, the excitatory/inhibitory `sign` vector, the `set_active_count`
//! slider value, and the `t < active` edge-gating check in `SimCore::step`. It
//! hashes the whole `activity_snapshot()` vector every sample rather than a
//! couple of role scalars, which is what makes it catch all of those.

mod fixtures;

use brain_core::core::format::{GraphFile, NeuronsFile};
use brain_core::core::sim::{SimConfig, SimCore};
use fixtures::*;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn hash_snapshot(s: &SimCore, bytes: &mut Vec<u8>) {
    for v in s.activity_snapshot() {
        // quantise to 1e-4 so trivial cross-platform float noise doesn't churn the hash
        bytes.extend(((v * 10_000.0).round() as i32).to_le_bytes());
    }
}

fn run_trace() -> String {
    let nf = NeuronsFile::parse(&neurons_bytes()).unwrap();
    let gf = GraphFile::parse(&graph_bytes()).unwrap();
    // `SimConfig { seed, ..default() }` rather than a post-construction
    // `cfg.seed = ...` to satisfy `clippy::field_reassign_with_default`
    // (-D warnings). `noise_sigma` stays at its nonzero default so the seeded
    // PRNG is drawn every tick — the trace must change when the seed changes.
    let cfg = SimConfig {
        seed: FIXTURE_SEED,
        ..SimConfig::default()
    };
    let mut s = SimCore::new(&nf, &gf, cfg);

    // Sensory roles from the shared fixture helpers instead of hardcoded index
    // ranges (also closes Minor 9).
    let mut inputs = std::collections::BTreeMap::new();
    for (name, ids) in fixture_input_roles() {
        inputs.insert(name, s.define_input_role(name, &ids));
    }
    // Readout roles are defined for parity with the production wiring even
    // though the trace hashes the whole activity snapshot, not role scalars.
    for (name, ids) in fixture_readout_roles() {
        s.define_readout_role(name, &ids);
    }

    // A synthetic input role over a high-index band, used ONLY to make the
    // `active_count` edge gate testable. The fixture's background edges are all
    // tiny (`w_raw` 1..7, `w_norm` 0.01), so without direct drive nothing past
    // the ~40 wired sensory/readout neurons ever fires and the gate
    // `if t < self.active` guards rows that are permanently silent — a mutation
    // to it would be invisible. Holding this band a hair below threshold makes
    // those rows live, so widening the gate (leaking one tick of synaptic drive
    // into a row the instant before the slider reaches it) flips spike timing.
    let probe = s.define_input_role("probe_band", &(210u32..470).collect::<Vec<_>>());

    // Fixture wiring (pipeline/gen_fixture.py): looming -> escape (strong),
    // light_l -> wing_r, light_r -> wing_l (contralateral steering). The sensory
    // windows below drive both the escape path and a contralateral steering
    // path; their downstream neurons sit scattered around threshold where the
    // per-tick Gaussian noise (hence the seed) and the exc/inh `sign` decide
    // individual spikes — that is what makes the hash seed / `noise_sigma` /
    // `sign` sensitive.
    let ramp_lo: u32 = 210;
    let ramp_hi: u32 = 460;
    s.set_active_count(ramp_lo as usize);
    let mut active = ramp_lo;

    let mut bytes: Vec<u8> = Vec::new();
    for t in 0..2600u32 {
        // Walk the neuron-count slider by one neuron per tick, up through the
        // probe band, hold, back down, and up again — all while the band is
        // being driven near threshold.
        let span = ramp_hi - ramp_lo;
        let desired = if t < 200 {
            ramp_lo
        } else if t < 200 + span {
            ramp_lo + (t - 200)
        } else if t < 1400 {
            ramp_hi
        } else if t < 1400 + span {
            ramp_hi - (t - 1400)
        } else if t < 1700 {
            ramp_lo
        } else if t < 1700 + span {
            ramp_lo + (t - 1700)
        } else {
            ramp_hi
        };
        if desired != active {
            active = desired;
            s.set_active_count(active as usize);
        }

        if (100..1300).contains(&t) {
            s.inject(inputs["looming"], 0.8);
            s.inject(inputs["proximity"], 0.8);
            s.inject(inputs["wind_l"], 0.8);
        }
        if (1500..2500).contains(&t) {
            s.inject(inputs["light_l"], 0.8); // light_l -> wing_r
            s.inject(inputs["light_r"], 0.8); // light_r -> wing_l (contralateral)
            s.inject(inputs["wind_r"], 0.8);
        }
        if (150..2550).contains(&t) {
            s.inject(probe, 0.9); // one-tick membrane bump just shy of v_threshold (1.0)
        }
        s.step(1);
        if t % 32 == 0 {
            hash_snapshot(&s, &mut bytes);
        }
    }

    format!("{:016x}", fnv1a64(&bytes))
}

#[test]
fn golden_trace_is_stable() {
    let got = run_trace();
    let expected_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden_trace.expected");
    if std::env::var("BLESS").is_ok() {
        std::fs::write(expected_path, format!("{got}\n")).unwrap();
        return;
    }
    let want = std::fs::read_to_string(expected_path)
        .expect("no golden_trace.expected — run once with BLESS=1")
        .trim()
        .to_string();
    assert_eq!(
        got, want,
        "trace changed; if intentional re-run with BLESS=1"
    );
}

#[test]
fn trace_is_reproducible_within_a_run() {
    assert_eq!(run_trace(), run_trace());
}
