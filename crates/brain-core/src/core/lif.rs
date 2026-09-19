//! Leaky integrate-and-fire (LIF) neuron model: parameters, per-neuron state
//! buffers, and one pure single-neuron integration step. No graph, no RNG, no
//! wasm — `core::sim` drives `integrate_one` per tick with a double-buffered
//! `input` that already folds in synaptic drive, injected stimulus, and noise.

/// Fixed per-simulation LIF coefficients. `leak` is the per-tick membrane
/// multiplier `exp(-dt/tau_m)`; `refrac_ticks` is the absolute refractory
/// period in ticks. Construct via [`LifParams::from_ms`] from millisecond
/// time constants.
#[derive(Clone, Copy, Debug)]
pub struct LifParams {
    pub leak: f32,
    pub v_threshold: f32,
    pub v_reset: f32,
    pub refrac_ticks: u16,
    pub noise_sigma: f32,
}

impl LifParams {
    /// Build params from millisecond time constants: `leak = exp(-dt_ms /
    /// tau_m_ms)` and `refrac_ticks = round(refrac_ms / dt_ms)`.
    pub fn from_ms(
        dt_ms: f32,
        tau_m_ms: f32,
        v_threshold: f32,
        v_reset: f32,
        refrac_ms: f32,
        noise_sigma: f32,
    ) -> Self {
        Self {
            leak: (-dt_ms / tau_m_ms).exp(),
            v_threshold,
            v_reset,
            refrac_ticks: (refrac_ms / dt_ms).round() as u16,
            noise_sigma,
        }
    }
}

impl Default for LifParams {
    fn default() -> Self {
        Self::from_ms(5.0, 20.0, 1.0, 0.0, 2.0, 0.02)
    }
}

/// Per-neuron mutable state, one entry per neuron. `v` is membrane potential,
/// `refrac` the remaining refractory ticks, `spike` the current-tick fire flag
/// (0/1) kept as bytes for cheap wasm transfer.
pub struct LifState {
    pub v: Vec<f32>,
    pub refrac: Vec<u16>,
    pub spike: Vec<u8>,
}

impl LifState {
    pub fn new(n: usize) -> Self {
        Self {
            v: vec![0.0; n],
            refrac: vec![0; n],
            spike: vec![0; n],
        }
    }
}

/// Pure single-neuron update. `input` already includes synaptic drive +
/// injected stimulus + noise for this tick. Order-independent across neurons
/// because the caller double-buffers `input`.
///
/// Returns `(new_v, fired, new_refrac)`. While refractory, `v` is clamped to
/// `v_reset`, no spike is emitted, and the refractory counter decrements.
pub fn integrate_one(v: f32, refrac: u16, input: f32, p: &LifParams) -> (f32, bool, u16) {
    integrate_one_with(v, refrac, input, p.leak, p.v_threshold, p)
}

/// Per-neuron variant of [`integrate_one`]: `leak` and `v_threshold` come from a
/// declared dynamics bundle (P2) instead of the uniform [`LifParams`]. Reset,
/// refractory and noise stay scalar, so an additive bundle changes only the two
/// fitted fields and the canonical uniform-LIF path is bit-identical.
pub fn integrate_one_with(
    v: f32,
    refrac: u16,
    input: f32,
    leak: f32,
    v_threshold: f32,
    p: &LifParams,
) -> (f32, bool, u16) {
    if refrac > 0 {
        return (p.v_reset, false, refrac - 1);
    }
    let v_new = leak * v + input;
    if v_new >= v_threshold {
        (p.v_reset, true, p.refrac_ticks)
    } else {
        (v_new, false, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> LifParams {
        LifParams::from_ms(5.0, 20.0, 1.0, 0.0, 2.0, 0.0)
    }

    #[test]
    fn leak_decays_toward_zero_with_no_input() {
        let params = p();
        let (v1, fired, _) = integrate_one(0.5, 0, 0.0, &params);
        assert!(!fired);
        assert!((v1 - 0.5 * params.leak).abs() < 1e-6);
        assert!(v1 < 0.5);
    }

    #[test]
    fn fires_and_resets_when_input_crosses_threshold() {
        let params = p();
        let (v1, fired, refrac) = integrate_one(0.0, 0, 1.5, &params);
        assert!(fired);
        assert_eq!(v1, params.v_reset);
        assert_eq!(refrac, params.refrac_ticks);
    }

    #[test]
    fn refractory_blocks_firing_and_counts_down() {
        let params = p();
        // big input but still refractory -> no spike, refrac decremented, v held at reset
        let (v1, fired, refrac) = integrate_one(0.0, 2, 10.0, &params);
        assert!(!fired);
        assert_eq!(refrac, 1);
        assert_eq!(v1, params.v_reset);
    }

    #[test]
    fn from_ms_computes_leak_and_refrac() {
        let params = LifParams::from_ms(5.0, 20.0, 1.0, 0.0, 2.0, 0.02);
        assert!((params.leak - (-0.25f32).exp()).abs() < 1e-6);
        assert_eq!(params.refrac_ticks, 0); // 2.0 / 5.0 = 0.4 -> rounds to 0
    }

    #[test]
    fn default_matches_from_ms_defaults() {
        assert_eq!(
            LifParams::default().leak,
            LifParams::from_ms(5.0, 20.0, 1.0, 0.0, 2.0, 0.02).leak
        );
    }

    #[test]
    fn state_new_is_zeroed() {
        let s = LifState::new(4);
        assert_eq!(s.v, vec![0.0; 4]);
        assert_eq!(s.refrac, vec![0u16; 4]);
        assert_eq!(s.spike, vec![0u8; 4]);
    }
}
