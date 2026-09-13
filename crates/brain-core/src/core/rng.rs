//! `SplitMix64` — a tiny, fast, fully deterministic PRNG. Its only consumer is
//! `core::sim`, which draws one Box–Muller Gaussian per active neuron per tick
//! for membrane noise (and only when `noise_sigma > 0`).

/// SplitMix64 — tiny, fast, fully deterministic. Used for membrane noise only.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32_unit(&mut self) -> f32 {
        // 24 random bits -> [0, 1)
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Box–Muller, no cached spare (so call count fully determines the stream).
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32_unit().max(1e-7);
        let u2 = self.next_f32_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn gaussian_is_finite_and_call_count_drives_stream() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..100 {
            let x = a.next_gaussian();
            assert!(x.is_finite());
        }
        // b consumes the same number of raw u64 draws (2 per gaussian) => same next value.
        for _ in 0..200 {
            b.next_u64();
        }
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
