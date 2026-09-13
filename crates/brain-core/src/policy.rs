use serde::{Deserialize, Serialize};

/// Camera geometry + cue encoder identity; decoders trained on another encoder are rejected.
pub const ENCODER_VERSION: &str = "bright-contrast-400-splay075-noaa-v3";

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
pub struct Policy {
    pub version: u32,
    pub encoder_version: String,
    pub dataset_hash: String,
    pub feature_ids: Vec<u32>,
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    pub layers: Vec<Layer>,
    pub action_limits: [f32; 4],
}
impl Policy {
    pub fn from_json(s: &str) -> Result<Self, String> {
        let p: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let n = p.feature_ids.len();
        if p.encoder_version != ENCODER_VERSION
            || p.version != 1
            || n == 0
            || p.mean.len() != n
            || p.scale.len() != n
            || p.layers.is_empty()
        {
            return Err("invalid policy header".into());
        }
        if p.mean.iter().any(|v| !v.is_finite())
            || p.scale.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || p.action_limits.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("invalid normalization or limits".into());
        }
        let mut width = n;
        for l in &p.layers {
            if l.weights.is_empty()
                || l.weights.len() != l.bias.len()
                || l.weights
                    .iter()
                    .any(|r| r.len() != width || r.iter().any(|x| !x.is_finite()))
                || l.bias.iter().any(|v| !v.is_finite())
            {
                return Err("invalid layer".into());
            }
            width = l.bias.len();
        }
        if width != 4 {
            return Err("policy must output four commands".into());
        }
        Ok(p)
    }
    pub fn infer(&self, features: &[f32]) -> Result<[f32; 4], String> {
        if features.len() != self.mean.len() || features.iter().any(|x| !x.is_finite()) {
            return Err("invalid neural features".into());
        }
        let mut x: Vec<f32> = features
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((x, m), s)| (x - m) / s)
            .collect();
        for (i, l) in self.layers.iter().enumerate() {
            x = l
                .weights
                .iter()
                .zip(&l.bias)
                .map(|(r, b)| {
                    let v = r.iter().zip(&x).fold(*b, |a, (w, x)| a + w * x);
                    if i + 1 == self.layers.len() {
                        v
                    } else {
                        v.tanh()
                    }
                })
                .collect();
        }
        Ok(std::array::from_fn(|i| {
            x[i].clamp(-1.0, 1.0) * self.action_limits[i]
        }))
    }
}
