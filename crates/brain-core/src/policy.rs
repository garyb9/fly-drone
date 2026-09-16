use serde::{Deserialize, Serialize};

/// Camera geometry + cue encoder identity; decoders trained on another encoder are rejected.
pub const ENCODER_VERSION: &str = "bright-contrast-400-splay075-noaa-loom150-v4";

/// Learned encoders pin their weights hash after one of these prefixes; Python checks the exact match.
pub const LEARNED_ENCODER_PREFIXES: &[&str] = &["learned-v5:", "learned-v6:"];

/// Squashing of the final layer: PPO actors clip, SAC actors use tanh.
#[derive(Serialize, Deserialize, Default, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Output {
    #[default]
    Clip,
    Tanh,
}

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
    #[serde(default)]
    pub output: Output,
}
impl Policy {
    pub fn from_json(s: &str) -> Result<Self, String> {
        let p: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let n = p.feature_ids.len();
        if (p.encoder_version != ENCODER_VERSION
            && !LEARNED_ENCODER_PREFIXES
                .iter()
                .any(|prefix| p.encoder_version.starts_with(prefix)))
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
        let output = self.output;
        Ok(std::array::from_fn(|i| {
            let v = match output {
                Output::Clip => x[i].clamp(-1.0, 1.0),
                Output::Tanh => x[i].tanh(),
            };
            v * self.action_limits[i]
        }))
    }
}

#[cfg(test)]
mod learned_encoder_tests {
    use super::*;

    fn policy_json(encoder: &str, output: Option<&str>) -> String {
        let mut v = serde_json::json!({
            "version": 1, "encoder_version": encoder, "dataset_hash": "h",
            "feature_ids": [0], "mean": [0.0], "scale": [1.0],
            "layers": [{"weights": [[3.0], [0.5], [0.0], [-3.0]], "bias": [0.0, 0.0, 0.0, 0.0]}],
            "action_limits": [1.0, 2.0, 1.0, 1.0]
        });
        if let Some(o) = output {
            v["output"] = serde_json::json!(o);
        }
        v.to_string()
    }

    #[test]
    fn accepts_v4_and_learned_v5_and_rejects_other_encoders() {
        assert!(Policy::from_json(&policy_json(ENCODER_VERSION, None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v5:0123456789abcdef", None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v4:0123", None)).is_err());
        assert!(Policy::from_json(&policy_json("something-else", None)).is_err());
        assert!(Policy::from_json(&policy_json(ENCODER_VERSION, Some("relu"))).is_err());
    }

    #[test]
    fn accepts_learned_v5_and_v6_and_rejects_other_encoders() {
        assert!(Policy::from_json(&policy_json("learned-v5:0123456789abcdef", None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v6:0123456789abcdef", None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v4:0123", None)).is_err());
        assert!(Policy::from_json(&policy_json("learned-", None)).is_err());
    }

    #[test]
    fn default_output_clips_and_tanh_output_squashes() {
        let clip = Policy::from_json(&policy_json(ENCODER_VERSION, None)).unwrap();
        assert_eq!(clip.infer(&[1.0]).unwrap(), [1.0, 1.0, 0.0, -1.0]);
        let tanh = Policy::from_json(&policy_json(ENCODER_VERSION, Some("tanh"))).unwrap();
        let out = tanh.infer(&[1.0]).unwrap();
        let expected = [3.0f32.tanh(), 0.5f32.tanh() * 2.0, 0.0, -(3.0f32.tanh())];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
