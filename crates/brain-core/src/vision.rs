//! Engineered eye adapter, not a reconstructed retina. RGB bytes only.
#[derive(Default)]
pub struct Eyes {
    previous: Option<[[f32; 2]; 2]>,
}
impl Eyes {
    pub fn reset(&mut self) {
        self.previous = None;
    }
    pub fn encode(
        &mut self,
        left: &[u8],
        right: &[u8],
        width: usize,
        height: usize,
    ) -> Result<[f32; 4], String> {
        let n = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(3))
            .ok_or("image too large")?;
        if n == 0 || left.len() != n || right.len() != n {
            return Err("expected two packed RGB images".into());
        }
        let mut current = [[0.0; 2]; 2];
        for (eye, bytes) in [left, right].iter().enumerate() {
            for pixel in bytes.chunks_exact(3) {
                let y = (0.2126 * pixel[0] as f32
                    + 0.7152 * pixel[1] as f32
                    + 0.0722 * pixel[2] as f32)
                    / 255.0;
                current[eye][0] += (y - 0.55).max(0.0) * 400.0 / (width * height) as f32;
                current[eye][1] += if y < 0.18 {
                    1.0 / (width * height) as f32
                } else {
                    0.0
                };
            }
        }
        let prev = self.previous.unwrap_or(current);
        self.previous = Some(current);
        // Sustained bright contrast plus ON increment. The 0.55 threshold and
        // gain 400 calibrate the demo camera range, not fly physiology. Dark-area expansion is a
        // deliberately simple looming proxy; camera rotation can also excite it.
        Ok([0, 1]
            .map(|i| {
                (current[i][0] * 1.5 + (current[i][0] - prev[i][0]).max(0.0) * 6.0).clamp(0.0, 2.0)
            })
            .into_iter()
            .chain([0, 1].map(|i| ((current[i][1] - prev[i][1]).max(0.0) * 12.0).clamp(0.0, 2.0)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reset_and_direction() {
        let mut e = Eyes::default();
        let a = e.encode(&[255; 12], &[0; 12], 2, 2).unwrap();
        assert!(a[0] > a[1]);
        assert_eq!(a[2], 0.0);
        let b = e.encode(&[0; 12], &[0; 12], 2, 2).unwrap();
        assert!(b[2] > 0.0);
        e.reset();
        assert_eq!(e.encode(&[0; 12], &[0; 12], 2, 2).unwrap()[2], 0.0);
    }
    #[test]
    fn rejects_bad_image() {
        assert!(Eyes::default().encode(&[], &[], 1, 1).is_err());
    }
}
