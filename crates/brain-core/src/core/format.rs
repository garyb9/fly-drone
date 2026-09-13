//! Read-only decoders for the committed `neurons.bin` / `graph.bin` fixture
//! bytes. Layout is authoritative in the Plan 01 doc's "Binary formats"
//! section (16-byte `neurons.bin` header + 24-byte records; 32-byte padded
//! `graph.bin` header + CSR arrays). Little-endian throughout.
//!
//! Fields / accessors here (`version`, `group_id`, `is_inhibitory`, …) are part
//! of the format contract and are consumed by `core::lif`, `core::sim` and the
//! wasm wrapper.

use std::convert::TryInto;

const NEURONS_MAGIC: u32 = 0x4E59_4C46; // "FLYN" LE
const GRAPH_MAGIC: u32 = 0x4759_4C46; // "FLYG" LE
const NEURON_REC: usize = 24;

#[derive(Debug, PartialEq, Eq)]
pub enum FormatError {
    TooShort {
        need: usize,
        got: usize,
    },
    BadMagic {
        expected: u32,
        got: u32,
    },
    BadOffsets,
    /// Declared arrays cannot fit the address space or the u32 CSR offsets.
    SizeOverflow,
    /// `neurons.bin` header claims more core neurons than total neurons. Left
    /// unchecked this reaches `set_active_count`'s `n.clamp(core_count, n)` with
    /// `min > max`, which panics and poisons the wasm module instance.
    BadHeader {
        core_count: u32,
        count: u32,
    },
}

#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
#[inline]
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
#[inline]
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
#[inline]
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}

pub struct NeuronsFile {
    pub version: u32,
    pub core_count: u32,
    pub ids: Vec<u64>,
    pub pos: Vec<[f32; 3]>,
    pub group_id: Vec<u16>,
    pub flags: Vec<u8>,
}

impl NeuronsFile {
    pub fn count(&self) -> usize {
        self.ids.len()
    }
    pub fn is_core(&self, i: usize) -> bool {
        self.flags[i] & 0b0001 != 0
    }
    pub fn is_inhibitory(&self, i: usize) -> bool {
        self.flags[i] & 0b0010 != 0
    }
    pub fn is_input(&self, i: usize) -> bool {
        self.flags[i] & 0b0100 != 0
    }
    pub fn is_readout(&self, i: usize) -> bool {
        self.flags[i] & 0b1000 != 0
    }

    pub fn parse(b: &[u8]) -> Result<Self, FormatError> {
        if b.len() < 16 {
            return Err(FormatError::TooShort {
                need: 16,
                got: b.len(),
            });
        }
        let magic = u32_at(b, 0);
        if magic != NEURONS_MAGIC {
            return Err(FormatError::BadMagic {
                expected: NEURONS_MAGIC,
                got: magic,
            });
        }
        let version = u32_at(b, 4);
        let count = u32_at(b, 8) as usize;
        let core_count = u32_at(b, 12);
        if core_count as usize > count {
            return Err(FormatError::BadHeader {
                core_count,
                count: count as u32,
            });
        }
        let need = count
            .checked_mul(NEURON_REC)
            .and_then(|n| n.checked_add(16))
            .ok_or(FormatError::SizeOverflow)?;
        if b.len() < need {
            return Err(FormatError::TooShort { need, got: b.len() });
        }
        let mut ids = Vec::with_capacity(count);
        let mut pos = Vec::with_capacity(count);
        let mut group_id = Vec::with_capacity(count);
        let mut flags = Vec::with_capacity(count);
        for k in 0..count {
            let o = 16 + k * NEURON_REC;
            ids.push(u64_at(b, o));
            pos.push([f32_at(b, o + 8), f32_at(b, o + 12), f32_at(b, o + 16)]);
            group_id.push(u16_at(b, o + 20));
            flags.push(b[o + 22]);
        }
        Ok(Self {
            version,
            core_count,
            ids,
            pos,
            group_id,
            flags,
        })
    }
}

pub struct GraphFile {
    pub version: u32,
    pub n_nodes: usize,
    pub w_norm: f32,
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    pub weights: Vec<i16>,
}

pub struct RowIter<'a> {
    t: &'a [u32],
    w: &'a [i16],
    i: usize,
}
impl Iterator for RowIter<'_> {
    type Item = (u32, i16);
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.t.len() {
            return None;
        }
        let out = (self.t[self.i], self.w[self.i]);
        self.i += 1;
        Some(out)
    }
}

impl GraphFile {
    pub fn n_edges(&self) -> usize {
        self.targets.len()
    }

    pub fn row(&self, i: usize) -> RowIter<'_> {
        let s = self.offsets[i] as usize;
        let e = self.offsets[i + 1] as usize;
        RowIter {
            t: &self.targets[s..e],
            w: &self.weights[s..e],
            i: 0,
        }
    }

    pub fn parse(b: &[u8]) -> Result<Self, FormatError> {
        if b.len() < 32 {
            return Err(FormatError::TooShort {
                need: 32,
                got: b.len(),
            });
        }
        let magic = u32_at(b, 0);
        if magic != GRAPH_MAGIC {
            return Err(FormatError::BadMagic {
                expected: GRAPH_MAGIC,
                got: magic,
            });
        }
        let version = u32_at(b, 4);
        let n_nodes = u32_at(b, 8) as usize;
        let edge_count = u64_at(b, 16);
        // CSR offsets are u32 even on a 64-bit host. Never truncate the u64
        // header on wasm32 before validating it.
        if edge_count > u32::MAX as u64 {
            return Err(FormatError::SizeOverflow);
        }
        let n_edges = usize::try_from(edge_count).map_err(|_| FormatError::SizeOverflow)?;
        let w_norm = f32_at(b, 24);

        let offset_count = n_nodes.checked_add(1).ok_or(FormatError::SizeOverflow)?;
        let off_bytes = offset_count
            .checked_mul(4)
            .ok_or(FormatError::SizeOverflow)?;
        let tgt_bytes = n_edges.checked_mul(4).ok_or(FormatError::SizeOverflow)?;
        let wt_bytes = n_edges.checked_mul(2).ok_or(FormatError::SizeOverflow)?;
        let need = 32usize
            .checked_add(off_bytes)
            .and_then(|n| n.checked_add(tgt_bytes))
            .and_then(|n| n.checked_add(wt_bytes))
            .ok_or(FormatError::SizeOverflow)?;
        if b.len() < need {
            return Err(FormatError::TooShort { need, got: b.len() });
        }

        let mut offsets = Vec::with_capacity(offset_count);
        for k in 0..=n_nodes {
            offsets.push(u32_at(b, 32 + k * 4));
        }
        if offsets[0] != 0
            || offsets[n_nodes] as usize != n_edges
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(FormatError::BadOffsets);
        }

        let tbase = 32 + off_bytes;
        let mut targets = Vec::with_capacity(n_edges);
        for k in 0..n_edges {
            let t = u32_at(b, tbase + k * 4);
            if t as usize >= n_nodes {
                return Err(FormatError::BadOffsets);
            }
            targets.push(t);
        }

        let wbase = tbase + tgt_bytes;
        let mut weights = Vec::with_capacity(n_edges);
        for k in 0..n_edges {
            weights.push(i16::from_le_bytes(
                b[wbase + k * 2..wbase + k * 2 + 2].try_into().unwrap(),
            ));
        }

        Ok(Self {
            version,
            n_nodes,
            w_norm,
            offsets,
            targets,
            weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neurons() -> Vec<u8> {
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../pipeline/out/fixture/neurons.bin"
        ))
        .expect("run `python pipeline/gen_fixture.py` first")
    }
    fn graph() -> Vec<u8> {
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../pipeline/out/fixture/graph.bin"
        ))
        .unwrap()
    }

    #[test]
    fn parses_fixture_neurons() {
        let n = NeuronsFile::parse(&neurons()).unwrap();
        assert_eq!(n.count(), 500);
        assert_eq!(n.core_count, 48);
        assert!((0..48).all(|i| n.is_core(i)));
        assert!(!(48..500).any(|i| n.is_core(i)));
        assert!((0..8).all(|i| n.is_input(i)));
        assert!((24..32).all(|i| n.is_readout(i)));
        assert!(n.pos.iter().all(|p| p.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn parses_fixture_graph_csr() {
        let g = GraphFile::parse(&graph()).unwrap();
        assert_eq!(g.n_nodes, 500);
        assert_eq!(g.offsets.len(), 501);
        assert_eq!(*g.offsets.last().unwrap() as usize, g.n_edges());
        assert!(g.offsets.windows(2).all(|w| w[0] <= w[1]));
        assert!(g.targets.iter().all(|&t| (t as usize) < g.n_nodes));
        assert!(g.w_norm > 0.0);
    }

    #[test]
    fn row_iter_matches_offsets() {
        let g = GraphFile::parse(&graph()).unwrap();
        for src in 0..8usize {
            let targets: Vec<u32> = g.row(src).map(|(t, _)| t).collect();
            for esc in 24u32..32 {
                assert!(
                    targets.contains(&esc),
                    "looming {src} -> escape {esc} missing"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = neurons();
        b[0] ^= 0xFF;
        assert!(matches!(
            NeuronsFile::parse(&b),
            Err(FormatError::BadMagic { .. })
        ));
    }

    #[test]
    fn rejects_core_count_exceeding_count() {
        // Hand-built 16-byte header: count = 3, core_count = 99.
        let mut b = Vec::new();
        b.extend(NEURONS_MAGIC.to_le_bytes());
        b.extend(1u32.to_le_bytes()); // version
        b.extend(3u32.to_le_bytes()); // count
        b.extend(99u32.to_le_bytes()); // core_count > count
        assert!(matches!(
            NeuronsFile::parse(&b),
            Err(FormatError::BadHeader {
                core_count: 99,
                count: 3
            })
        ));
    }

    #[test]
    fn rejects_edge_counts_that_cannot_be_addressed_by_csr() {
        for count in [u32::MAX as u64 + 1, (1u64 << 32) + 10, u64::MAX] {
            let mut b = graph();
            b[16..24].copy_from_slice(&count.to_le_bytes());
            assert!(matches!(
                GraphFile::parse(&b),
                Err(FormatError::SizeOverflow)
            ));
        }
    }

    #[test]
    fn rejects_huge_array_headers_before_allocation() {
        let mut nb = neurons();
        nb[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            NeuronsFile::parse(&nb),
            Err(FormatError::SizeOverflow | FormatError::TooShort { .. })
        ));
        let mut gb = graph();
        gb[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            GraphFile::parse(&gb),
            Err(FormatError::SizeOverflow | FormatError::TooShort { .. })
        ));
        gb[8..12].copy_from_slice(&500u32.to_le_bytes());
        gb[16..24].copy_from_slice(&(u32::MAX as u64).to_le_bytes());
        assert!(matches!(
            GraphFile::parse(&gb),
            Err(FormatError::SizeOverflow | FormatError::TooShort { .. })
        ));
    }

    #[test]
    fn rejects_truncated() {
        let b = graph();
        assert!(matches!(
            GraphFile::parse(&b[..40]),
            Err(FormatError::TooShort { .. })
        ));
    }
}
