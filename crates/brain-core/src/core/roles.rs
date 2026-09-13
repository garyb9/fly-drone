//! `Roles` — named input/readout groups over neuron indices. An input role is
//! a set of neurons that `SimCore::inject` drives; a readout role is a set that
//! `SimCore::readout` averages activity over. Role ids are the insertion index
//! into the respective list.

#[derive(Default)]
pub struct Roles {
    input: Vec<(String, Vec<u32>)>,
    readout: Vec<(String, Vec<u32>)>,
}

impl Roles {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn define_input(&mut self, name: &str, ids: &[u32]) -> u32 {
        self.input.push((name.to_string(), ids.to_vec()));
        (self.input.len() - 1) as u32
    }
    pub fn define_readout(&mut self, name: &str, ids: &[u32]) -> u32 {
        self.readout.push((name.to_string(), ids.to_vec()));
        (self.readout.len() - 1) as u32
    }
    pub fn input_neurons(&self, id: u32) -> &[u32] {
        self.input
            .get(id as usize)
            .map(|r| r.1.as_slice())
            .unwrap_or(&[])
    }
    pub fn readout_neurons(&self, id: u32) -> &[u32] {
        self.readout
            .get(id as usize)
            .map(|r| r.1.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_insertion_index_per_list() {
        let mut r = Roles::new();
        assert_eq!(r.define_input("a", &[1, 2]), 0);
        assert_eq!(r.define_input("b", &[3]), 1);
        // readout ids are a separate namespace, also starting at 0
        assert_eq!(r.define_readout("x", &[9]), 0);
        assert_eq!(r.define_readout("y", &[8, 7]), 1);
    }

    #[test]
    fn lookup_returns_stored_neurons() {
        let mut r = Roles::new();
        let a = r.define_input("a", &[4, 5, 6]);
        let x = r.define_readout("x", &[10, 11]);
        assert_eq!(r.input_neurons(a), &[4, 5, 6]);
        assert_eq!(r.readout_neurons(x), &[10, 11]);
    }

    #[test]
    fn unknown_id_yields_empty_slice() {
        let mut r = Roles::new();
        r.define_input("a", &[1]);
        r.define_readout("x", &[2]);
        assert!(r.input_neurons(99).is_empty());
        assert!(r.readout_neurons(99).is_empty());
        // input id space does not leak into readout lookups
        assert!(r.readout_neurons(0).len() == 1);
        assert!(r.input_neurons(0).len() == 1);
    }
}
