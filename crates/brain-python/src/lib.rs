use brain_core::{
    core::{
        format::{GraphFile, NeuronsFile},
        sim::{SimConfig, SimCore},
    },
    policy::Policy,
    vision::Eyes,
};
use pyo3::{exceptions::PyValueError, prelude::*};
fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
#[pyclass]
struct Brain {
    inner: SimCore,
    eyes: Eyes,
    policy: Option<Policy>,
}
#[pymethods]
impl Brain {
    #[new]
    fn new(neurons: &[u8], graph: &[u8], seed: u64) -> PyResult<Self> {
        let n = NeuronsFile::parse(neurons).map_err(|e| err(format!("{e:?}")))?;
        let g = GraphFile::parse(graph).map_err(|e| err(format!("{e:?}")))?;
        if n.count() != g.n_nodes {
            return Err(err("graph size mismatch"));
        }
        Ok(Self {
            inner: SimCore::new(
                &n,
                &g,
                SimConfig {
                    seed,
                    ..Default::default()
                },
            ),
            eyes: Eyes::default(),
            policy: None,
        })
    }
    fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
        self.eyes.reset();
    }
    fn clear_vision_history(&mut self) {
        self.eyes.reset();
    }
    fn neuron_count(&self) -> usize {
        self.inner.neuron_count()
    }
    fn input_role(&mut self, name: &str, ids: Vec<u32>) -> PyResult<u32> {
        self.validate(&ids)?;
        Ok(self.inner.define_input_role(name, &ids))
    }
    fn readout_role(&mut self, name: &str, ids: Vec<u32>) -> PyResult<u32> {
        self.validate(&ids)?;
        Ok(self.inner.define_readout_role(name, &ids))
    }
    fn inject(&mut self, role: u32, value: f32) -> PyResult<()> {
        if !value.is_finite() || !(0.0..=2.0).contains(&value) {
            return Err(err("current must be finite in [0,2]"));
        }
        self.inner.inject(role, value);
        Ok(())
    }
    fn stimulate(&mut self, ids: Vec<u32>, value: f32) -> PyResult<()> {
        self.validate(&ids)?;
        if !value.is_finite() {
            return Err(err("nonfinite current"));
        }
        self.inner.inject_cells(&ids, value);
        Ok(())
    }
    fn silence(&mut self, ids: Vec<u32>, value: bool) -> PyResult<()> {
        self.validate(&ids)?;
        self.inner.silence_cells(&ids, value);
        Ok(())
    }
    fn restore(&mut self) {
        self.inner.clear_interventions();
    }
    fn bias(&mut self, ids: Vec<u32>, value: f32) -> PyResult<()> {
        self.validate(&ids)?;
        if !value.is_finite() {
            return Err(err("nonfinite bias"));
        }
        self.inner.set_bias(&ids, value);
        Ok(())
    }
    fn step(&mut self, py: Python<'_>, ticks: u32) -> PyResult<()> {
        if ticks > 10000 {
            return Err(err("tick batch too large"));
        }
        py.allow_threads(|| self.inner.step(ticks));
        Ok(())
    }
    fn readout(&self, role: u32) -> f32 {
        self.inner.readout(role)
    }
    fn activity(&self, ids: Vec<u32>) -> PyResult<Vec<f32>> {
        self.validate(&ids)?;
        Ok(ids
            .iter()
            .map(|i| self.inner.activity()[*i as usize])
            .collect())
    }
    fn snapshot(&self) -> Vec<f32> {
        self.inner.activity_snapshot()
    }
    fn encode(
        &mut self,
        left: &[u8],
        right: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<[f32; 4]> {
        self.eyes.encode(left, right, width, height).map_err(err)
    }
    fn load_policy(
        &mut self,
        json: &str,
        dataset_hash: &str,
        feature_ids: Vec<u32>,
    ) -> PyResult<()> {
        let p = Policy::from_json(json).map_err(err)?;
        if p.dataset_hash != dataset_hash || p.feature_ids != feature_ids {
            return Err(err("policy dataset/features mismatch"));
        }
        self.policy = Some(p);
        Ok(())
    }
    fn infer(&self, features: Vec<f32>) -> PyResult<[f32; 4]> {
        self.policy
            .as_ref()
            .ok_or_else(|| err("no policy loaded"))?
            .infer(&features)
            .map_err(err)
    }
}
impl Brain {
    fn validate(&self, ids: &[u32]) -> PyResult<()> {
        if ids.iter().any(|i| *i as usize >= self.inner.neuron_count()) {
            Err(err("neuron index out of range"))
        } else {
            Ok(())
        }
    }
}
#[pymodule]
fn _brain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Brain>()?;
    Ok(())
}
