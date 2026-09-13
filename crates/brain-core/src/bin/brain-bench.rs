//! Native full-graph benchmark, intentionally independent of Python and MuJoCo.
use brain_core::core::{
    format::{GraphFile, NeuronsFile},
    sim::{SimConfig, SimCore},
};
use std::{path::PathBuf, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/malecns".into()),
    );
    let start = Instant::now();
    let mut sim = {
        let nf = NeuronsFile::parse(&std::fs::read(root.join("neurons.bin"))?)
            .map_err(|e| format!("{e:?}"))?;
        let gf = GraphFile::parse(&std::fs::read(root.join("graph.bin"))?)
            .map_err(|e| format!("{e:?}"))?;
        SimCore::new(
            &nf,
            &gf,
            SimConfig {
                seed: 42,
                ..Default::default()
            },
        )
    };
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let mappings: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("sensory-mappings.json"))?)?;
    let ids: Vec<u32> = serde_json::from_value(mappings["inputs"]["light_l"].clone())?;
    let role = sim.define_input_role("light_l", &ids);
    let mut times = Vec::new();
    for _ in 0..400 {
        let t = Instant::now();
        sim.inject(role, 1.2);
        sim.step(1);
        times.push(t.elapsed().as_secs_f64() * 1000.);
    }
    times.sort_by(f64::total_cmp);
    let proc_status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss = proc_status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .unwrap_or("unavailable");
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"neurons":sim.neuron_count(),"ticks":400,"load_ms":load_ms,"tick_ms_p50":times[200],"tick_ms_p95":times[380],"rss":rss,"architecture":std::env::consts::ARCH,"note":"Native Rust only; desktop measurement, not onboard power or payload validation. Full graph, sustained left visual current, default seeded noise."})
        )?
    );
    Ok(())
}
