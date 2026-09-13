//! Shared constants + loaders for integration tests. Mirrors the "Fixture
//! contract" in the Plan 01 doc and pipeline/gen_fixture.py.
//!
//! `#[path]`-included by both `golden_trace.rs` and any future integration
//! test, so which items are "used" depends on the current test set. Today
//! `golden_trace.rs` consumes `FIXTURE_SEED`, `neurons_bytes`, `graph_bytes`
//! and the two `fixture_*_roles` helpers, but not `FIXTURE_N` / `FIXTURE_CORE`
//! (kept as the documented contract values for tests still to come), so the
//! module still needs the dead-code allow.
#![allow(dead_code)]

pub const FIXTURE_N: usize = 500;
pub const FIXTURE_CORE: u32 = 48;
pub const FIXTURE_SEED: u64 = 42;

fn read(name: &str) -> Vec<u8> {
    let p = format!(
        "{}/../../pipeline/out/fixture/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read(&p).unwrap_or_else(|_| panic!("missing {p}; run python pipeline/gen_fixture.py"))
}
pub fn neurons_bytes() -> Vec<u8> {
    read("neurons.bin")
}
pub fn graph_bytes() -> Vec<u8> {
    read("graph.bin")
}

pub fn fixture_input_roles() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("looming", (0..8).collect()),
        ("light_l", (8..12).collect()),
        ("light_r", (12..16).collect()),
        ("proximity", (16..20).collect()),
        ("wind_l", (20..22).collect()),
        ("wind_r", (22..24).collect()),
    ]
}
pub fn fixture_readout_roles() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("escape", (24..32).collect()),
        ("wing_l", (32..36).collect()),
        ("wing_r", (36..40).collect()),
        ("thrust", (40..44).collect()),
        ("yaw_torque", (44..48).collect()),
    ]
}
