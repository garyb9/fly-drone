import * as THREE from "three";
import { els, setLeadingText, setText } from "../dom";
import type { Viewer } from "../app/context";
import { droneTarget, obstacleTarget, targetTarget } from "../app/motion";
import { setTabAvailable } from "../app/tabs";
import { quaternionInto, vector, vectorInto, WORLD_ROTATION } from "../scene/transform";
import { OBSTACLE_BASE_RADIUS, TARGET_BASE_RADIUS, THREAT_TRAIL_MAX } from "../scene/stage";
import type { Room } from "../scene/world";
import type { Frame, Metadata } from "../types";
import { setupBrainGraph, updateActivity } from "./brain";
import { formatRoamEvent } from "./format";
import { updateAttitudeGauges } from "./gauges";
import { meters, pushScope } from "./meters";

// Scratch objects for the per-frame path: converting a server pose allocates nothing.
const scratchVec = new THREE.Vector3();
const scratchQuat = new THREE.Quaternion();
const flyDelta = new THREE.Vector3();
// Two update rates: the smooth pose / camera / eye-feed path follows every server frame while
// the DOM text, meters and scopes only need to read at ~10 Hz. Keeping the node churn out of
// the frame path leaves the instrument just as readable.
const SLOW_INTERVAL_MS = 100;
let lastSlowAt = -Infinity;
let initialized = false;

const READOUT_KEYS = ["power_l", "power_r", "steer_l", "steer_r", "escape", "wing_l", "wing_r"];

export function updateRoom(viewer: Viewer, room: Room): void {
  const { stage } = viewer;
  stage.applyRoom(room);
  stage.target.scale.setScalar(room.beacon_radius / TARGET_BASE_RADIUS);
  stage.obstacle.scale.setScalar(room.threat_radius / OBSTACLE_BASE_RADIUS);
  const span = (room.half_size ?? 4) * 2;
  setText(els.dims, `${span.toFixed(1)} × ${span.toFixed(1)} M · ${room.kind.toUpperCase()}`);
  const half = room.half_size ?? 4;
  stage.axisGizmo.position.set(-(half + 1), 0.3, -(half + 1));
}

// Object layout (pillars, beacons, threats, arena spawns) is seeded by the reset seed, so the
// live viewer starts every run on a fresh seed.
const randomSeed = () => Math.floor(Math.random() * 1_000_000);

function renderRoamLevels(viewer: Viewer, levels: number[]): void {
  const host = els["roam-levels"];
  host.innerHTML = levels
    .map(
      (level) =>
        `<button data-level="${level}" title="Reset the arena at level ${level} with a fresh seed">L${level}</button>`,
    )
    .join("");
  host.querySelectorAll<HTMLButtonElement>("button").forEach((button) => {
    button.onclick = () =>
      viewer.send({ op: "reset", level: Number(button.dataset.level), seed: randomSeed() });
  });
}

export function setupBrain(viewer: Viewer, m: Metadata): void {
  const { stage } = viewer;
  setupBrainGraph(stage.brain, m);
  // Land on free roam by default when a decoder is actually loaded for it (otherwise the
  // drone would just hold still there). Once per browser session: a reload should attach to
  // the running simulation rather than restart it.
  if (!initialized && sessionStorage.getItem("fly-drone.landed") !== "1") {
    sessionStorage.setItem("fly-drone.landed", "1");
    if (m.task_policy_status?.free_roam === "loaded") {
      viewer.send({ op: "reset", task: "free_roam", seed: randomSeed() });
    }
  }
  initialized = true;
  setText(
    els.mode,
    m.policy === "trained" ? "CONNECTOME POLICY" : "PID BASELINE · BRAIN OBSERVING",
  );
  renderRoamLevels(viewer, m.levels);
  if (m.cameras) stage.setEyeGeometry({ splay: m.cameras.splay, fovyDeg: m.cameras.fovy_deg });
  updateRoom(viewer, m.room);
}

function updateRoamOverlays(viewer: Viewer, f: Frame): void {
  const { stage } = viewer;
  const roam = f.free_roam;
  const active = f.task === "free_roam" && roam !== null;
  els["ghost-banner"].hidden = !(active && roam.ghost);
  stage.beaconRing.visible = active;
  stage.clearanceRing.visible = active;
  if (active && roam) {
    stage.beaconRing.position.set(stage.target.position.x, 0.004, stage.target.position.z);
    (stage.beaconRing.material as THREE.MeshBasicMaterial).opacity = roam.beacon_visible
      ? 0.55
      : 0.12;
    stage.clearanceRing.position.set(stage.drone.position.x, 0.004, stage.drone.position.z);
    stage.clearanceRing.scale.setScalar(Math.max(0.05, roam.clearance));
    (stage.clearanceRing.material as THREE.MeshBasicMaterial).color.set(
      roam.clearance < 0.5 ? 0xffb15c : 0x6fe2ff,
    );
  }
  // The threat is parked at a far z when idle; only an in-arena position draws a trail.
  const threatActive = active && f.state.obstacle[2] < 3;
  if (threatActive) {
    const point = vector(f.state.obstacle);
    const last = stage.threatTrail[stage.threatTrail.length - 1];
    if (!last || last.distanceToSquared(point) > 1e-6) {
      stage.threatTrail.push(point);
      if (stage.threatTrail.length > THREAT_TRAIL_MAX) stage.threatTrail.shift();
    }
  } else if (stage.threatTrail.length) {
    stage.threatTrail.length = 0;
  }
  stage.threatTrailLine.visible = threatActive && stage.threatTrail.length > 1;
  if (stage.threatTrailLine.visible) {
    const attr = stage.threatTrailGeometry.getAttribute("position") as THREE.BufferAttribute;
    stage.threatTrail.forEach((point, i) => attr.setXYZ(i, point.x, point.y, point.z));
    attr.needsUpdate = true;
    stage.threatTrailGeometry.setDrawRange(0, stage.threatTrail.length);
  }
  (stage.target.material as THREE.MeshStandardMaterial).opacity = active && roam?.ghost ? 0.25 : 1;
  (stage.obstacle.material as THREE.MeshStandardMaterial).opacity =
    active && roam?.ghost ? 0.25 : 1;
}

function setPressed(button: HTMLButtonElement, on: boolean): void {
  button.classList.toggle("active", on);
  button.setAttribute("aria-pressed", String(on));
}

function updateFreeRoam(f: Frame): void {
  const roam = f.free_roam;
  const active = f.task === "free_roam" && roam !== null;
  setTabAvailable("roam", active);
  if (!active || !roam) return;
  setText(els["roam-beacons"], roam.beacons_per_min.toFixed(2));
  setText(els["roam-collisions"], roam.collisions_per_min.toFixed(2));
  setText(els["roam-threats"], `${roam.threats_dodged} / ${roam.threats_hit}`);
  setText(els["roam-coverage"], `${Math.round(roam.coverage * 100)}%`);
  setText(els["roam-clearance"], `${roam.clearance.toFixed(2)} m`);
  setText(els["roam-level"], `L${roam.level}`);
  setPressed(els["roam-silence-sensory"], roam.silenced.includes("sensory"));
  setPressed(els["roam-silence-light"], roam.silenced.includes("light"));
  setPressed(els["roam-silence-loom"], roam.silenced.includes("loom"));
  setPressed(els["roam-ghost"], roam.ghost);
  els["roam-levels"]
    .querySelectorAll<HTMLButtonElement>("button")
    .forEach((button) =>
      button.classList.toggle("active", Number(button.dataset.level) === roam.level),
    );
  setText(els["roam-visibility"], roam.beacon_visible ? "Beacon visible" : "Beacon out of view");
  const event = roam.events[roam.events.length - 1];
  setText(els["roam-event"], event ? formatRoamEvent(event) : "No free-roam events yet.");
}

// The one place that says why the drone is (or isn't) moving: none / limits mismatch both
// hold the command at zero, which otherwise just looks like a stuck aircraft.
function updateStatusStrip(f: Frame): void {
  const strip = els["status-strip"];
  const state = f.policy_status ?? "none";
  const parts: string[] = [];
  let severity: "ok" | "warn" | "error" = "ok";
  if (state === "loaded") {
    parts.push(`DECODER ${f.active_policy ?? "loaded"}`);
  } else if (state === "limits mismatch") {
    parts.push(`DECODER ${f.active_policy ?? "?"} · LIMITS MISMATCH — HOLDING`);
    severity = "error";
  } else {
    parts.push("DECODER none — HOLDING");
    severity = "warn";
  }
  parts.push(f.task.toUpperCase());
  if (f.free_roam) parts.push(`L${f.free_roam.level}`);
  parts.push(`SEED ${f.seed}`);
  parts.push(`RTF ${f.real_time_factor.toFixed(2)}×`);
  parts.push(`${f.missed_deadlines} missed`);
  setText(strip, parts.join("  ·  "));
  if (strip.dataset.state !== severity) strip.dataset.state = severity;
}

// Which decoder inputs push each command, as gradient × (feature − mean). Refreshed only when
// the server's attribution_seq changes (it costs ~90 ms, so it arrives a few times a second).
const ATTRIBUTION_CHANNELS = ["forward", "lateral", "climb", "yaw"];
let lastAttributionSeq = -1;
function updateAttribution(f: Frame): void {
  setTabAvailable("attribution", f.attribution !== null);
  if (!f.attribution || f.attribution_seq === lastAttributionSeq) return;
  lastAttributionSeq = f.attribution_seq;
  els.attribution.innerHTML = ATTRIBUTION_CHANNELS.map((channel) => {
    const entry = f.attribution!.channels[channel];
    if (!entry) return "";
    const peak = Math.max(1e-9, ...entry.types.map((t) => Math.abs(t.value)));
    const bars = entry.types
      .slice(0, 4)
      .map(
        (t) =>
          `<div class="attr-row" title="${t.type} ${t.value >= 0 ? "+" : ""}${t.value.toFixed(3)}"><span>${t.type.replace(/_/g, " ")}</span><div class="attr-track"><i class="${t.value >= 0 ? "pos" : "neg"}" style="width:${(Math.abs(t.value) / peak) * 100}%"></i></div><em>${t.value >= 0 ? "+" : ""}${t.value.toFixed(3)}</em></div>`,
      )
      .join("");
    return `<div class="attr-channel"><span class="attr-name">${channel}</span>${bars}<span class="attr-cells">cells ${entry.cells.slice(0, 4).join(" · ")}</span></div>`;
  }).join("");
}

function updateFast(viewer: Viewer, f: Frame): void {
  const { stage } = viewer;
  viewer.latest = f;
  els.dot.classList.add("live");
  vectorInto(droneTarget.pos, f.state.position);
  droneTarget.quat.copy(WORLD_ROTATION).multiply(quaternionInto(scratchQuat, f.state.quaternion));
  const airframe = stage.getAirframe();
  airframe.rotors.forEach((r, i) => (r.rotation.z = f.state.rotor_phase[i]));
  vectorInto(targetTarget, f.state.target);
  vectorInto(obstacleTarget, f.state.obstacle);
  scratchVec.fromArray(f.fly.position);
  flyDelta.subVectors(scratchVec, stage.fly.position);
  stage.fly.position.copy(scratchVec);
  stage.fly.quaternion.copy(quaternionInto(scratchQuat, f.fly.quaternion));
  stage.flyview.camera.position.add(flyDelta);
  stage.flyview.controls.target.copy(stage.fly.position);
  const powerL = f.readouts["power_l"] ?? 0,
    powerR = f.readouts["power_r"] ?? 0;
  // Beat frequency and amplitude follow wing power, so the shell visibly works harder when
  // the descending/motor neurons drive it harder. This is a cosmetic envelope, not a model.
  const beatAmp = 0.22 + 0.55 * Math.max(powerL, powerR);
  for (const w of stage.wings) {
    const power = w.sign === 1 ? powerL : powerR;
    const phase = f.time * 2 * Math.PI * (7 + 7 * power);
    w.node.rotation.x = w.rest + w.sign * beatAmp * (Math.sin(phase) + 0.2 * Math.sin(phase * 2));
  }
  updateActivity(f);
  updateAttitudeGauges(f);
  // The eye feed is the drone's own view, so it stays at frame rate.
  const eyeImgs = [els.eye0, els.eye1];
  f.cameras.forEach((c, i) => (eyeImgs[i].src = `data:image/jpeg;base64,${c}`));
  updateRoamOverlays(viewer, f);
}

function updateSlow(f: Frame): void {
  setText(els.status, f.paused ? "Simulation paused" : "Local simulation connected");
  setLeadingText(els.altitude, f.state.position[2].toFixed(2));
  setLeadingText(els.speed, Math.hypot(...f.state.velocity).toFixed(2));
  setLeadingText(els["velocity-axes"], f.state.velocity.map((v) => v.toFixed(1)).join(" / "));
  setLeadingText(els.simtime, f.time.toFixed(2));
  setLeadingText(els.rtf, f.real_time_factor.toFixed(2));
  setText(els.tick, `TICK ${f.tick.toLocaleString()}`);
  setText(els.episode, `EPISODE ${f.episode}`);
  setText(els.pause, f.paused ? "Resume" : "Pause");
  const sensory = f.sensory;
  const sensoryRows: [string, number, string][] = sensory
    ? Object.entries(sensory).map(([k, v]) => [k.replace(/_/g, " "), v, v.toFixed(3)])
    : f.cues.map((v, i) => [
        ["light L", "light R", "loom L", "loom R"][i] ?? `cue ${i}`,
        v,
        v.toFixed(2),
      ]);
  meters(els.cues, sensoryRows, 2);
  meters(
    els.readouts,
    READOUT_KEYS.map((k) => [
      k.replace("_", " "),
      f.readouts[k] ?? 0,
      (f.readouts[k] ?? 0).toFixed(3),
    ]),
  );
  meters(
    els.motors,
    f.state.actual_rpm.map((v, i) => [
      `M${i + 1}`,
      v,
      `${Math.round(v)} / ${Math.round(f.state.commanded_rpm[i])}`,
    ]),
    22000,
  );
  const budget = f.budget;
  const p50 = budget?.tick_ms_p50 ?? null;
  const p95 = budget?.tick_ms_p95 ?? null;
  const budgetTarget = budget?.target_ms ?? 5;
  meters(
    els.budget,
    p50 === null || p95 === null
      ? []
      : [
          ["tick p50", p50, `${p50.toFixed(2)} / ${budgetTarget.toFixed(0)} ms`],
          ["tick p95", p95, `${p95.toFixed(2)} / ${budgetTarget.toFixed(0)} ms`],
        ],
    budgetTarget,
  );
  setText(
    els["budget-note"],
    budget?.samples
      ? `${budget.samples} samples · ${f.real_time_factor.toFixed(2)}× real time · ${f.missed_deadlines} missed deadlines`
      : "No trained decoder loaded — nothing to time.",
  );
  // History behind the meters: one rolling scope per signal group.
  pushScope(
    els["cues-scope"],
    sensoryRows.map(([, value]) => value),
    { min: 0, max: 2 },
  );
  pushScope(
    els["readouts-scope"],
    READOUT_KEYS.map((k) => f.readouts[k] ?? 0),
  );
  pushScope(els["motors-scope"], f.state.actual_rpm, { min: 0, max: 22000 });
  if (p50 !== null && p95 !== null) {
    pushScope(els["budget-scope"], [p50, p95], { min: 0, target: budgetTarget });
  }
  setText(els.command, `Motion [m/s, rad/s]: ${f.command.map((v) => v.toFixed(2)).join(" · ")}`);
  setText(
    els.error,
    f.error ??
      `${f.missed_deadlines} missed frame deadlines · Full graph running · Fly panel uses modeled dynamics`,
  );
  updateFreeRoam(f);
  updateStatusStrip(f);
  updateAttribution(f);
}

export function updateFrame(viewer: Viewer, f: Frame): void {
  updateFast(viewer, f);
  const now = performance.now();
  if (now - lastSlowAt >= SLOW_INTERVAL_MS) {
    lastSlowAt = now;
    updateSlow(f);
  }
}
