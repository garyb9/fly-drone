import * as THREE from "three";
import { els, setText } from "../dom";
import { vectorInto } from "../scene/transform";
import type { Frame } from "../types";
import { attitudeFromQuaternion, scaledArrowLength } from "./format";

// Clamp range for the vertical gauges: past this many degrees the needle/fill just pins at
// the extreme rather than continuing to move, so small dials stay legible at any attitude.
const GAUGE_CLAMP_DEG = 60;
const scratchVec = new THREE.Vector3();

export function setArrow(
  arrow: THREE.ArrowHelper,
  position: THREE.Vector3,
  direction: THREE.Vector3,
  magnitude: number,
): void {
  if (magnitude < 1e-4) {
    arrow.visible = false;
    return;
  }
  arrow.visible = true;
  arrow.position.copy(position);
  arrow.setDirection(direction.normalize());
  const len = scaledArrowLength(magnitude);
  arrow.setLength(len, Math.min(0.08, len * 0.3), Math.min(0.05, len * 0.2));
}

function setGauge(
  prefix: "roll" | "pitch" | "yaw",
  angleDeg: number,
  mode: "rotate" | "translate",
) {
  const clamped = Math.max(-GAUGE_CLAMP_DEG, Math.min(GAUGE_CLAMP_DEG, angleDeg));
  els[`${prefix}-needle`].style.transform =
    mode === "rotate"
      ? `rotate(${clamped}deg)`
      : `translateY(${(-clamped / GAUGE_CLAMP_DEG) * 20}px)`;
  els[`${prefix}-fill`].style.height = `${50 + (clamped / GAUGE_CLAMP_DEG) * 50}%`;
  setText(els[`${prefix}-value`], `${angleDeg.toFixed(0)}°`);
}

export function updateAttitudeGauges(f: Frame): void {
  const { roll, pitch, yaw } = attitudeFromQuaternion(f.state.quaternion);
  setGauge("roll", roll, "rotate");
  setGauge("pitch", pitch, "translate");
  setGauge("yaw", yaw, "rotate");
  // Backend addition; degrade gracefully until an older server restarts to pick it up.
  const av = f.state.angular_velocity;
  setText(
    els["turn-rate"],
    av?.length === 3 ? `${((Math.hypot(...av) * 180) / Math.PI).toFixed(0)}°/s` : "—",
  );
}

export type DebugArrows = {
  heading: THREE.ArrowHelper;
  velocity: THREE.ArrowHelper;
  command: THREE.ArrowHelper;
};

// Draws where the nose points, where the drone is actually going (plant velocity), and what
// the decoder just commanded (rotated into world frame like plant.advance). The three should
// mostly agree for a well-behaved brain; persistent divergence is the "life of its own" symptom.
export function updateDebugVectors(drone: THREE.Object3D, arrows: DebugArrows, f: Frame): void {
  const pos = drone.position;
  arrows.heading.position.copy(pos);
  arrows.heading.visible = true;
  scratchVec.set(1, 0, 0).applyQuaternion(drone.quaternion);
  arrows.heading.setDirection(scratchVec);
  arrows.heading.setLength(0.3, 0.08, 0.05);

  setArrow(
    arrows.velocity,
    pos,
    vectorInto(scratchVec, f.state.velocity),
    Math.hypot(...f.state.velocity),
  );

  const yaw = attitudeFromQuaternion(f.state.quaternion).yaw * (Math.PI / 180);
  const cosY = Math.cos(yaw),
    sinY = Math.sin(yaw);
  // Command in the MuJoCo world frame, mapped straight onto the scene axes (x, z, -y).
  scratchVec.set(
    cosY * f.command[0] - sinY * f.command[1],
    f.command[2],
    -(sinY * f.command[0] + cosY * f.command[1]),
  );
  setArrow(arrows.command, pos, scratchVec, Math.hypot(f.command[0], f.command[1], f.command[2]));
}
