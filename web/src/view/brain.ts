import * as THREE from "three";
import { ACTIVITY_HOT_COLOR, type Viewport } from "../scene/stage";
import type { Frame, Metadata } from "../types";

// The connectome point cloud and its measured links. Group base colours are lerped toward the
// activity colour each frame so the cloud reads as anatomy when quiet and as activity when hot.

let cellBaseColors: THREE.Color[] = [];
let links: number[][] = [];
let points: THREE.Points | undefined;
let lines: THREE.LineSegments | undefined;
const tmpColor = new THREE.Color();
const GROUP_DEFAULT = new THREE.Color(0x3b7483);

function groupColor(index: number): THREE.Color {
  // Golden-angle hue stepping gives stable, well-separated colours across the 27 groups.
  return new THREE.Color().setHSL(((index * 137.508) % 360) / 360, 0.5, 0.62);
}

export function setupBrainGraph(brain: Viewport, m: Metadata): void {
  if (points) {
    brain.scene.remove(points);
    points.geometry.dispose();
    (points.material as THREE.Material).dispose();
  }
  if (lines) {
    brain.scene.remove(lines);
    lines.geometry.dispose();
    (lines.material as THREE.Material).dispose();
  }
  const positions = m.cells.flatMap((c) => c.position);
  cellBaseColors = m.cells.map((c) => groupColor(c.group ?? 0));
  links = m.links;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(new Float32Array(positions.length), 3),
  );
  points = new THREE.Points(
    geo,
    new THREE.PointsMaterial({ size: 0.019, vertexColors: true, transparent: true, opacity: 0.9 }),
  );
  brain.scene.add(points);

  const linePos = m.links.flatMap(([a, b]) => [...m.cells[a].position, ...m.cells[b].position]);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.Float32BufferAttribute(linePos, 3));
  lg.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(linePos.length), 3));
  lines = new THREE.LineSegments(
    lg,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.3 }),
  );
  brain.scene.add(lines);

  const bounds = new THREE.Box3().setFromBufferAttribute(
    geo.getAttribute("position") as THREE.BufferAttribute,
  );
  brain.controls.target.copy(bounds.getCenter(new THREE.Vector3()));
  brain.camera.position.copy(brain.controls.target).add(new THREE.Vector3(0, 0.2, 5));
  brain.controls.update();
}

export function updateActivity(f: Frame): void {
  if (!points || !lines) return;
  const attr = points.geometry.getAttribute("color");
  f.activity.forEach((v, i) => {
    const c = tmpColor
      .copy(cellBaseColors[i] ?? GROUP_DEFAULT)
      .lerp(ACTIVITY_HOT_COLOR, Math.min(1, v * 3));
    attr.setXYZ(i, c.r, c.g, c.b);
  });
  attr.needsUpdate = true;
  const lc = lines.geometry.getAttribute("color");
  links.forEach(([a], i) => {
    const c = tmpColor
      .copy(cellBaseColors[a] ?? GROUP_DEFAULT)
      .lerp(ACTIVITY_HOT_COLOR, Math.min(1, f.activity[a] * 3));
    lc.setXYZ(i * 2, c.r, c.g, c.b);
    lc.setXYZ(i * 2 + 1, c.r, c.g, c.b);
  });
  lc.needsUpdate = true;
}
