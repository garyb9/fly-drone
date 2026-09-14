import * as THREE from "three";
import { THEME } from "./theme";

// Mirrors the WebSocket "room" schema (server.py, connectome-free-roam-arena branch):
// a room describes the current arena's walls/bands/pillars generically so the
// frontend never needs to hardcode wall counts/positions for legacy vs. free-roam.
// All coordinates are MuJoCo world, Z-up metres.
export type RoomBox = { center: number[]; half_extents: number[] };
export type RoomPillar = {
  center: number[];
  radius: number;
  height: number;
  // Metres; a thin near-black ring at eye height (world z 1.0), tuned so pillars read
  // as ~5-17% loom-cue frames instead of ~55-70% for a fully dark pillar. Null means
  // the whole pillar renders dark (no ring cut into it).
  ring_height: number | null;
};
export type Room = {
  kind: string;
  half_size: number | null;
  walls: RoomBox[];
  bands: RoomBox[];
  pillars: RoomPillar[];
  beacon_radius: number;
  threat_radius: number;
};

type ToScene = (v: number[]) => THREE.Vector3;

function disposeChild(obj: THREE.Object3D) {
  const mesh = obj as THREE.Mesh | THREE.LineSegments;
  mesh.geometry?.dispose();
  const material = mesh.material;
  if (Array.isArray(material)) material.forEach((m) => m.dispose());
  else material?.dispose();
}

function clearGroup(group: THREE.Group) {
  while (group.children.length) {
    disposeChild(group.children.pop()!);
  }
}

// A room box's half-extents are given in MuJoCo world axes [hx, hy, hz]; under the
// scene's [x, z, -y] axis mapping a box stays axis-aligned, so only the extent order
// changes (world z becomes scene y, world y becomes scene z) — no rotation needed.
function addBox(
  group: THREE.Group,
  spec: RoomBox,
  toScene: ToScene,
  fillColor: number,
  edgeColor: number,
  opacity: number,
) {
  const [hx, hy, hz] = spec.half_extents;
  const geo = new THREE.BoxGeometry(hx * 2, hz * 2, hy * 2);
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({
      color: fillColor,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
      roughness: 0.92,
      metalness: 0,
    }),
  );
  mesh.position.copy(toScene(spec.center));
  group.add(mesh);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({
      color: edgeColor,
      transparent: true,
      opacity: Math.min(1, opacity * 5),
    }),
  );
  edges.position.copy(mesh.position);
  group.add(edges);
}

// Eye-height ring center is a fixed world z, not derived from a pillar's own center
// (which sits at height/2) — matches the MuJoCo scene, not a per-pillar field.
const PILLAR_RING_CENTER_Z = 1.0;
// Grey (luma ~0.35) to match what the drone's own eyes render; a fully dark pillar
// (no ring) reads as a much stronger loom cue than the physical setup intends.
const PILLAR_GREY = 0x595959;
const PILLAR_DARK = 0x08090b;

// A pillar stands along world +Z, which maps to scene +Y — CylinderGeometry's default
// axis — so it also needs no rotation, just a position.
function addPillar(group: THREE.Group, spec: RoomPillar, toScene: ToScene) {
  const geo = new THREE.CylinderGeometry(spec.radius, spec.radius, spec.height, 20);
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({
      color: spec.ring_height === null ? PILLAR_DARK : PILLAR_GREY,
      roughness: 0.85,
      metalness: 0,
    }),
  );
  mesh.position.copy(toScene(spec.center));
  group.add(mesh);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: THEME.wall, transparent: true, opacity: 0.5 }),
  );
  edges.position.copy(mesh.position);
  group.add(edges);
  if (spec.ring_height !== null) {
    const ringGeo = new THREE.CylinderGeometry(
      spec.radius * 1.01,
      spec.radius * 1.01,
      spec.ring_height,
      20,
      1,
      true,
    );
    const ring = new THREE.Mesh(
      ringGeo,
      new THREE.MeshStandardMaterial({
        color: PILLAR_DARK,
        roughness: 0.9,
        side: THREE.DoubleSide,
      }),
    );
    ring.position.copy(toScene([spec.center[0], spec.center[1], PILLAR_RING_CENTER_Z]));
    group.add(ring);
  }
}

// Rebuilds the room group from scratch on every call — rooms change only on connect
// and on task reset (not per-frame), so this isn't a hot path.
export function applyRoom(group: THREE.Group, room: Room, toScene: ToScene) {
  clearGroup(group);
  for (const wall of room.walls) addBox(group, wall, toScene, THEME.wallFill, THEME.wall, 0.16);
  // Bands are the dark inner strips MuJoCo renders at eye height for looming contrast —
  // drawn darker/more opaque than the walls so they read as an inset band, not a wall.
  for (const band of room.bands) addBox(group, band, toScene, 0x030a0f, THEME.wall, 0.4);
  for (const pillar of room.pillars) addPillar(group, pillar, toScene);
}

// A soft radial glow that tracks the drone across the floor grid — the "reactive
// grid" decision. A canvas-texture decal is simpler and just as convincing as a
// custom shader for a single soft blob, and avoids a shader-compile failure mode.
export function createGlowDecal(color: number, size = 2.4): THREE.Mesh {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d")!;
  const c = new THREE.Color(color);
  const rgb = `${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)}`;
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, `rgba(${rgb},0.5)`);
  gradient.addColorStop(1, `rgba(${rgb},0)`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(canvas);
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}
