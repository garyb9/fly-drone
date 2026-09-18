import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import "./style.css";
import { applyCssTokens } from "./theme/apply-css-tokens";
import { PALETTE, hexToInt } from "./theme/tokens";
import { THEME } from "./scene/theme";
import { applyRoom, createAxisGizmo, createGlowDecal, type Room } from "./scene/world";
import {
  buildDrone,
  DEFAULT_EYES,
  type Airframe,
  type AirframeId,
  type EyeGeometry,
} from "./scene/drone";
import { renderFlightView } from "./ui/flightView";
import { renderHudPinned } from "./ui/hudPinned";
import { renderBottomBar } from "./ui/bottomBar";
import { renderFooter } from "./ui/footer";
import { Scope } from "./charts/scope";
import type { Frame, FreeRoamEvent, Metadata } from "./types";

applyCssTokens();

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
<main>
${renderFlightView()}
<div class="hud-layer">
${renderHudPinned()}
${renderBottomBar()}
</div>
</main>
${renderFooter()}`;
function runBootSequence(): void {
  // Derived from the DOM (every .bp-boot element, in document order) rather than a
  // hardcoded selector list, so adding bp-boot to a new element can't silently leave
  // it out of the sequence (and stuck invisible) the way .instruments once did.
  const targets = Array.from(document.querySelectorAll<HTMLElement>(".bp-boot"));
  targets.forEach((target, i) => {
    setTimeout(() => {
      target.classList.add("bp-boot-run");
      // The boot animation sets transform:translateY(...) with animation-fill-mode:
      // forwards, which outranks any later author-set transform (e.g. the drawer's
      // open/collapsed state) in the cascade. Drop both classes once it finishes so
      // the animation's transform stops shadowing the element's real state.
      target.addEventListener(
        "animationend",
        () => target.classList.remove("bp-boot", "bp-boot-run"),
        { once: true },
      );
    }, i * 120);
  });
}
runBootSequence();
const el = (id: string) => document.getElementById(id)!;
// The brain-activity pulse and the fly viewport's background share the site's
// existing accent/surface tokens rather than introducing new colors.
const ACTIVITY_HOT = hexToInt(PALETTE.successTeal);
const LEGACY_VIEWPORT_BG = hexToInt(PALETTE.card);
function view(id: string, position: number[], target: number[]) {
  const host = el(id),
    scene = new THREE.Scene();
  scene.background = new THREE.Color(THEME.bg);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  camera.position.fromArray(position);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.fromArray(target);
  controls.enableDamping = true;
  controls.update();
  scene.add(new THREE.HemisphereLight(0xc2ebef, 0x24303a, 2));
  const light = new THREE.DirectionalLight(0xffefd8, 3);
  light.position.set(3, 7, 4);
  scene.add(light);
  new ResizeObserver(() => {
    const { width, height } = host.getBoundingClientRect();
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }).observe(host);
  return { scene, camera, renderer, controls };
}
const world = view("world", [2.5, 2.2, 3.2], [0.3, 0.8, 0]);
world.scene.fog = new THREE.Fog(THEME.bg, 7, 20);
const grid = new THREE.GridHelper(12, 48, THEME.grid, THEME.gridMinor);
world.scene.add(grid);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(18, 18),
  new THREE.MeshStandardMaterial({ color: THEME.wallFill, roughness: 0.95 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.005;
world.scene.add(floor);
// Rebuilt from the WebSocket "room" message (walls/bands/pillars) — see scene/world.ts.
// Empty until the first metadata/room message arrives.
const roomGroup = new THREE.Group();
world.scene.add(roomGroup);
// World-axis reference, placed just outside the arena's walls (repositioned in updateRoom() to
// track the current room's size) — a fixed frame to read the heading/velocity/commanded vectors
// against, since those vectors alone don't disambiguate "turning" from "actually moving".
const axisGizmo = createAxisGizmo(0.6);
world.scene.add(axisGizmo);
// Debug vectors from the drone's own position: nose heading (where it's pointed), measured
// velocity (finite difference of position, i.e. where it's actually going), and the raw command
// the decoder just issued (where it's trying to go). Divergence between these three is exactly
// the "life of its own" / "direction looks off" symptom this is meant to make visible.
const HEADING_COLOR = THEME.amber;
const VELOCITY_COLOR = THEME.velocity;
const COMMAND_COLOR = THEME.command;
const headingArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  0.3,
  HEADING_COLOR,
  0.08,
  0.05,
);
const velocityArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  0.001,
  VELOCITY_COLOR,
  0.08,
  0.05,
);
const commandArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  0.001,
  COMMAND_COLOR,
  0.08,
  0.05,
);
world.scene.add(headingArrow, velocityArrow, commandArrow);
const glowDecal = createGlowDecal(THEME.amber);
glowDecal.position.y = 0.002;
world.scene.add(glowDecal);
const mat = (c: number, metalness = 0.2) =>
  new THREE.MeshStandardMaterial({ color: c, metalness, roughness: 0.45 });
const drone = new THREE.Group();
world.scene.add(drone);
// Hardware airframe shell (A = 250 mm with the connectome on the ground, B = 450 mm with an
// onboard companion). Built in the drone's local Z-up frame; the flight plant is still the
// Crazyflie CF2X, so the shell is deliberately larger than the simulated collision body.
const AIRFRAME_KEY = "fly-drone.airframe";
const LANDED_KEY = "fly-drone.landed";
let airframeId: AirframeId = localStorage.getItem(AIRFRAME_KEY) === "B" ? "B" : "A";
let eyeGeometry: EyeGeometry = DEFAULT_EYES;
let airframe: Airframe = buildDrone(airframeId, eyeGeometry);
drone.add(airframe.group);
function setAirframe(id: AirframeId) {
  airframe.group.removeFromParent();
  airframe.group.traverse((n) => {
    const mesh = n as THREE.Mesh;
    mesh.geometry?.dispose();
    if (Array.isArray(mesh.material)) mesh.material.forEach((m) => m.dispose());
    else mesh.material?.dispose();
  });
  airframeId = id;
  airframe = buildDrone(id, eyeGeometry);
  airframe.fov.visible = el("toggle-fov").classList.contains("active");
  airframe.guards.visible = el("toggle-guards").classList.contains("active");
  drone.add(airframe.group);
  el("af-a").classList.toggle("active", id === "A");
  el("af-b").classList.toggle("active", id === "B");
  localStorage.setItem(AIRFRAME_KEY, id);
}
// Flat ground marker keeps the airframe easy to find; it does not tilt with the body.
const locator = new THREE.Mesh(
  new THREE.RingGeometry(0.1, 0.115, 48),
  new THREE.MeshBasicMaterial({
    color: THEME.amber,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
  }),
);
locator.rotation.x = -Math.PI / 2;
locator.position.y = 0.003;
world.scene.add(locator);
// Base radii match the legacy room's beacon_radius/threat_radius; other rooms (e.g.
// free_roam) report different values in the "room" message, applied as a uniform
// scale in setupBrain()/applyRoom() rather than rebuilding the sphere geometry.
const TARGET_BASE_RADIUS = 0.22;
const OBSTACLE_BASE_RADIUS = 0.25;
const target = new THREE.Mesh(
  new THREE.SphereGeometry(TARGET_BASE_RADIUS, 24, 16),
  new THREE.MeshStandardMaterial({
    color: THEME.amber,
    emissive: THEME.targetEmissive,
    emissiveIntensity: 0.4,
    transparent: true,
  }),
);
world.scene.add(target);
// Marks whether the beacon is geometrically visible to the drone (free-roam frames carry
// beacon_visible). Positioned on the beacon each frame, at a fixed metric radius.
const beaconRing = new THREE.Mesh(
  new THREE.RingGeometry(0.42, 0.5, 48),
  new THREE.MeshBasicMaterial({
    color: THEME.amber,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
  }),
);
beaconRing.rotation.x = -Math.PI / 2;
beaconRing.visible = false;
world.scene.add(beaconRing);
const obstacle = new THREE.Mesh(
  new THREE.SphereGeometry(OBSTACLE_BASE_RADIUS, 24, 16),
  mat(THEME.obstacleStroke),
);
(obstacle.material as THREE.MeshStandardMaterial).transparent = true;
world.scene.add(obstacle);
// Surface-distance halo from free_roam.clearance: reads as "how much room is left" at a glance.
const clearanceRing = new THREE.Mesh(
  new THREE.RingGeometry(0.97, 1.0, 48),
  new THREE.MeshBasicMaterial({
    color: THEME.velocity,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  }),
);
clearanceRing.rotation.x = -Math.PI / 2;
clearanceRing.visible = false;
world.scene.add(clearanceRing);
// The threat's recent path, sampled from obstacle positions (the frame carries no velocity).
const threatTrail: THREE.Vector3[] = [];
const THREAT_TRAIL_MAX = 60;
const threatTrailGeometry = new THREE.BufferGeometry();
threatTrailGeometry.setAttribute(
  "position",
  new THREE.Float32BufferAttribute(new Float32Array(THREAT_TRAIL_MAX * 3), 3),
);
const threatTrailLine = new THREE.Line(
  threatTrailGeometry,
  new THREE.LineBasicMaterial({ color: THEME.command, transparent: true, opacity: 0.6 }),
);
threatTrailLine.frustumCulled = false;
threatTrailLine.visible = false;
world.scene.add(threatTrailLine);
const brain = view("brain", [0, 0, 4], [0, 0, 0]);
brain.scene.background = new THREE.Color(0x0d171e);
const flyview = view("fly", [4, 2.5, 4], [0, 1, 0]);
flyview.scene.background = new THREE.Color(LEGACY_VIEWPORT_BG);
flyview.scene.add(new THREE.GridHelper(6, 12, 0x35535b, 0x20313a));
// Cool rim light separates the dark chitin from the dark background; the shared key light
// alone left the body reading as a silhouette.
const flyRim = new THREE.DirectionalLight(0x6fe2ff, 1.4);
flyRim.position.set(-3, 2.5, -3);
flyview.scene.add(flyRim);
const wings: { node: THREE.Object3D; rest: number; sign: number }[] = [];
const fly = new THREE.Group();
flyview.scene.add(fly);
new GLTFLoader().load(
  "/fly.glb",
  (g) => {
    const b = new THREE.Box3().setFromObject(g.scene),
      size = b.getSize(new THREE.Vector3());
    const center = b.getCenter(new THREE.Vector3());
    g.scene.position.sub(center);
    const model = new THREE.Group();
    model.add(g.scene);
    model.scale.setScalar(0.9 / Math.max(size.x, size.y, size.z));
    fly.add(model);
    g.scene.traverse((n) => {
      const mesh = n as THREE.Mesh;
      if (mesh.isMesh) {
        const m = mesh.material as THREE.MeshStandardMaterial;
        if (m && "roughness" in m) {
          m.roughness = Math.max(m.roughness, 0.55);
          m.metalness = Math.min(m.metalness, 0.1);
        }
        if (n.name === "l_wing" || n.name === "r_wing") {
          m.transparent = true;
          m.opacity = 0.92;
          m.side = THREE.DoubleSide;
        }
      }
      if (n.name === "l_wing" || n.name === "r_wing")
        wings.push({ node: n, rest: n.rotation.x, sign: n.name === "l_wing" ? 1 : -1 });
    });
  },
  undefined,
  () => {
    fly.add(new THREE.Mesh(new THREE.SphereGeometry(0.15, 16, 16), mat(ACTIVITY_HOT)));
  },
);
let metadata: Metadata | undefined,
  latest: Frame | undefined,
  points: THREE.Points | undefined,
  lines: THREE.LineSegments | undefined;
// Per-rendered-cell anatomical group and its base colour; activity is lerped on top of
// this base each frame so the cloud reads as anatomy when quiet and as activity when hot.
let cellBaseColors: THREE.Color[] = [];
const tmpColor = new THREE.Color();
const GROUP_DEFAULT = new THREE.Color(0x3b7483);
function groupColor(index: number): THREE.Color {
  // Golden-angle hue stepping gives stable, well-separated colours across the 27 groups.
  return new THREE.Color().setHSL(((index * 137.508) % 360) / 360, 0.5, 0.62);
}
let socket: WebSocket,
  following = true,
  cameraMode: "orbit" | "fpv" | "tpv" = "orbit",
  initialized = false;
const WORLD_HOME = { position: [2.5, 2.2, 3.2], target: [0.3, 0.8, 0] };
const rotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const vector = (v: number[]) => new THREE.Vector3(v[0], v[2], -v[1]);
const quaternion = (v: number[]) => new THREE.Quaternion(v[1], v[2], v[3], v[0]);
// Server frames arrive at the sim's real-time rate (well under the display refresh), so the
// drone and moving objects are eased toward each new pose instead of snapping. A time-constant
// smoother behaves the same under variable frame rates and adds only ~60 ms of visible lag.
const droneTarget = { pos: new THREE.Vector3(), quat: new THREE.Quaternion() };
const targetTarget = new THREE.Vector3();
const obstacleTarget = new THREE.Vector3();
const POSE_TAU = 0.06;
// Meters are rebuilt only when the row count or scale changes; otherwise the cached nodes
// have their text/width patched in place, so 20 Hz telemetry stops thrashing innerHTML.
type MeterRow = { name: HTMLSpanElement; bar: HTMLElement; value: HTMLElement };
const meterCache = new Map<
  string,
  { max: number; rows: MeterRow[] }
>();
function meters(id: string, values: [string, number, string][], max = 1) {
  const host = el(id);
  let cache = meterCache.get(id);
  if (!cache || cache.rows.length !== values.length || cache.max !== max) {
    host.innerHTML = values
      .map(() => `<div class="meter"><span></span><div><i></i></div><em></em></div>`)
      .join("");
    cache = {
      max,
      rows: Array.from(host.querySelectorAll<HTMLElement>(".meter")).map((root) => ({
        name: root.querySelector("span")!,
        bar: root.querySelector("i")!,
        value: root.querySelector("em")!,
      })),
    };
    meterCache.set(id, cache);
  }
  values.forEach(([name, value, label], i) => {
    const row = cache!.rows[i];
    row.name.textContent = name;
    row.bar.style.width = `${Math.min(100, Math.max(0, (value / max) * 100))}%`;
    row.value.textContent = label;
  });
}
// Live history for the same signals the meters show: max=Infinity keeps the node; a series
// count change (e.g. encoder v4's 4 cues vs v5/v6's population groups) rebuilds the scope.
const scopeCache = new Map<string, { canvas: HTMLCanvasElement; scope: Scope }>();
const SCOPE_COLORS = [
  PALETTE.successTeal,
  PALETTE.velocity,
  PALETTE.amber,
  PALETTE.command,
  PALETTE.axisX,
  PALETTE.axisY,
  PALETTE.axisZ,
];
function pushScope(
  id: string,
  values: number[],
  options: { min?: number; max?: number; target?: number } = {},
) {
  const canvas = el(id) as HTMLCanvasElement;
  let entry = scopeCache.get(id);
  if (!entry || entry.scope.series !== values.length) {
    entry = {
      canvas,
      scope: new Scope(values.length, 300, {
        colors: values.map((_, i) => SCOPE_COLORS[i % SCOPE_COLORS.length]),
        ...options,
      }),
    };
    scopeCache.set(id, entry);
  }
  entry.scope.push(values);
}
function drawScopes() {
  for (const { canvas, scope } of scopeCache.values()) {
    const dpr = Math.min(devicePixelRatio, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (ctx) scope.draw(ctx, width, height);
  }
}
function updateRoom(room: Room) {
  applyRoom(roomGroup, room, vector);
  target.scale.setScalar(room.beacon_radius / TARGET_BASE_RADIUS);
  obstacle.scale.setScalar(room.threat_radius / OBSTACLE_BASE_RADIUS);
  const span = (room.half_size ?? 4) * 2;
  el("dims").textContent = `${span.toFixed(1)} × ${span.toFixed(1)} M · ${room.kind.toUpperCase()}`;
  const half = room.half_size ?? 4;
  axisGizmo.position.set(-(half + 1), 0.3, -(half + 1));
}
function setupBrain(m: Metadata) {
  metadata = m;
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
  // Land on free roam by default when a decoder is actually loaded for it (otherwise the
  // drone would just hold still there). Once per browser session: a reload should attach to
  // the running simulation (same episode, tick continues) rather than restart it.
  if (!initialized && sessionStorage.getItem(LANDED_KEY) !== "1") {
    sessionStorage.setItem(LANDED_KEY, "1");
    if (m.task_policy_status?.free_roam === "loaded") {
      send({ op: "reset", task: "free_roam", seed: randomSeed() });
    }
  }
  initialized = true;
  el("mode").textContent =
    m.policy === "trained" ? "CONNECTOME POLICY" : "PID BASELINE · BRAIN OBSERVING";
  renderRoamLevels(m.levels);
  if (
    m.cameras &&
    (m.cameras.splay !== eyeGeometry.splay || m.cameras.fovy_deg !== eyeGeometry.fovyDeg)
  ) {
    eyeGeometry = { splay: m.cameras.splay, fovyDeg: m.cameras.fovy_deg };
    setAirframe(airframeId);
  }
  updateRoom(m.room);
}
// Object layout (pillars, beacons, threats, arena spawns) is seeded by the reset seed. The
// live viewer starts every run on a fresh seed so consecutive runs differ.
const randomSeed = () => Math.floor(Math.random() * 1_000_000);
function runTrial() {
  // No task/ablation selector any more: reset the server's current task with a new seed.
  send({ op: "reset", seed: randomSeed() });
}
// Exaggerates one second of travel so low speeds (a few tenths of a m/s) are still visible as
// short arrows rather than points, while clamping so a fast dodge doesn't dwarf the arena.
const VECTOR_SCALE = 1.2;
const MIN_ARROW_LEN = 0.12;
const MAX_ARROW_LEN = 1.0;
function scaledArrowLength(magnitude: number): number {
  return Math.min(MAX_ARROW_LEN, Math.max(MIN_ARROW_LEN, magnitude * VECTOR_SCALE));
}
function setArrow(
  arrow: THREE.ArrowHelper,
  position: THREE.Vector3,
  direction: THREE.Vector3,
  magnitude: number,
) {
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
// Roll/pitch/yaw in degrees from the MuJoCo world-frame quaternion (w, x, y, z), standard
// aerospace convention. Shared by the debug command-arrow rotation and the attitude gauges.
function attitudeFromQuaternion(q: number[]): { roll: number; pitch: number; yaw: number } {
  const [w, x, y, z] = q;
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const deg = (r: number) => (r * 180) / Math.PI;
  return { roll: deg(roll), pitch: deg(pitch), yaw: deg(yaw) };
}
// Draws where the nose points (heading), where the drone is actually going (measured velocity,
// straight from the plant's own world-frame velocity), and what the decoder just commanded
// (rotated into world frame the same way plant.advance() does). The three should mostly agree
// for a well-behaved brain; persistent divergence is exactly the "life of its own" symptom.
function updateDebugVectors(f: Frame) {
  const pos = drone.position;
  headingArrow.position.copy(pos);
  headingArrow.visible = true;
  headingArrow.setDirection(new THREE.Vector3(1, 0, 0).applyQuaternion(drone.quaternion));
  headingArrow.setLength(0.3, 0.08, 0.05);

  setArrow(velocityArrow, pos, vector(f.state.velocity), Math.hypot(...f.state.velocity));

  const yaw = attitudeFromQuaternion(f.state.quaternion).yaw * (Math.PI / 180);
  const cosY = Math.cos(yaw),
    sinY = Math.sin(yaw);
  const worldCmd = [
    cosY * f.command[0] - sinY * f.command[1],
    sinY * f.command[0] + cosY * f.command[1],
    f.command[2],
  ];
  setArrow(commandArrow, pos, vector(worldCmd), Math.hypot(...worldCmd));
}
// Clamp range for the vertical gauges: past this many degrees the needle/fill just pins at
// the extreme rather than continuing to move, so small dials stay legible at any attitude.
const GAUGE_CLAMP_DEG = 60;
function setGauge(prefix: string, angleDeg: number, mode: "rotate" | "translate") {
  const clamped = Math.max(-GAUGE_CLAMP_DEG, Math.min(GAUGE_CLAMP_DEG, angleDeg));
  const needle = el(`${prefix}-needle`);
  needle.style.transform =
    mode === "rotate"
      ? `rotate(${clamped}deg)`
      : `translateY(${(-clamped / GAUGE_CLAMP_DEG) * 20}px)`;
  el(`${prefix}-fill`).style.height = `${50 + (clamped / GAUGE_CLAMP_DEG) * 50}%`;
  el(`${prefix}-value`).textContent = `${angleDeg.toFixed(0)}°`;
}
function updateAttitudeGauges(f: Frame) {
  const { roll, pitch, yaw } = attitudeFromQuaternion(f.state.quaternion);
  setGauge("roll", roll, "rotate");
  setGauge("pitch", pitch, "translate");
  setGauge("yaw", yaw, "rotate");
  // Backend addition; degrade gracefully until an older server restarts to pick it up.
  const av = f.state.angular_velocity;
  el("turn-rate").textContent =
    av?.length === 3 ? `${((Math.hypot(...av) * 180) / Math.PI).toFixed(0)}°/s` : "—";
}
function setPressed(id: string, on: boolean) {
  const button = el(id);
  button.classList.toggle("active", on);
  button.setAttribute("aria-pressed", String(on));
}
function formatRoamEvent(event: FreeRoamEvent): string {
  const when = event.time !== undefined ? `t${event.time.toFixed(1)}s ` : "";
  switch (event.type) {
    case "beacon_collected":
      return `${when}beacon #${event.count} collected`;
    case "threat_launched":
      return `${when}threat launched (${(event.side ?? 0) > 0 ? "left" : "right"})`;
    case "threat_passed":
      return `${when}threat dodged · min ${event.min_distance?.toFixed(2) ?? "—"} m`;
    case "threat_hit":
      return `${when}threat hit`;
    case "collision":
      return `${when}collision: ${event.kinds?.join(", ") ?? ""}`;
    default:
      return `${when}${event.type.replace(/_/g, " ")}`;
  }
}
function renderRoamLevels(levels: number[]) {
  const host = el("roam-levels");
  host.innerHTML = levels
    .map(
      (level) =>
        `<button data-level="${level}" title="Reset the arena at level ${level} with a fresh seed">L${level}</button>`,
    )
    .join("");
  host.querySelectorAll<HTMLButtonElement>("button").forEach((button) => {
    button.onclick = () =>
      send({ op: "reset", level: Number(button.dataset.level), seed: randomSeed() });
  });
}
function updateRoamOverlays(f: Frame) {
  const roam = f.free_roam;
  const active = f.task === "free_roam" && roam !== null;
  el("ghost-banner").hidden = !(active && roam.ghost);
  beaconRing.visible = active;
  clearanceRing.visible = active;
  if (active && roam) {
    beaconRing.position.set(target.position.x, 0.004, target.position.z);
    (beaconRing.material as THREE.MeshBasicMaterial).opacity = roam.beacon_visible
      ? 0.55
      : 0.12;
    clearanceRing.position.set(drone.position.x, 0.004, drone.position.z);
    clearanceRing.scale.setScalar(Math.max(0.05, roam.clearance));
    (clearanceRing.material as THREE.MeshBasicMaterial).color.set(
      roam.clearance < 0.5 ? THEME.amber : THEME.velocity,
    );
  }
  // The threat is parked at a far z when idle; only an in-arena position draws a trail.
  const threatActive = active && f.state.obstacle[2] < 3;
  if (threatActive) {
    const point = vector(f.state.obstacle);
    const last = threatTrail[threatTrail.length - 1];
    if (!last || last.distanceToSquared(point) > 1e-6) {
      threatTrail.push(point);
      if (threatTrail.length > THREAT_TRAIL_MAX) threatTrail.shift();
    }
  } else if (threatTrail.length) {
    threatTrail.length = 0;
  }
  threatTrailLine.visible = threatActive && threatTrail.length > 1;
  if (threatTrailLine.visible) {
    const attr = threatTrailGeometry.getAttribute("position") as THREE.BufferAttribute;
    threatTrail.forEach((point, i) => attr.setXYZ(i, point.x, point.y, point.z));
    attr.needsUpdate = true;
    threatTrailGeometry.setDrawRange(0, threatTrail.length);
  }
  (target.material as THREE.MeshStandardMaterial).opacity =
    active && roam?.ghost ? 0.25 : 1;
  (obstacle.material as THREE.MeshStandardMaterial).opacity =
    active && roam?.ghost ? 0.25 : 1;
}
function updateFreeRoam(f: Frame) {
  const roam = f.free_roam;
  const active = f.task === "free_roam" && roam !== null;
  el("roam-group").hidden = !active;
  if (!active || !roam) return;
  el("roam-beacons").textContent = roam.beacons_per_min.toFixed(2);
  el("roam-collisions").textContent = roam.collisions_per_min.toFixed(2);
  el("roam-threats").textContent = `${roam.threats_dodged} / ${roam.threats_hit}`;
  el("roam-coverage").textContent = `${Math.round(roam.coverage * 100)}%`;
  el("roam-clearance").textContent = `${roam.clearance.toFixed(2)} m`;
  el("roam-level").textContent = `L${roam.level}`;
  const silenced = new Set(roam.silenced);
  setPressed("roam-silence-sensory", silenced.has("sensory"));
  setPressed("roam-silence-light", silenced.has("light"));
  setPressed("roam-silence-loom", silenced.has("loom"));
  setPressed("roam-ghost", roam.ghost);
  el("roam-levels")
    .querySelectorAll<HTMLButtonElement>("button")
    .forEach((button) =>
      button.classList.toggle("active", Number(button.dataset.level) === roam.level),
    );
  el("roam-visibility").textContent = roam.beacon_visible
    ? "Beacon visible"
    : "Beacon out of view";
  const event = roam.events[roam.events.length - 1];
  el("roam-event").textContent = event ? formatRoamEvent(event) : "No free-roam events yet.";
}
// The one place that says why the drone is (or isn't) moving: none / limits mismatch both
// hold the command at zero, which otherwise just looks like a stuck aircraft.
function updateStatusStrip(f: Frame) {
  const strip = el("status-strip");
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
  strip.textContent = parts.join("  ·  ");
  strip.dataset.state = severity;
}
// Which decoder inputs push each command, as gradient × (feature − mean). Refreshed only when
// the server's attribution_seq changes (it costs ~90 ms, so it arrives a few times a second).
const ATTRIBUTION_CHANNELS = ["forward", "lateral", "climb", "yaw"];
let lastAttributionSeq = -1;
function updateAttribution(f: Frame) {
  const group = el("attribution-group");
  group.hidden = f.attribution === null;
  if (!f.attribution || f.attribution_seq === lastAttributionSeq) return;
  lastAttributionSeq = f.attribution_seq;
  const host = el("attribution");
  host.innerHTML = ATTRIBUTION_CHANNELS.map((channel) => {
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
function update(f: Frame) {
  latest = f;
  el("status").textContent = f.paused ? "Simulation paused" : "Local simulation connected";
  el("dot").classList.add("live");
  droneTarget.pos.copy(vector(f.state.position));
  droneTarget.quat.copy(rotation).multiply(quaternion(f.state.quaternion));
  airframe.rotors.forEach((r, i) => (r.rotation.z = f.state.rotor_phase[i]));
  targetTarget.copy(vector(f.state.target));
  obstacleTarget.copy(vector(f.state.obstacle));
  const flyDelta = new THREE.Vector3().fromArray(f.fly.position).sub(fly.position);
  fly.position.fromArray(f.fly.position);
  fly.quaternion.copy(quaternion(f.fly.quaternion));
  flyview.camera.position.add(flyDelta);
  flyview.controls.target.copy(fly.position);
  const powerL = f.readouts["power_l"] ?? 0,
    powerR = f.readouts["power_r"] ?? 0;
  // Beat frequency and amplitude follow wing power, so the shell visibly works harder when
  // the descending/motor neurons drive it harder. This is a cosmetic envelope, not a
  // calibrated wingbeat model.
  const beatAmp = 0.22 + 0.55 * Math.max(powerL, powerR);
  for (const w of wings) {
    const power = w.sign === 1 ? powerL : powerR;
    const phase = f.time * 2 * Math.PI * (7 + 7 * power);
    w.node.rotation.x = w.rest + w.sign * beatAmp * (Math.sin(phase) + 0.2 * Math.sin(phase * 2));
  }
  if (points && metadata) {
    const attr = points.geometry.getAttribute("color");
    const hot = new THREE.Color(ACTIVITY_HOT);
    f.activity.forEach((v, i) => {
      const c = tmpColor.copy(cellBaseColors[i] ?? GROUP_DEFAULT).lerp(hot, Math.min(1, v * 3));
      attr.setXYZ(i, c.r, c.g, c.b);
    });
    attr.needsUpdate = true;
    const lc = lines!.geometry.getAttribute("color");
    metadata.links.forEach(([a], i) => {
      const c = tmpColor
        .copy(cellBaseColors[a] ?? GROUP_DEFAULT)
        .lerp(hot, Math.min(1, f.activity[a] * 3));
      lc.setXYZ(i * 2, c.r, c.g, c.b);
      lc.setXYZ(i * 2 + 1, c.r, c.g, c.b);
    });
    lc.needsUpdate = true;
  }
  el("altitude").innerHTML = `${f.state.position[2].toFixed(2)}<small> m</small>`;
  el("speed").innerHTML = `${Math.hypot(...f.state.velocity).toFixed(2)}<small> m/s</small>`;
  el("velocity-axes").innerHTML =
    `${f.state.velocity.map((v) => v.toFixed(1)).join(" / ")}<small> m/s</small>`;
  el("simtime").innerHTML = `${f.time.toFixed(2)}<small> s</small>`;
  el("rtf").innerHTML = `${f.real_time_factor.toFixed(2)}<small> ×</small>`;
  updateAttitudeGauges(f);
  el("tick").textContent = `TICK ${f.tick.toLocaleString()}`;
  el("episode").textContent = `EPISODE ${f.episode}`;
  el("pause").textContent = f.paused ? "Resume" : "Pause";
  f.cameras.forEach(
    (c, i) => ((el(`eye${i}`) as HTMLImageElement).src = `data:image/jpeg;base64,${c}`),
  );
  const sensory = f.sensory;
  const sensoryRows: [string, number, string][] = sensory
    ? Object.entries(sensory).map(([k, v]) => [k.replace(/_/g, " "), v, v.toFixed(3)])
    : f.cues.map((v, i) => [
        ["light L", "light R", "loom L", "loom R"][i] ?? `cue ${i}`,
        v,
        v.toFixed(2),
      ]);
  meters("cues", sensoryRows, 2);
  meters(
    "readouts",
    ["power_l", "power_r", "steer_l", "steer_r", "escape", "wing_l", "wing_r"].map((k) => [
      k.replace("_", " "),
      f.readouts[k] ?? 0,
      (f.readouts[k] ?? 0).toFixed(3),
    ]),
  );
  meters(
    "motors",
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
    "budget",
    p50 === null || p95 === null
      ? []
      : [
          ["tick p50", p50, `${p50.toFixed(2)} / ${budgetTarget.toFixed(0)} ms`],
          ["tick p95", p95, `${p95.toFixed(2)} / ${budgetTarget.toFixed(0)} ms`],
        ],
    budgetTarget,
  );
  el("budget-note").textContent = budget?.samples
    ? `${budget.samples} samples · ${f.real_time_factor.toFixed(2)}× real time · ${f.missed_deadlines} missed deadlines`
    : "No trained decoder loaded — nothing to time.";
  // History behind the meters: one rolling scope per signal group (15 s at 20 Hz).
  pushScope(
    "cues-scope",
    sensoryRows.map(([, value]) => value),
    { min: 0, max: 2 },
  );
  pushScope(
    "readouts-scope",
    ["power_l", "power_r", "steer_l", "steer_r", "escape", "wing_l", "wing_r"].map(
      (k) => f.readouts[k] ?? 0,
    ),
  );
  pushScope("motors-scope", f.state.actual_rpm, { min: 0, max: 22000 });
  if (p50 !== null && p95 !== null) {
    pushScope("budget-scope", [p50, p95], { min: 0, target: budgetTarget });
  }
  el("command").textContent =
    `Motion [m/s, rad/s]: ${f.command.map((v) => v.toFixed(2)).join(" · ")}`;
  el("error").textContent =
    f.error ??
    `${f.missed_deadlines} missed frame deadlines · Full graph running · Fly panel uses modeled dynamics`;
  updateFreeRoam(f);
  updateRoamOverlays(f);
  updateStatusStrip(f);
  updateAttribution(f);
}
function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.onmessage = (e) => {
    const message = JSON.parse(e.data);
    if (message.type === "metadata") setupBrain(message);
    else if (message.type === "frame") update(message);
    else if (message.type === "room") updateRoom(message);
    else if (message.error) el("error").textContent = message.error;
  };
  socket.onclose = () => {
    el("status").textContent = "Disconnected · reconnecting";
    el("dot").classList.remove("live");
    setTimeout(connect, 1500);
  };
  socket.onerror = () => {
    el("error").textContent = "Start the local service: fly-drone serve";
  };
}
function send(message: object) {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}
el("pause").onclick = () => send({ op: "pause", value: !latest?.paused });
el("reset").onclick = () => runTrial();
function setCameraMode(mode: "orbit" | "fpv" | "tpv") {
  cameraMode = cameraMode === mode ? "orbit" : mode;
  world.controls.enabled = cameraMode === "orbit";
  if (cameraMode !== "orbit") {
    // Auto-orbit drives the orbit camera; leaving it on would fight the canned FPV/TPV pose.
    world.controls.autoRotate = false;
    el("cam-orbit").classList.remove("active");
  }
  el("cam-fpv").classList.toggle("active", cameraMode === "fpv");
  el("cam-tpv").classList.toggle("active", cameraMode === "tpv");
}
el("cam-follow").onclick = () => {
  following = !following;
  el("cam-follow").classList.toggle("active", following);
};
el("cam-recenter").onclick = () => world.controls.target.copy(drone.position);
el("cam-reset").onclick = () => {
  cameraMode = "orbit";
  following = false;
  el("cam-follow").classList.remove("active");
  el("cam-fpv").classList.remove("active");
  el("cam-tpv").classList.remove("active");
  world.controls.enabled = true;
  world.camera.position.fromArray(WORLD_HOME.position);
  world.controls.target.fromArray(WORLD_HOME.target);
  world.controls.update();
};
el("cam-fpv").onclick = () => setCameraMode("fpv");
el("cam-tpv").onclick = () => setCameraMode("tpv");
el("af-a").onclick = () => setAirframe("A");
el("af-b").onclick = () => setAirframe("B");
el("toggle-fov").onclick = () => {
  const on = !el("toggle-fov").classList.contains("active");
  airframe.fov.visible = on;
  el("toggle-fov").classList.toggle("active", on);
};
el("toggle-guards").onclick = () => {
  const on = !el("toggle-guards").classList.contains("active");
  airframe.guards.visible = on;
  el("toggle-guards").classList.toggle("active", on);
};
el("eye-fx").onclick = () => {
  const on = !el("eye-fx").classList.contains("active");
  el("eye0").closest(".eyes")?.classList.toggle("fx", on);
  el("eye-fx").classList.toggle("active", on);
};
el("cam-orbit").onclick = () => {
  const on = !world.controls.autoRotate;
  world.controls.autoRotate = on;
  world.controls.autoRotateSpeed = 0.8;
  el("cam-orbit").classList.toggle("active", on);
};
document.querySelectorAll<HTMLButtonElement>("[data-target]").forEach(
  (b) =>
    (b.onclick = () =>
      send({
        op: "objects",
        target: [2, b.dataset.target === "left" ? 1.3 : b.dataset.target === "right" ? -1.3 : 0, 1],
      })),
);
el("loom").onclick = () => {
  const p = latest?.state.position ?? [0, 0, 1];
  send({ op: "objects", obstacle: [Math.min(3.5, p[0] + 0.65), p[1], Math.max(0.3, p[2])] });
};
el("clear").onclick = () => send({ op: "objects", obstacle: [2, -2, 1] });
// Free-roam probes. The server validates each one and reports any refusal through #error.
el("roam-beacon").onclick = () => {
  const f = latest;
  if (!f) return;
  const { yaw } = attitudeFromQuaternion(f.state.quaternion);
  const p = f.state.position;
  const distance = 1.5;
  send({
    op: "place_beacon",
    x: p[0] + Math.cos(yaw) * distance,
    y: p[1] + Math.sin(yaw) * distance,
  });
};
el("roam-threat").onclick = () => send({ op: "launch_threat" });
el("roam-silence-sensory").onclick = () =>
  send({
    op: "pathway",
    name: "sensory",
    silenced: !latest?.free_roam?.silenced.includes("sensory"),
  });
el("roam-silence-light").onclick = () =>
  send({
    op: "pathway",
    name: "light",
    silenced: !latest?.free_roam?.silenced.includes("light"),
  });
el("roam-silence-loom").onclick = () =>
  send({
    op: "pathway",
    name: "loom",
    silenced: !latest?.free_roam?.silenced.includes("loom"),
  });
el("roam-ghost").onclick = () =>
  send({ op: "ghost", value: !latest?.free_roam?.ghost });
el("roam-restore").onclick = () => send({ op: "restore" });
function applyCameraMode() {
  if (cameraMode === "fpv") {
    const cam = airframe.eyes[0];
    const eye = new THREE.Vector3(cam.x, 0, cam.z)
      .applyQuaternion(drone.quaternion)
      .add(drone.position);
    const ahead = new THREE.Vector3(1, 0, 0).applyQuaternion(drone.quaternion).add(eye);
    world.camera.position.copy(eye);
    world.camera.up.set(0, 1, 0);
    world.camera.lookAt(ahead);
    world.controls.target.copy(ahead);
  } else if (cameraMode === "tpv") {
    const chase = new THREE.Vector3(-0.7, 0.32, 0)
      .applyQuaternion(drone.quaternion)
      .add(drone.position);
    world.camera.position.copy(chase);
    world.camera.up.set(0, 1, 0);
    world.camera.lookAt(drone.position);
    world.controls.target.copy(drone.position);
  } else if (following) {
    world.controls.target.copy(drone.position);
  }
}
let lastRender = performance.now();
function render(now = performance.now()) {
  requestAnimationFrame(render);
  const dt = Math.min(0.1, (now - lastRender) / 1000);
  lastRender = now;
  const k = 1 - Math.exp(-dt / POSE_TAU);
  drone.position.lerp(droneTarget.pos, k);
  drone.quaternion.slerp(droneTarget.quat, k);
  // The beacon teleports when collected/placed; interpolating it made a caught ball appear
  // to be dragged to the next spot. Snap it; only the thrown threat glides.
  target.position.copy(targetTarget);
  obstacle.position.lerp(obstacleTarget, k);
  locator.position.set(drone.position.x, 0.003, drone.position.z);
  glowDecal.position.set(drone.position.x, 0.002, drone.position.z);
  if (latest) updateDebugVectors(latest);
  drawScopes();
  applyCameraMode();
  for (const v of [world, brain, flyview]) {
    v.controls.update();
    v.renderer.render(v.scene, v.camera);
  }
}
connect();
render();
