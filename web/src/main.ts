import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import "./style.css";
import { applyCssTokens } from "./theme/apply-css-tokens";
import { PALETTE, hexToInt } from "./theme/tokens";
import { THEME } from "./scene/theme";
import { applyRoom, createAxisGizmo, createGlowDecal, type Room } from "./scene/world";
import { buildDrone, type Airframe, type AirframeId } from "./scene/drone";
import { renderFlightView } from "./ui/flightView";
import { renderHudPinned } from "./ui/hudPinned";
import { renderDrawer } from "./ui/drawer";
import { renderFooter } from "./ui/footer";

applyCssTokens();

type Cell = {
  id: string;
  type: string;
  side: string;
  group?: number;
  position: number[];
  measured: boolean;
};
type Metadata = {
  ids: number[];
  cells: Cell[];
  links: number[][];
  groups: string[];
  neurons: number;
  features: number;
  policy: string;
  dataset_hash: string;
  tasks: string[];
  ablations: string[];
  room: Room;
  task_policy_status: Record<string, "loaded" | "none">;
};
type FreeRoamEvent = {
  type: string;
  time: number;
  side?: number;
  count?: number;
  kinds?: Record<string, number>;
  min_distance?: number;
  hit?: boolean;
  dodged?: boolean;
};
type FreeRoam = {
  level: number;
  beacons: number;
  collisions: number;
  collision_kinds: Record<string, number>;
  threats_finished: number;
  threats_dodged: number;
  threats_hit: number;
  visited_cells: number;
  beacon_visible: boolean;
  clearance: number;
  ghost: boolean;
  silenced: string[];
  events: FreeRoamEvent[];
};
type Outcome = {
  bearing: number;
  obstacle_distance: number;
  launched: boolean;
  displacement: number;
  result: { success: boolean; terminated: boolean; collision: boolean } | null;
};
type ReportRun = { seed: number; success: boolean; side: string };
type Report = {
  path: string;
  policy: string;
  task: string;
  seconds: number | null;
  acceptance: Record<string, boolean | number>;
  modes: Record<string, ReportRun[]>;
};
type Frame = {
  seq: number;
  episode: number;
  tick: number;
  physics_tick: number;
  time: number;
  paused: boolean;
  state: {
    position: number[];
    quaternion: number[];
    velocity: number[];
    angular_velocity?: number[];
    actual_rpm: number[];
    commanded_rpm: number[];
    rotor_phase: number[];
    target: number[];
    obstacle: number[];
  };
  fly: { position: number[]; quaternion: number[]; ticks: number };
  activity: number[];
  readouts: Record<string, number>;
  cues: number[];
  command: number[];
  cameras: string[];
  real_time_factor: number;
  missed_deadlines: number;
  error?: string;
  task: string;
  ablation: string;
  seed: number;
  active_policy: string | null;
  policy_status?: "none" | "loaded" | "limits mismatch";
  outcome: Outcome;
  free_roam: FreeRoam | null;
};
// Without a flying decoder every motion command is zero: say so instead of looking stuck.
const HOLDING: Record<string, string> = {
  none: "DRONE HOLDING — no decoder loaded for this task",
  "limits mismatch":
    "DRONE HOLDING — no decoder trained for this task yet (loaded one is for another room)",
};
const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
<main>
${renderFlightView()}
<div class="hud-layer">
${renderHudPinned()}
${renderDrawer()}
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
let airframeId: AirframeId = localStorage.getItem(AIRFRAME_KEY) === "B" ? "B" : "A";
let airframe: Airframe = buildDrone(airframeId);
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
  airframe = buildDrone(id);
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
  }),
);
world.scene.add(target);
const obstacle = new THREE.Mesh(
  new THREE.SphereGeometry(OBSTACLE_BASE_RADIUS, 24, 16),
  mat(THEME.obstacleStroke),
);
world.scene.add(obstacle);
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
// Lists the anatomical groups present among the rendered cells, biggest first. Counts are
// of rendered (subset) cells, not the full 166,700, so the legend is indicative, not a census.
function renderGroupLegend(m: Metadata) {
  const host = document.getElementById("group-legend");
  if (!host) return;
  const present = new Map<number, number>();
  for (const c of m.cells) present.set(c.group ?? 0, (present.get(c.group ?? 0) ?? 0) + 1);
  host.innerHTML = [...present.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([g, n]) => {
      const name = (m.groups[g] ?? `group ${g}`).replace(/_/g, " ");
      return `<span title="${n} rendered cells"><i style="background:#${groupColor(g).getHexString()}"></i>${name}</span>`;
    })
    .join("");
}
let socket: WebSocket,
  following = false,
  cameraMode: "orbit" | "fpv" | "tpv" = "orbit",
  reports: Report[] = [],
  replayPolicy: string | undefined,
  initialized = false;
const WORLD_HOME = { position: [2.5, 2.2, 3.2], target: [0.3, 0.8, 0] };
const rotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const vector = (v: number[]) => new THREE.Vector3(v[0], v[2], -v[1]);
const quaternion = (v: number[]) => new THREE.Quaternion(v[1], v[2], v[3], v[0]);
function meters(id: string, values: [string, number, string][], max = 1) {
  el(id).innerHTML = values
    .map(
      ([name, value, label]) =>
        `<div class="meter"><span>${name}</span><div><i style="width:${Math.min(100, Math.max(0, (value / max) * 100))}%"></i></div><em>${label}</em></div>`,
    )
    .join("");
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
  const select = el("neuron") as HTMLSelectElement;
  select.replaceChildren();
  m.cells.forEach((c, i) => {
    const option = document.createElement("option");
    option.value = String(m.ids[i]);
    option.textContent = `${c.type} · ${c.side} · ${c.id}`;
    select.append(option);
  });
  const taskSelect = el("task") as HTMLSelectElement;
  taskSelect.replaceChildren();
  m.tasks.forEach((t) => {
    const option = document.createElement("option");
    option.value = t;
    option.textContent = TASK_LABELS[t] ?? t;
    taskSelect.append(option);
  });
  // Land on free roam by default when a decoder is actually loaded for it (otherwise the
  // drone would just hold still there); only on first connect, not on every reconnect.
  if (!initialized && m.task_policy_status?.free_roam === "loaded") {
    taskSelect.value = "free_roam";
    runTrial();
  }
  initialized = true;
  el("mode").textContent =
    m.policy === "trained" ? "CONNECTOME POLICY" : "PID BASELINE · BRAIN OBSERVING";
  renderGroupLegend(m);
  updateRoom(m.room);
  void loadReports();
}
const select = (id: string) => el(id) as HTMLSelectElement;
const CONDITIONS: Record<string, string> = {
  none: "INTACT",
  zero: "ZEROED FEATURES",
  sensory: "VISION SILENCED",
  shuffle: "SHUFFLED FEATURES",
};
// Uppercase values feed the #trial summary directly; "visual"/"looming" must keep
// their exact existing strings — scripts/browser-check.mjs asserts on them literally.
const TASK_LABELS: Record<string, string> = {
  visual: "Steer to target",
  looming: "Dodge obstacle",
  approach: "Approach target",
  track: "Track orbiting target",
  steer_dodge: "Steer and dodge",
  escape: "Climb to escape",
  free_roam: "Free roam",
};
const TASK_TRIAL_LABEL: Record<string, string> = {
  visual: "STEER TO TARGET",
  looming: "DODGE OBSTACLE",
  approach: "APPROACH TARGET",
  track: "TRACK TARGET",
  steer_dodge: "STEER AND DODGE",
  escape: "CLIMB TO ESCAPE",
  free_roam: "FREE ROAM",
};
async function loadReports() {
  try {
    reports = await (await fetch("/api/reports")).json();
  } catch {
    reports = [];
  }
  const picker = select("report");
  picker.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = reports.length ? "Choose an evaluation report" : "No reports under runs/";
  picker.append(none);
  reports.forEach((r, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = `${r.task} · ${r.path}`;
    picker.append(option);
  });
  renderSeeds();
}
function renderSeeds() {
  const grid = el("seeds");
  grid.replaceChildren();
  const report = reports[Number(select("report").value)];
  if (!select("report").value || !report) {
    el("report-summary").textContent = "";
    replayPolicy = undefined;
    return;
  }
  replayPolicy = report.policy;
  select("task").value = report.task;
  const mode = select("ablation").value;
  const runs = report.modes[mode] ?? [];
  const passed = runs.filter((r) => r.success).length;
  el("report-summary").textContent =
    `${report.policy} · ${mode}: ${passed}/${runs.length} passed. Click a seed to replay it.`;
  for (const run of runs) {
    const button = document.createElement("button");
    button.className = run.success ? "seed pass" : "seed fail";
    button.textContent = String(run.seed);
    button.title = `${run.success ? "passed" : "failed"} · ${run.side ?? ""}`;
    button.onclick = () => {
      (el("seed") as HTMLInputElement).value = String(run.seed);
      runTrial();
    };
    grid.append(button);
  }
}
function runTrial() {
  const message: Record<string, unknown> = {
    op: "reset",
    seed: Number((el("seed") as HTMLInputElement).value) || 0,
    task: select("task").value,
    ablation: select("ablation").value,
  };
  if (replayPolicy) message.policy = replayPolicy;
  send(message);
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
function describeRoamEvent(e: FreeRoamEvent): string {
  const t = e.time.toFixed(1);
  switch (e.type) {
    case "beacon_collected":
      return `<p class="roam-log-entry ok">${t}s · beacon collected (${e.count})</p>`;
    case "collision":
      return `<p class="roam-log-entry warn">${t}s · collision: ${Object.keys(e.kinds ?? {}).join(", ") || "?"}</p>`;
    case "threat_launched":
      return `<p class="roam-log-entry warn">${t}s · threat launched (${(e.side ?? 0) > 0 ? "left" : "right"})</p>`;
    case "threat_hit":
      return `<p class="roam-log-entry warn">${t}s · threat hit</p>`;
    case "threat_passed":
      return `<p class="roam-log-entry ok">${t}s · threat dodged</p>`;
    default:
      return `<p class="roam-log-entry">${t}s · ${e.type}</p>`;
  }
}
function updateRoamHud(f: Frame) {
  const hud = el("roam-hud");
  const r = f.free_roam;
  el("roam-tab").hidden = !r;
  if (!r) {
    hud.hidden = true;
    return;
  }
  // Only the active tab's panel should be visible — updateRoamHud runs every frame
  // regardless of which tab is selected, so it must defer to switchTab's choice
  // instead of force-showing itself over whatever panel is actually open.
  const activeTab = document.querySelector<HTMLButtonElement>(".tab-btn.active")?.dataset.tab;
  hud.hidden = activeTab !== "roam";
  const minutes = Math.max(f.time / 60, 1 / 60);
  el("roam-level").textContent = `LEVEL ${r.level}`;
  el("roam-beacons").textContent = (r.beacons / minutes).toFixed(2);
  el("roam-collisions").textContent = (r.collisions / minutes).toFixed(2);
  el("roam-dodged").textContent = String(r.threats_dodged);
  el("roam-hit").textContent = String(r.threats_hit);
  el("roam-cells").textContent = String(r.visited_cells);
  el("roam-log").innerHTML = r.events.slice().reverse().map(describeRoamEvent).join("");
}
function update(f: Frame) {
  latest = f;
  el("status").textContent = f.paused ? "Simulation paused" : "Local simulation connected";
  el("dot").classList.add("live");
  drone.position.copy(vector(f.state.position));
  locator.position.set(drone.position.x, 0.003, drone.position.z);
  glowDecal.position.set(drone.position.x, 0.002, drone.position.z);
  drone.quaternion.copy(rotation).multiply(quaternion(f.state.quaternion));
  airframe.rotors.forEach((r, i) => (r.rotation.z = f.state.rotor_phase[i]));
  updateDebugVectors(f);
  target.position.copy(vector(f.state.target));
  obstacle.position.copy(vector(f.state.obstacle));
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
  meters(
    "cues",
    f.cues.map((v, i) => [["light L", "light R", "loom L", "loom R"][i], v, v.toFixed(2)]),
    2,
  );
  meters(
    "readouts",
    ["power_l", "power_r", "steer_l", "steer_r"].map((k) => [
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
  el("command").textContent =
    `Motion [m/s, rad/s]: ${f.command.map((v) => v.toFixed(2)).join(" · ")}`;
  updateRoamHud(f);
  el("error").textContent =
    f.error ??
    `${f.missed_deadlines} missed frame deadlines · Full graph running · Fly panel uses modeled dynamics`;
  if (f.task) {
    el("trial").textContent =
      `SEED ${f.seed} · ${TASK_TRIAL_LABEL[f.task] ?? f.task.toUpperCase()} · ${CONDITIONS[f.ablation] ?? f.ablation.toUpperCase()}${f.policy_status === "loaded" && f.active_policy ? ` · ${f.active_policy}` : " · NO DECODER"}`;
    const holding = HOLDING[f.policy_status ?? (f.active_policy ? "loaded" : "none")];
    if (holding) {
      el("outcome").textContent = holding;
      el("outcome").className = "outcome fail";
      return;
    }
    const o = f.outcome;
    const live =
      f.task === "looming"
        ? `obstacle ${o.obstacle_distance.toFixed(2)} m${o.launched ? " (launched)" : ""} · drift ${o.displacement.toFixed(2)} m`
        : `target bearing ${o.bearing.toFixed(2)} rad`;
    const verdict = o.result
      ? o.result.success
        ? " · PASSED"
        : o.result.collision
          ? " · FAILED (collision)"
          : " · FAILED"
      : "";
    el("outcome").textContent = `Outcome: ${live}${verdict}`;
    el("outcome").className = o.result
      ? o.result.success
        ? "outcome pass"
        : "outcome fail"
      : "outcome";
  }
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
el("run").onclick = () => runTrial();
select("report").onchange = () => renderSeeds();
select("ablation").onchange = () => renderSeeds();
function setCameraMode(mode: "orbit" | "fpv" | "tpv") {
  cameraMode = cameraMode === mode ? "orbit" : mode;
  world.controls.enabled = cameraMode === "orbit";
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
document
  .querySelectorAll<HTMLButtonElement>("[data-op]")
  .forEach(
    (b) =>
      (b.onclick = () =>
        send({ op: b.dataset.op, index: Number((el("neuron") as HTMLSelectElement).value) })),
  );
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
function setDrawerCollapsed(collapsed: boolean) {
  el("drawer").classList.toggle("collapsed", collapsed);
  el("drawer-collapse").querySelector("i")!.textContent = collapsed ? "›" : "‹";
}
function switchTab(name: string) {
  document
    .querySelectorAll<HTMLElement>(".drawer-body [data-panel]")
    .forEach((panel) => (panel.hidden = panel.dataset.panel !== name));
  document
    .querySelectorAll<HTMLButtonElement>(".tab-btn")
    .forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === name));
  setDrawerCollapsed(false);
}
document.querySelectorAll<HTMLButtonElement>(".tab-btn").forEach((btn) => {
  btn.onclick = () => {
    if (btn.classList.contains("active"))
      setDrawerCollapsed(!el("drawer").classList.contains("collapsed"));
    else switchTab(btn.dataset.tab!);
  };
});
el("drawer-collapse").onclick = () =>
  setDrawerCollapsed(!el("drawer").classList.contains("collapsed"));
function render() {
  requestAnimationFrame(render);
  for (const v of [world, brain, flyview]) {
    v.controls.update();
    v.renderer.render(v.scene, v.camera);
  }
}
connect();
render();
