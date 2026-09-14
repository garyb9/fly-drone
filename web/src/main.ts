import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import "./style.css";
import { THEME } from "./scene/theme";
import { applyRoom, createGlowDecal, type Room } from "./scene/world";

type Cell = { id: string; type: string; side: string; position: number[]; measured: boolean };
type Metadata = {
  ids: number[];
  cells: Cell[];
  links: number[][];
  neurons: number;
  features: number;
  policy: string;
  dataset_hash: string;
  tasks: string[];
  ablations: string[];
  room: Room;
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
};
// Without a flying decoder every motion command is zero: say so instead of looking stuck.
const HOLDING: Record<string, string> = {
  none: "DRONE HOLDING — no decoder loaded for this task",
  "limits mismatch":
    "DRONE HOLDING — no decoder trained for this task yet (loaded one is for another room)",
};
const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
<header><a class="brand" href="/">F<span>↗</span>D</a><div><h1>Fly / Drone</h1><p>CONNECTOME LABORATORY <span class="version">001</span></p></div><div class="connection"><i id="dot"></i><span id="status">Connecting to simulation</span></div></header>
<main><section class="card replay"><div class="panel-head"><span><b class="index">05</b> TRIALS &amp; REPLAY</span><span id="trial">SEED 42 · VISUAL · INTACT</span></div><div class="replay-grid"><div><h3>RUN A TRIAL</h3><div class="replay-form"><label>Task<select id="task"></select></label><label>Brain<select id="ablation"><option value="none">Intact</option><option value="zero">Zeroed features</option><option value="sensory">Vision silenced</option><option value="shuffle">Shuffled features</option></select></label><label>Seed<input id="seed" type="number" value="1000" min="0" step="1"></label><button id="run" class="primary">Run trial</button></div><p id="outcome" class="outcome">Outcome: —</p></div><div><h3>REPLAY AN EVALUATION</h3><div class="replay-form"><label>Report<select id="report"><option value="">No reports loaded</option></select></label></div><p id="report-summary" class="outcome"></p><div id="seeds" class="seed-grid" aria-label="Evaluation seeds"></div></div></div></section><section class="world card"><div class="panel-head"><span><b class="index">01</b> FLIGHT ENVIRONMENT</span><span id="mode">INITIALIZING</span></div><div id="world" class="viewport"></div><div class="world-overlay"><div class="eyebrow">MALECNS → CF2X</div><h2>A different body.<br>The same wiring.</h2><p>166,700 neurons · 4 rotors · 2 eyes</p></div><div class="world-bottom"><div><span>ALTITUDE</span><strong id="altitude">—<small> m</small></strong></div><div><span>SIMULATION</span><strong id="simtime">0.00<small> s</small></strong></div><div><span>REAL TIME</span><strong id="rtf">—<small> ×</small></strong></div><button id="follow">Follow drone</button></div><div class="world-frame"><i></i><i></i><i></i><i></i></div><div class="world-dims" id="dims"></div><div class="world-hint">DRAG TO ORBIT · SCROLL TO ZOOM</div></section>
<section class="card brain-card"><div class="panel-head"><span><b class="index">02</b> LIVING GRAPH</span><span id="tick">TICK 0</span></div><div id="brain" class="viewport"></div><div class="brain-legend"><span><i></i> measured activity</span><span>selected anatomical connections</span></div><div class="inspect"><select id="neuron" aria-label="Neuron to inspect"><option>Loading neurons…</option></select><div class="button-row"><button data-op="pulse">Pulse</button><button data-op="hold">Hold</button><button data-op="silence">Silence</button><button data-op="restore">Restore</button></div></div></section>
<section class="card fly-card"><div class="panel-head"><span><b class="index">03</b> PARALLEL BODY</span><span>FLY</span></div><div id="fly" class="viewport"></div><p class="caption">Same neural readouts. Independent trajectory.<br>Illustrative fly dynamics; not calibrated biomechanics.</p></section>
<section class="card signals"><div class="panel-head"><span><b class="index">04</b> SENSORY → NEURAL → MOTION</span><span id="episode">EPISODE 0</span></div><div class="signal-grid"><div class="eyes"><figure><img id="eye0" alt="Left simulated eye"><figcaption>LEFT EYE</figcaption></figure><figure><img id="eye1" alt="Right simulated eye"><figcaption>RIGHT EYE</figcaption></figure></div><div><h3>SENSORY CURRENT</h3><div id="cues" class="meters"></div></div><div><h3>NEURAL READOUT</h3><div id="readouts" class="meters"></div></div><div><h3>ACTUAL / COMMANDED RPM</h3><div id="motors" class="meters"></div></div></div></section>
<section class="card controls"><div><h3>EXPERIMENT CONTROLS</h3><div class="button-row"><button id="pause" class="primary">Pause</button><button id="reset">Reset trial</button></div></div><div><h3>VISUAL TARGET</h3><div class="button-row"><button data-target="left">Left</button><button data-target="center">Center</button><button data-target="right">Right</button></div></div><div><h3>OBSTACLE</h3><div class="button-row"><button id="loom">Place ahead</button><button id="clear">Move aside</button></div></div><div class="notes"><span id="command">Motion command: —</span><span id="error">Waiting for the local Rust + MuJoCo service.</span></div></section>
</main><footer><span>ANATOMICAL WIRING · MODELED NEURONS · LEARNED DECODING</span><span>MaleCNS v1.0 · FlyEM / Cambridge / MRC LMB / Google Research · CC-BY 4.0</span></footer>`;
const el = (id: string) => document.getElementById(id)!;
// Colors outside the blueprint-schematic scope: the brain graph's activity pulse and
// the fly viewport's background/grid keep their original values (fly stays untouched).
const ACTIVITY_HOT = 0x8be6d5;
const LEGACY_VIEWPORT_BG = 0x101c23;
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
const glowDecal = createGlowDecal(THEME.amber);
glowDecal.position.y = 0.002;
world.scene.add(glowDecal);
const mat = (c: number, metalness = 0.2) =>
  new THREE.MeshStandardMaterial({ color: c, metalness, roughness: 0.45 });
const drone = new THREE.Group();
world.scene.add(drone);
function box(w: number, h: number, d: number, m: THREE.Material, x = 0, y = 0, z = 0) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  mesh.position.set(x, y, z);
  return mesh;
}
// Drawn at physical scale so visible clearance to the obstacle matches the collision model.
const body = new THREE.Group();
drone.add(body);
// Flat ground marker keeps the small airframe easy to find; it does not tilt with the body.
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
body.add(box(0.055, 0.045, 0.009, mat(0x33484d), 0, 0, 0.005));
body.add(box(0.032, 0.022, 0.014, mat(0xb9c5bd), 0, 0, 0.017));
body.add(box(0.015, 0.02, 0.003, mat(THEME.amber), 0.02, 0, 0.011));
const rotors: THREE.Group[] = [];
const motorXY = [
  [1, 1],
  [-1, 1],
  [-1, -1],
  [1, -1],
].map(([x, y]) => [(x * 0.0397) / Math.sqrt(2), (y * 0.0397) / Math.sqrt(2)]);
for (const [i, [x, y]] of motorXY.entries()) {
  const arm = box(Math.hypot(x, y), 0.008, 0.005, mat(0x263f45), x / 2, y / 2, 0);
  arm.rotation.z = Math.atan2(y, x);
  body.add(arm);
  const motor = new THREE.Mesh(
    new THREE.CylinderGeometry(0.006, 0.006, 0.015, 12),
    mat(i < 2 ? THEME.line : 0x697c84),
  );
  motor.rotation.x = Math.PI / 2;
  motor.position.set(x, y, 0.004);
  body.add(motor);
  const rotor = new THREE.Group();
  rotor.position.set(x, y, 0.014);
  rotor.add(box(0.045, 0.004, 0.0015, mat(i % 2 === 0 ? THEME.line : THEME.amber)));
  rotors.push(rotor);
  body.add(rotor);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.022, 0.023, 40),
    new THREE.MeshBasicMaterial({
      color: i % 2 === 0 ? THEME.line : THEME.amber,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.2,
    }),
  );
  ring.position.set(x, y, 0.015);
  body.add(ring);
}
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
let socket: WebSocket,
  following = false,
  reports: Report[] = [],
  replayPolicy: string | undefined;
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
  el("mode").textContent =
    m.policy === "trained" ? "CONNECTOME POLICY" : "PID BASELINE · BRAIN OBSERVING";
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
function update(f: Frame) {
  latest = f;
  el("status").textContent = f.paused ? "Simulation paused" : "Local simulation connected";
  el("dot").classList.add("live");
  drone.position.copy(vector(f.state.position));
  locator.position.set(drone.position.x, 0.003, drone.position.z);
  glowDecal.position.set(drone.position.x, 0.002, drone.position.z);
  drone.quaternion.copy(rotation).multiply(quaternion(f.state.quaternion));
  rotors.forEach((r, i) => (r.rotation.z = f.state.rotor_phase[i]));
  target.position.copy(vector(f.state.target));
  obstacle.position.copy(vector(f.state.obstacle));
  const flyDelta = new THREE.Vector3().fromArray(f.fly.position).sub(fly.position);
  fly.position.fromArray(f.fly.position);
  fly.quaternion.copy(quaternion(f.fly.quaternion));
  flyview.camera.position.add(flyDelta);
  flyview.controls.target.copy(fly.position);
  for (const w of wings) {
    const power = f.readouts[w.sign === 1 ? "power_l" : "power_r"] ?? 0;
    w.node.rotation.x = w.rest + w.sign * Math.sin(f.time * 2 * Math.PI * 8) * power * 0.6;
  }
  if (following) {
    world.controls.target.copy(drone.position);
  }
  if (points && metadata) {
    const attr = points.geometry.getAttribute("color");
    const base = new THREE.Color(0x3b7483),
      hot = new THREE.Color(ACTIVITY_HOT);
    f.activity.forEach((v, i) => {
      const c = base.clone().lerp(hot, Math.min(1, v * 3));
      attr.setXYZ(i, c.r, c.g, c.b);
    });
    attr.needsUpdate = true;
    const lc = lines!.geometry.getAttribute("color");
    metadata.links.forEach(([a], i) => {
      const c = base.clone().lerp(hot, Math.min(1, f.activity[a] * 3));
      lc.setXYZ(i * 2, c.r, c.g, c.b);
      lc.setXYZ(i * 2 + 1, c.r, c.g, c.b);
    });
    lc.needsUpdate = true;
  }
  el("altitude").innerHTML = `${f.state.position[2].toFixed(2)}<small> m</small>`;
  el("simtime").innerHTML = `${f.time.toFixed(2)}<small> s</small>`;
  el("rtf").innerHTML = `${f.real_time_factor.toFixed(2)}<small> ×</small>`;
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
el("follow").onclick = () => {
  following = !following;
  el("follow").textContent = following ? "Free camera" : "Follow drone";
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
function render() {
  requestAnimationFrame(render);
  for (const v of [world, brain, flyview]) {
    v.controls.update();
    v.renderer.render(v.scene, v.camera);
  }
}
connect();
render();
