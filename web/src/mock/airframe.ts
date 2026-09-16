import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { applyCssTokens } from "../theme/apply-css-tokens";
import { THEME } from "../scene/theme";

applyCssTokens();

type AirframeId = "A" | "B";
type EnvId = "blueprint" | "replica" | "textured";

// Exact sensory geometry from the simulated plant (python/fly_drone/plant.py), so the
// mock cannot drift from what the brain actually sees.
const EYE_SPLAY = 0.75;
const EYE_FOVY = (75 * Math.PI) / 180;

const EXTRA = {
  carbon: 0x1b2a30,
  plate: 0x2b3b42,
  motor: 0x697c84,
  metal: 0x9fb0b6,
  dark: 0x08090b,
  grey: 0x595959,
};

type AirframeSpec = {
  id: AirframeId;
  title: string;
  frameClass: string;
  diagonal: number;
  propR: number;
  body: [number, number, number];
  camX: number;
  camY: number;
  camSize: number;
  onboard: boolean;
  battery: [number, number, number];
  note: string;
};

const SPECS: Record<AirframeId, AirframeSpec> = {
  A: {
    id: "A",
    title: "Option A · brain on the ground",
    frameClass: "250 mm",
    diagonal: 0.25,
    propR: 0.0635,
    body: [0.1, 0.08, 0.004],
    camX: 0.045,
    camY: 0.012,
    camSize: 0.014,
    onboard: false,
    battery: [0.05, 0.035, 0.012],
    note: "2 cameras + flight controller + radio/video link. Desktop runs the connectome; velocity commands return over MAVLink. ~0.4–0.7 kg all-up.",
  },
  B: {
    id: "B",
    title: "Option B · onboard companion",
    frameClass: "450 mm / 7 in",
    diagonal: 0.45,
    propR: 0.089,
    body: [0.17, 0.13, 0.005],
    camX: 0.085,
    camY: 0.022,
    camSize: 0.02,
    onboard: true,
    battery: [0.09, 0.06, 0.02],
    note: "Adds a Jetson Orin-class companion computer and real-time scheduler next to the flight controller. ~0.8–1.5 kg, 8–15 min.",
  },
};

const mat = (color: number, opts: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, metalness: 0.3, roughness: 0.5, ...opts });

function box(w: number, h: number, d: number, m: THREE.Material, x = 0, y = 0, z = 0): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  mesh.position.set(x, y, z);
  return mesh;
}

function tube(r: number, h: number, m: THREE.Material, seg = 16): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), m);
  return mesh;
}

function fovCone(len: number, halfAngle: number, color: number): THREE.Group {
  const r = Math.tan(halfAngle) * len;
  const geo = new THREE.ConeGeometry(r, len, 40, 1, true);
  geo.translate(0, -len / 2, 0);
  geo.rotateZ(-Math.PI / 2);
  const group = new THREE.Group();
  group.add(
    new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    ),
  );
  group.add(
    new THREE.LineSegments(
      new THREE.EdgesGeometry(geo, 40),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 }),
    ),
  );
  return group;
}

const QUAD: [number, number][] = [
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

type Airframe = {
  group: THREE.Group;
  rotors: THREE.Group[];
  guards: THREE.Group;
  fov: THREE.Group;
  cameraEyes: THREE.Vector3[];
};

function buildAirframe(spec: AirframeSpec): Airframe {
  const group = new THREE.Group();
  const rotors: THREE.Group[] = [];
  const guards = new THREE.Group();
  const fov = new THREE.Group();
  const cameraEyes: THREE.Vector3[] = [];
  const half = spec.diagonal / 2;
  const [bw, bd, bh] = spec.body;
  const z0 = 0.02;

  group.add(box(bw, bd, bh, mat(EXTRA.carbon), 0, 0, z0));
  group.add(box(bw * 0.82, bd * 0.82, bh, mat(EXTRA.plate), 0, 0, z0 + 0.018));
  const [batW, batD, batH] = spec.battery;
  group.add(box(batW, batD, batH, mat(0x22343b), 0, 0, z0 + 0.018 + batH / 2));

  for (const [dx, dy] of QUAD) {
    const mx = (dx * half) / Math.sqrt(2);
    const my = (dy * half) / Math.sqrt(2);
    const angle = Math.atan2(my, mx);
    const armLen = Math.hypot(mx, my);

    const arm = box(armLen, spec.diagonal * 0.05, 0.005, mat(EXTRA.carbon), mx / 2, my / 2, z0);
    arm.rotation.z = angle;
    group.add(arm);

    const motorR = spec.diagonal * 0.05;
    const motor = tube(motorR, 0.018, mat(EXTRA.motor), 18);
    motor.rotation.x = Math.PI / 2;
    motor.position.set(mx, my, z0 + 0.009);
    group.add(motor);

    const rotor = new THREE.Group();
    rotor.position.set(mx, my, z0 + 0.022);
    rotor.add(box(spec.propR * 2, 0.006, 0.0014, mat(EXTRA.dark)));
    rotor.add(box(0.006, spec.propR * 2, 0.0014, mat(EXTRA.dark)));
    rotor.add(tube(motorR * 0.7, 0.008, mat(EXTRA.metal), 12).rotateX(Math.PI / 2));
    const disc = tube(
      spec.propR,
      0.0006,
      new THREE.MeshBasicMaterial({
        color: THEME.line,
        transparent: true,
        opacity: 0.12,
        depthWrite: false,
      }),
      28,
    );
    disc.rotation.x = Math.PI / 2;
    rotor.add(disc);
    rotors.push(rotor);
    group.add(rotor);

    const guard = new THREE.Mesh(
      new THREE.TorusGeometry(spec.propR * 1.08, 0.003, 8, 40),
      mat(EXTRA.motor),
    );
    guard.position.set(mx, my, z0 + 0.022);
    guards.add(guard);
  }
  group.add(guards);

  for (const side of [1, -1] as const) {
    const eye = new THREE.Vector3(spec.camX, side * spec.camY, z0 + 0.008);
    cameraEyes.push(eye);
    group.add(box(spec.camSize, spec.camSize, spec.camSize, mat(EXTRA.dark), eye.x, eye.y, eye.z));
    const lens = tube(spec.camSize * 0.32, spec.camSize * 0.5, mat(0x0b1418), 14);
    lens.rotation.z = Math.PI / 2;
    lens.position.set(eye.x + spec.camSize * 0.6, eye.y, eye.z);
    group.add(lens);
    const cone = fovCone(
      spec.diagonal * 1.3,
      EYE_FOVY / 2,
      side > 0 ? THEME.amber : THEME.velocity,
    );
    cone.position.copy(eye);
    cone.rotation.z = side * EYE_SPLAY;
    fov.add(cone);
  }
  group.add(fov);

  if (spec.onboard) {
    const compute = box(0.1, 0.075, 0.02, mat(0x33484d), 0, -0.01, z0 + 0.055);
    group.add(compute);
    for (let i = 0; i < 6; i++) {
      group.add(box(0.088, 0.004, 0.012, mat(EXTRA.metal), 0, -0.044 + i * 0.0135, z0 + 0.068));
    }
    group.add(box(0.04, 0.04, 0.005, mat(0x1f3a3f), 0.05, 0.03, z0 + 0.032));
    const mast = tube(0.003, 0.09, mat(EXTRA.metal), 10);
    mast.rotation.z = Math.PI / 2;
    mast.position.set(-0.05, 0.05, z0 + 0.045);
    group.add(mast);
    const gps = box(0.03, 0.03, 0.006, mat(0xdfe8e6), -0.05, 0.05, z0 + 0.092);
    group.add(gps);
  } else {
    group.add(box(0.028, 0.028, 0.01, mat(0x2f464d), 0.02, -0.03, z0 + 0.032));
    const whip = tube(0.0018, 0.11, mat(EXTRA.dark), 8);
    whip.rotation.x = -Math.PI / 2.4;
    whip.position.set(-0.05, 0.03, z0 + 0.05);
    group.add(whip);
  }

  return { group, rotors, guards, fov, cameraEyes };
}

function addArenaProps(group: THREE.Group) {
  const beacon = new THREE.Mesh(
    new THREE.SphereGeometry(0.3, 24, 16),
    new THREE.MeshStandardMaterial({
      color: THEME.amber,
      emissive: THEME.targetEmissive,
      emissiveIntensity: 1.2,
    }),
  );
  beacon.position.set(1.6, 0.9, 1.0);
  group.add(beacon);
  const lamp = new THREE.PointLight(THEME.amber, 3, 5);
  lamp.position.copy(beacon.position);
  group.add(lamp);

  const threat = new THREE.Mesh(
    new THREE.SphereGeometry(0.25, 24, 16),
    mat(0x0a0d0f, { roughness: 0.95 }),
  );
  threat.position.set(0.2, 1.7, 1.0);
  group.add(threat);

  for (const [px, py] of [
    [-1.7, 0.5],
    [0.3, 1.9],
    [1.9, 0.3],
  ] as [number, number][]) {
    const pillar = tube(0.3, 1.5, mat(EXTRA.grey, { roughness: 0.85 }), 24);
    pillar.rotation.x = Math.PI / 2;
    pillar.position.set(px, py, 0.75);
    group.add(pillar);
    const ring = new THREE.Mesh(
      new THREE.CylinderGeometry(0.303, 0.303, 0.3, 24, 1, true),
      new THREE.MeshStandardMaterial({
        color: EXTRA.dark,
        side: THREE.DoubleSide,
        roughness: 0.9,
      }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.set(px, py, 1.0);
    group.add(ring);
  }
}

function canvasTexture(base: number, ink: number, spacing: number): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = `rgb(${base},${base},${base})`;
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = `rgba(${ink},${ink},${ink},0.55)`;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 256; i += spacing) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 256);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(256, i);
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 4);
  return tex;
}

function buildEnvironment(id: EnvId): THREE.Group {
  const group = new THREE.Group();
  const grid = new THREE.GridHelper(7, 28, THEME.grid, THEME.gridMinor);
  grid.rotation.x = Math.PI / 2;
  group.add(grid);

  if (id === "blueprint") {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(7, 7),
      new THREE.MeshStandardMaterial({ color: THEME.wallFill, roughness: 0.95 }),
    );
    floor.position.z = -0.002;
    group.add(floor);
    return group;
  }

  const textured = id === "textured";
  const floorMat = textured
    ? new THREE.MeshStandardMaterial({
        map: canvasTexture(0x59, 0x4a, 32),
        color: 0xffffff,
        roughness: 0.9,
      })
    : mat(EXTRA.grey, { roughness: 0.95 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), floorMat);
  group.add(floor);

  const wallMat = textured
    ? new THREE.MeshStandardMaterial({
        map: canvasTexture(0x59, 0x44, 48),
        color: 0xffffff,
        roughness: 0.9,
        side: THREE.DoubleSide,
      })
    : mat(EXTRA.grey, { roughness: 0.92, side: THREE.DoubleSide });
  for (const [sx, sy, w, d] of [
    [0, 3, 6, 0.08],
    [0, -3, 6, 0.08],
    [3, 0, 0.08, 6],
    [-3, 0, 0.08, 6],
  ] as [number, number, number, number][]) {
    group.add(box(w, d, 1.5, wallMat, sx, sy, 0.75));
    group.add(
      box(
        w * 0.995,
        d * 0.995,
        0.4,
        new THREE.MeshStandardMaterial({
          color: EXTRA.dark,
          roughness: 0.9,
          side: THREE.DoubleSide,
        }),
        sx,
        sy,
        1.0,
      ),
    );
  }

  const net = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(6, 6, 1.5)),
    new THREE.LineBasicMaterial({ color: THEME.line, transparent: true, opacity: 0.18 }),
  );
  net.position.z = 0.75;
  group.add(net);

  addArenaProps(group);
  return group;
}

const stage = document.getElementById("stage")!;
const scene = new THREE.Scene();
scene.background = new THREE.Color(THEME.bg);
scene.fog = new THREE.Fog(THEME.bg, 4, 14);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xc2ebef, 0x24303a, 2));
const key = new THREE.DirectionalLight(0xffefd8, 3);
key.position.set(2, -3, 5);
scene.add(key);

const state = {
  airframe: "A" as AirframeId,
  env: "blueprint" as EnvId,
  guards: true,
  fov: true,
  spin: true,
};

const envGroup = new THREE.Group();
scene.add(envGroup);
let airframe: Airframe;
const drone = new THREE.Group();
scene.add(drone);
drone.position.z = 0.35;

function mountAirframe(id: AirframeId) {
  airframe = buildAirframe(SPECS[id]);
  drone.add(airframe.group);
}

function refreshAirframe() {
  airframe.group.removeFromParent();
  airframe.group.traverse((n) => {
    const mesh = n as THREE.Mesh;
    mesh.geometry?.dispose();
  });
  mountAirframe(state.airframe);
  airframe.guards.visible = state.guards;
  airframe.fov.visible = state.fov;
  const spec = SPECS[state.airframe];
  const dist = spec.diagonal * 2.4;
  controls.target.set(0, 0, 0.3);
  camera.position.set(dist * 0.8, -dist, dist * 0.8);
  controls.update();
  renderSpec(spec);
}

function refreshEnv() {
  envGroup.clear();
  envGroup.add(buildEnvironment(state.env));
}

function buttons(
  hostId: string,
  entries: [string, string][],
  isActive: (value: string) => boolean,
  pick: (value: string) => void,
) {
  const host = document.getElementById(hostId)!;
  host.replaceChildren();
  for (const [value, label] of entries) {
    const b = document.createElement("button");
    b.textContent = label;
    b.classList.toggle("active", isActive(value));
    b.onclick = () => {
      pick(value);
      wire();
    };
    host.append(b);
  }
}

function renderSpec(spec: AirframeSpec) {
  document.getElementById("spec")!.innerHTML = `
    <dt>Frame</dt><dd>${spec.frameClass}</dd>
    <dt>Motor diagonal</dt><dd>${(spec.diagonal * 1000).toFixed(0)} mm</dd>
    <dt>Prop radius</dt><dd>${(spec.propR * 1000).toFixed(0)} mm</dd>
    <dt>Cameras</dt><dd>2 · ±0.75 rad · 75° vfov</dd>
    <dt>Combined FOV</dt><dd>~177°</dd>
    <dt>Onboard compute</dt><dd>${spec.onboard ? "Jetson-class" : "— (ground)"}</dd>`;
  document.getElementById("note")!.textContent = spec.note;
}

function wire() {
  buttons(
    "airframe",
    [
      ["A", "Option A · 250 mm"],
      ["B", "Option B · 450 mm"],
    ],
    (v) => v === state.airframe,
    (v) => {
      state.airframe = v as AirframeId;
      refreshAirframe();
    },
  );
  buttons(
    "options",
    [
      ["guards", "Prop guards"],
      ["fov", "Eye FOV"],
      ["spin", "Spin rotors"],
    ],
    (v) => state[v as "guards" | "fov" | "spin"],
    (v) => {
      const k = v as "guards" | "fov" | "spin";
      state[k] = !state[k];
      if (k === "guards") airframe.guards.visible = state.guards;
      if (k === "fov") airframe.fov.visible = state.fov;
    },
  );
  buttons(
    "environment",
    [
      ["blueprint", "Blueprint"],
      ["replica", "Replica arena"],
      ["textured", "Textured"],
    ],
    (v) => v === state.env,
    (v) => {
      state.env = v as EnvId;
      refreshEnv();
    },
  );
}

mountAirframe(state.airframe);
refreshEnv();
wire();
refreshAirframe();

new ResizeObserver(() => {
  const { width, height } = stage.getBoundingClientRect();
  renderer.setSize(width, height);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}).observe(stage);

let last = performance.now();
function frame(now: number) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (state.spin) for (const r of airframe.rotors) r.rotation.z += dt * 45;
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(frame);
