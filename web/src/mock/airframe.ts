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

const at = <T extends THREE.Object3D>(o: T, x: number, y: number, z: number): T => {
  o.position.set(x, y, z);
  return o;
};

function cable(points: [number, number, number][], r = 0.0016): THREE.Mesh {
  const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)));
  return new THREE.Mesh(
    new THREE.TubeGeometry(curve, 20, r, 6, false),
    mat(0x0d1418, { roughness: 0.85 }),
  );
}

function boltRing(
  radius: number,
  count: number,
  m: THREE.Material,
  z: number,
  boltR = 0.0011,
): THREE.Group {
  const g = new THREE.Group();
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    const bolt = tube(boltR, 0.0025, m, 6);
    bolt.rotation.x = Math.PI / 2;
    g.add(at(bolt, Math.cos(a) * radius, Math.sin(a) * radius, z));
  }
  return g;
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
  const upperZ = z0 + 0.018;
  const motorR = spec.diagonal * 0.05;
  const carbon = mat(EXTRA.carbon, { metalness: 0.15, roughness: 0.55 });
  const alu = mat(EXTRA.metal, { metalness: 0.8, roughness: 0.28 });
  const anodized = (i: number) =>
    mat(i % 2 === 0 ? 0x2f4a52 : 0x39312c, { metalness: 0.75, roughness: 0.32 });
  const prop = new THREE.MeshStandardMaterial({
    color: 0x9aa6aa,
    roughness: 0.35,
    metalness: 0.05,
    transparent: true,
    opacity: 0.85,
  });

  group.add(box(bw, bd, bh, carbon, 0, 0, z0));
  group.add(box(bw * 0.82, bd * 0.82, bh, mat(EXTRA.plate), 0, 0, upperZ));
  group.add(boltRing(bw * 0.3, 6, alu, z0 + bh));
  for (const [ox, oy] of [
    [bw * 0.34, bd * 0.34],
    [bw * 0.34, -bd * 0.34],
    [-bw * 0.34, bd * 0.34],
    [-bw * 0.34, -bd * 0.34],
  ] as [number, number][]) {
    group.add(at(tube(spec.diagonal * 0.02, 0.018, alu, 8), ox, oy, z0 + 0.009));
  }

  const [batW, batD, batH] = spec.battery;
  group.add(at(box(batW, batD, batH, mat(0x22343b, { roughness: 0.7 })), 0, 0, upperZ + batH / 2));
  for (const d of [-0.26, 0.26]) {
    group.add(
      at(
        box(batW * 1.04, 0.006, batH * 1.06, mat(0x0e1416, { roughness: 0.85 })),
        0,
        batD * d,
        upperZ + batH / 2,
      ),
    );
  }

  const fcX = bw * 0.26;
  const fcY = -bd * 0.26;
  const fcZ = upperZ + 0.006;
  group.add(at(box(0.04, 0.04, 0.006, mat(0x14343a, { roughness: 0.55 })), fcX, fcY, fcZ));
  group.add(at(box(0.012, 0.024, 0.005, mat(0x0c1a1e)), fcX - 0.014, fcY, fcZ + 0.004));
  group.add(
    at(
      new THREE.Mesh(
        new THREE.SphereGeometry(0.0026, 8, 8),
        new THREE.MeshStandardMaterial({
          color: THEME.amber,
          emissive: THEME.amber,
          emissiveIntensity: 2,
        }),
      ),
      fcX + 0.014,
      fcY + 0.014,
      fcZ + 0.005,
    ),
  );

  for (const [i, [dx, dy]] of QUAD.entries()) {
    const mx = (dx * half) / Math.sqrt(2);
    const my = (dy * half) / Math.sqrt(2);
    const angle = Math.atan2(my, mx);
    const armLen = Math.hypot(mx, my);

    const arm = box(armLen, spec.diagonal * 0.05, 0.005, carbon, mx / 2, my / 2, z0);
    arm.rotation.z = angle;
    group.add(arm);
    const clamp = box(
      spec.diagonal * 0.075,
      spec.diagonal * 0.06,
      0.006,
      mat(EXTRA.plate),
      mx * 0.45,
      my * 0.45,
      z0,
    );
    clamp.rotation.z = angle;
    group.add(clamp);

    const base = tube(motorR * 1.06, 0.006, mat(0x1c2a2e), 20);
    base.rotation.x = Math.PI / 2;
    group.add(at(base, mx, my, z0 + 0.006));
    const bell = tube(motorR, 0.014, anodized(i), 20);
    bell.rotation.x = Math.PI / 2;
    group.add(at(bell, mx, my, z0 + 0.014));
    const cap = tube(motorR * 0.86, 0.004, alu, 20);
    cap.rotation.x = Math.PI / 2;
    group.add(at(cap, mx, my, z0 + 0.023));
    const nut = tube(motorR * 0.32, 0.006, mat(EXTRA.dark), 6);
    nut.rotation.x = Math.PI / 2;
    group.add(at(nut, mx, my, z0 + 0.028));

    const rotor = new THREE.Group();
    rotor.position.set(mx, my, z0 + 0.03);
    for (const d of [1, -1] as const) {
      const blade = box(spec.propR, spec.propR * 0.22, 0.0016, prop, d * spec.propR * 0.5, 0, 0);
      blade.rotation.x = d * 0.34;
      rotor.add(blade);
    }
    rotor.add(tube(motorR * 0.5, 0.006, alu, 12).rotateX(Math.PI / 2));
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
      new THREE.TorusGeometry(spec.propR * 1.08, 0.0028, 8, 40),
      mat(EXTRA.motor),
    );
    guards.add(at(guard, mx, my, z0 + 0.02));
  }
  group.add(guards);

  for (const side of [1, -1] as const) {
    const eye = new THREE.Vector3(spec.camX, side * spec.camY, z0 + 0.008);
    cameraEyes.push(eye);
    group.add(
      at(box(spec.camSize * 1.6, 0.004, 0.004, alu), eye.x - spec.camSize * 0.75, eye.y, eye.z),
    );
    group.add(
      at(
        box(
          spec.camSize,
          spec.camSize * 0.9,
          spec.camSize * 0.9,
          mat(0x10181c, { roughness: 0.4 }),
        ),
        eye.x,
        eye.y,
        eye.z,
      ),
    );
    const lens = tube(
      spec.camSize * 0.34,
      spec.camSize * 0.7,
      mat(0x05090b, { metalness: 0.6, roughness: 0.25 }),
      18,
    );
    lens.rotation.z = Math.PI / 2;
    group.add(at(lens, eye.x + spec.camSize * 0.75, eye.y, eye.z));
    const hood = new THREE.Mesh(
      new THREE.TorusGeometry(spec.camSize * 0.36, spec.camSize * 0.06, 8, 20),
      mat(EXTRA.dark),
    );
    hood.rotation.y = Math.PI / 2;
    group.add(at(hood, eye.x + spec.camSize * 0.95, eye.y, eye.z));
    const glass = tube(
      spec.camSize * 0.3,
      0.002,
      new THREE.MeshStandardMaterial({ color: 0x0a1a22, metalness: 0.9, roughness: 0.1 }),
      18,
    );
    glass.rotation.z = Math.PI / 2;
    group.add(at(glass, eye.x + spec.camSize * 1.05, eye.y, eye.z));
    group.add(
      cable([
        [fcX, fcY, fcZ + 0.004],
        [eye.x * 0.5, eye.y * 0.6, z0 + 0.012],
        [eye.x - spec.camSize * 0.5, eye.y, eye.z],
      ]),
    );

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

  for (const sy of [-bd * 0.32, bd * 0.32]) {
    group.add(at(box(bw * 0.7, 0.005, 0.004, carbon), 0, sy, z0 - 0.032));
    for (const lx of [-bw * 0.28, bw * 0.28]) {
      group.add(at(box(0.004, 0.004, 0.03, carbon), lx, sy, z0 - 0.017));
    }
  }

  if (spec.onboard) {
    const bellyZ = z0 - 0.024;
    group.add(
      at(
        box(0.1, 0.075, 0.02, mat(0x33484d, { metalness: 0.5, roughness: 0.4 })),
        0,
        0.006,
        bellyZ,
      ),
    );
    for (let i = 0; i < 6; i++) {
      group.add(at(box(0.09, 0.003, 0.012, alu), 0, -0.028 + i * 0.012, bellyZ - 0.013));
    }
    group.add(
      at(
        new THREE.Mesh(new THREE.TorusGeometry(0.016, 0.002, 6, 20), mat(0x11181b)),
        0.03,
        0.02,
        bellyZ - 0.013,
      ),
    );
    group.add(at(tube(0.003, 0.09, alu, 8), -bw * 0.3, -bd * 0.3, upperZ + 0.045));
    group.add(at(box(0.03, 0.03, 0.006, mat(0xdfe8e6)), -bw * 0.3, -bd * 0.3, upperZ + 0.093));
    for (const s of [1, -1] as const) {
      const ant = tube(0.0018, 0.09, mat(EXTRA.dark), 8);
      ant.rotation.set(Math.PI / 2.6, 0, s * 0.35);
      group.add(at(ant, bw * 0.34, s * bd * 0.28, upperZ + 0.05));
    }
    group.add(
      cable([
        [fcX, fcY, fcZ],
        [0, 0, bellyZ + 0.012],
        [-0.02, 0.01, bellyZ],
      ]),
    );
  } else {
    const vtxX = bw * 0.3;
    const vtxY = -bd * 0.3;
    group.add(
      at(box(0.03, 0.03, 0.012, mat(0x24363c, { roughness: 0.5 })), vtxX, vtxY, upperZ + 0.024),
    );
    for (let i = 0; i < 4; i++) {
      group.add(at(box(0.024, 0.002, 0.008, alu), vtxX, vtxY - 0.009 + i * 0.006, upperZ + 0.032));
    }
    const whip = tube(0.0018, 0.12, mat(EXTRA.dark), 8);
    whip.rotation.x = -Math.PI / 2.3;
    group.add(at(whip, -bw * 0.3, bd * 0.26, upperZ + 0.055));
    const stub = tube(0.0018, 0.05, mat(EXTRA.dark), 8);
    stub.rotation.set(Math.PI / 2.6, 0, -0.3);
    group.add(at(stub, bw * 0.32, bd * 0.28, upperZ + 0.042));
    group.add(
      cable([
        [fcX, fcY, fcZ],
        [vtxX, vtxY, upperZ + 0.02],
        [vtxX, vtxY, upperZ + 0.024],
      ]),
    );
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
