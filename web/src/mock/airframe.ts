import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { applyCssTokens } from "../theme/apply-css-tokens";
import { THEME } from "../scene/theme";
import {
  buildAirframe,
  SPECS,
  type Airframe,
  type AirframeId,
  type AirframeSpec,
} from "../scene/drone";

applyCssTokens();

type EnvId = "blueprint" | "replica" | "textured";

const EXTRA = { grey: 0x595959, dark: 0x08090b };

const mat = (color: number, opts: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, metalness: 0.3, roughness: 0.5, ...opts });

function box(w: number, h: number, d: number, m: THREE.Material, x = 0, y = 0, z = 0): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  mesh.position.set(x, y, z);
  return mesh;
}

function tube(r: number, h: number, m: THREE.Material, seg = 16): THREE.Mesh {
  return new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), m);
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
