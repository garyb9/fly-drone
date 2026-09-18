import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { els } from "../dom";
import { PALETTE, hexToInt } from "../theme/tokens";
import { THEME } from "./theme";
import { applyRoom, createAxisGizmo, createGlowDecal, type Room } from "./world";
import { vector } from "./transform";
import {
  buildDrone,
  DEFAULT_EYES,
  type Airframe,
  type AirframeId,
  type EyeGeometry,
} from "./drone";

// The three WebGL viewports (world / brain / fly body) and every object in them. `createStage`
// is the only place that touches renderer setup; the frame logic and controls read the refs.

const ACTIVITY_HOT = hexToInt(PALETTE.successTeal);
export const ACTIVITY_HOT_COLOR = new THREE.Color(ACTIVITY_HOT);
const LEGACY_VIEWPORT_BG = hexToInt(PALETTE.card);

export const TARGET_BASE_RADIUS = 0.22;
export const OBSTACLE_BASE_RADIUS = 0.25;
export const THREAT_TRAIL_MAX = 60;
export const WORLD_HOME = { position: [2.5, 2.2, 3.2], target: [0.3, 0.8, 0] };
const AIRFRAME_KEY = "fly-drone.airframe";

export type Viewport = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
};

export type Wing = { node: THREE.Object3D; rest: number; sign: number };

export type Stage = {
  world: Viewport;
  brain: Viewport;
  flyview: Viewport;
  roomGroup: THREE.Group;
  axisGizmo: THREE.Group;
  drone: THREE.Group;
  locator: THREE.Mesh;
  target: THREE.Mesh;
  beaconRing: THREE.Mesh;
  obstacle: THREE.Mesh;
  clearanceRing: THREE.Mesh;
  threatTrail: THREE.Vector3[];
  threatTrailGeometry: THREE.BufferGeometry;
  threatTrailLine: THREE.Line;
  headingArrow: THREE.ArrowHelper;
  velocityArrow: THREE.ArrowHelper;
  commandArrow: THREE.ArrowHelper;
  glowDecal: THREE.Mesh;
  fly: THREE.Group;
  wings: Wing[];
  getAirframe: () => Airframe;
  setAirframe: (id: AirframeId) => void;
  setEyeGeometry: (eyes: EyeGeometry) => void;
  applyRoom: (room: Room) => void;
};

function createViewport(host: HTMLElement, position: number[], target: number[]): Viewport {
  const scene = new THREE.Scene();
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

const mat = (c: number, metalness = 0.2) =>
  new THREE.MeshStandardMaterial({ color: c, metalness, roughness: 0.45 });

export function createStage(): Stage {
  const world = createViewport(els.world, [2.5, 2.2, 3.2], [0.3, 0.8, 0]);
  world.scene.fog = new THREE.Fog(THEME.bg, 7, 20);
  world.scene.add(new THREE.GridHelper(12, 48, THEME.grid, THEME.gridMinor));
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(18, 18),
    new THREE.MeshStandardMaterial({ color: THEME.wallFill, roughness: 0.95 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.005;
  world.scene.add(floor);

  // Rebuilt from the WebSocket "room" message (walls/bands/pillars) — see scene/world.ts.
  const roomGroup = new THREE.Group();
  world.scene.add(roomGroup);

  // World-axis reference, placed just outside the arena's walls (repositioned in updateRoom)
  // so heading/velocity/commanded vectors can be read against a fixed frame.
  const axisGizmo = createAxisGizmo(0.6);
  world.scene.add(axisGizmo);

  // Nose heading, measured velocity, and the raw decoder command, drawn from the drone's own
  // position. Divergence between them is the "life of its own" symptom made visible.
  const headingArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(),
    0.3,
    THEME.amber,
    0.08,
    0.05,
  );
  const velocityArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(),
    0.001,
    THEME.velocity,
    0.08,
    0.05,
  );
  const commandArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(),
    0.001,
    THEME.command,
    0.08,
    0.05,
  );
  world.scene.add(headingArrow, velocityArrow, commandArrow);

  const glowDecal = createGlowDecal(THEME.amber);
  glowDecal.position.y = 0.002;
  world.scene.add(glowDecal);

  const drone = new THREE.Group();
  world.scene.add(drone);
  // Hardware airframe shell: A = 250 mm with the connectome on the ground, B = 450 mm with an
  // onboard companion. Built in the drone's local Z-up frame; the flight plant stays CF2X.
  let airframeId: AirframeId = localStorage.getItem(AIRFRAME_KEY) === "B" ? "B" : "A";
  let eyeGeometry: EyeGeometry = DEFAULT_EYES;
  let airframe: Airframe = buildDrone(airframeId, eyeGeometry);
  drone.add(airframe.group);

  const setAirframe = (id: AirframeId): void => {
    airframe.group.removeFromParent();
    airframe.group.traverse((n) => {
      const mesh = n as THREE.Mesh;
      mesh.geometry?.dispose();
      if (Array.isArray(mesh.material)) mesh.material.forEach((m) => m.dispose());
      else mesh.material?.dispose();
    });
    airframeId = id;
    airframe = buildDrone(id, eyeGeometry);
    airframe.fov.visible = els["toggle-fov"].classList.contains("active");
    airframe.guards.visible = els["toggle-guards"].classList.contains("active");
    drone.add(airframe.group);
    els["af-a"].classList.toggle("active", id === "A");
    els["af-b"].classList.toggle("active", id === "B");
    localStorage.setItem(AIRFRAME_KEY, id);
  };

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

  // Marks whether the beacon is geometrically visible to the drone (free-roam beacon_visible).
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

  // Surface-distance halo from free_roam.clearance.
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

  const brain = createViewport(els.brain, [0, 0, 4], [0, 0, 0]);
  brain.scene.background = new THREE.Color(0x0d171e);

  const flyview = createViewport(els.fly, [4, 2.5, 4], [0, 1, 0]);
  flyview.scene.background = new THREE.Color(LEGACY_VIEWPORT_BG);
  flyview.scene.add(new THREE.GridHelper(6, 12, 0x35535b, 0x20313a));
  const flyRim = new THREE.DirectionalLight(0x6fe2ff, 1.4);
  flyRim.position.set(-3, 2.5, -3);
  flyview.scene.add(flyRim);
  const wings: Wing[] = [];
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

  const setEyeGeometry = (eyes: EyeGeometry): void => {
    if (eyes.splay === eyeGeometry.splay && eyes.fovyDeg === eyeGeometry.fovyDeg) return;
    eyeGeometry = eyes;
    setAirframe(airframeId);
  };

  return {
    world,
    brain,
    flyview,
    roomGroup,
    axisGizmo,
    drone,
    locator,
    target,
    beaconRing,
    obstacle,
    clearanceRing,
    threatTrail,
    threatTrailGeometry,
    threatTrailLine,
    headingArrow,
    velocityArrow,
    commandArrow,
    glowDecal,
    fly,
    wings,
    getAirframe: () => airframe,
    setAirframe,
    setEyeGeometry,
    applyRoom: (room) => applyRoom(roomGroup, room, vector),
  };
}
