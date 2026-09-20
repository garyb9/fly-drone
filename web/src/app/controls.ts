import * as THREE from "three";
import type { Viewer } from "./context";
import { els } from "../dom";
import { TABS, setTab } from "./tabs";
import { WORLD_HOME } from "../scene/stage";
import { attitudeFromQuaternion } from "../view/format";

// Camera + control wiring. The render loop reads `applyCameraMode`, so camera state lives here.

let following = true;
let cameraMode: "orbit" | "fpv" | "tpv" = "orbit";
const camEye = new THREE.Vector3();
const camAhead = new THREE.Vector3();

function runTrial(viewer: Viewer): void {
  // No task/ablation selector any more: reset the server's current task with a new seed.
  viewer.send({ op: "reset", seed: Math.floor(Math.random() * 1_000_000) });
}

function setCameraMode(viewer: Viewer, mode: "orbit" | "fpv" | "tpv"): void {
  const { stage } = viewer;
  cameraMode = cameraMode === mode ? "orbit" : mode;
  stage.world.controls.enabled = cameraMode === "orbit";
  if (cameraMode !== "orbit") {
    // Auto-orbit drives the orbit camera; leaving it on would fight the canned FPV/TPV pose.
    stage.world.controls.autoRotate = false;
    els["cam-orbit"].classList.remove("active");
  }
  els["cam-fpv"].classList.toggle("active", cameraMode === "fpv");
  els["cam-tpv"].classList.toggle("active", cameraMode === "tpv");
}

export function applyCameraMode(viewer: Viewer): void {
  const { stage } = viewer;
  if (cameraMode === "fpv") {
    const cam = stage.getAirframe().eyes[0];
    camEye.set(cam.x, 0, cam.z).applyQuaternion(stage.drone.quaternion).add(stage.drone.position);
    camAhead.set(1, 0, 0).applyQuaternion(stage.drone.quaternion).add(camEye);
    stage.world.camera.position.copy(camEye);
    stage.world.camera.up.set(0, 1, 0);
    stage.world.camera.lookAt(camAhead);
    stage.world.controls.target.copy(camAhead);
  } else if (cameraMode === "tpv") {
    camEye.set(-0.7, 0.32, 0).applyQuaternion(stage.drone.quaternion).add(stage.drone.position);
    stage.world.camera.position.copy(camEye);
    stage.world.camera.up.set(0, 1, 0);
    stage.world.camera.lookAt(stage.drone.position);
    stage.world.controls.target.copy(stage.drone.position);
  } else if (following) {
    stage.world.controls.target.copy(stage.drone.position);
  }
}

// Semantic-vector palette: default blueprint colours, or an Okabe-Ito colour-blind-safe set
// (yellow / blue / vermillion). Persisted per browser; the labelled X/Y/Z gizmo is unchanged.
const CVD_SAFE_KEY = "fly-drone.cvd-safe";
const VECTOR_COLORS = {
  default: { heading: "#ffb15c", velocity: "#6fe2ff", command: "#ff6fd8" },
  safe: { heading: "#f0e442", velocity: "#0072b2", command: "#d55e00" },
};

function applyVectorPalette(viewer: Viewer, safe: boolean): void {
  const colors = safe ? VECTOR_COLORS.safe : VECTOR_COLORS.default;
  viewer.stage.headingArrow.setColor(new THREE.Color(colors.heading));
  viewer.stage.velocityArrow.setColor(new THREE.Color(colors.velocity));
  viewer.stage.commandArrow.setColor(new THREE.Color(colors.command));
  els["legend-heading"].style.background = colors.heading;
  els["legend-velocity"].style.background = colors.velocity;
  els["legend-command"].style.background = colors.command;
  els["cb-safe"].classList.toggle("active", safe);
  els["cb-safe"].setAttribute("aria-pressed", String(safe));
}

function setHelp(open: boolean): void {
  els["help-overlay"].hidden = !open;
  if (open) els["help-close"].focus();
}

export function initControls(viewer: Viewer): void {
  for (const tab of TABS) {
    els[`tab-${tab}`].addEventListener("click", () => setTab(tab));
  }
  els.pause.onclick = () => viewer.send({ op: "pause", value: !viewer.latest?.paused });
  els.reset.onclick = () => runTrial(viewer);
  els["cam-follow"].onclick = () => {
    following = !following;
    els["cam-follow"].classList.toggle("active", following);
  };
  els["cam-recenter"].onclick = () =>
    viewer.stage.world.controls.target.copy(viewer.stage.drone.position);
  els["cam-reset"].onclick = () => {
    cameraMode = "orbit";
    following = false;
    els["cam-follow"].classList.remove("active");
    els["cam-fpv"].classList.remove("active");
    els["cam-tpv"].classList.remove("active");
    viewer.stage.world.controls.enabled = true;
    viewer.stage.world.camera.position.fromArray(WORLD_HOME.position);
    viewer.stage.world.controls.target.fromArray(WORLD_HOME.target);
    viewer.stage.world.controls.update();
  };
  els["cam-fpv"].onclick = () => setCameraMode(viewer, "fpv");
  els["cam-tpv"].onclick = () => setCameraMode(viewer, "tpv");
  els["cam-orbit"].onclick = () => {
    const on = !viewer.stage.world.controls.autoRotate;
    viewer.stage.world.controls.autoRotate = on;
    viewer.stage.world.controls.autoRotateSpeed = 0.8;
    els["cam-orbit"].classList.toggle("active", on);
  };
  els["af-a"].onclick = () => viewer.stage.setAirframe("A");
  els["af-b"].onclick = () => viewer.stage.setAirframe("B");
  els["toggle-fov"].onclick = () => {
    const on = !els["toggle-fov"].classList.contains("active");
    viewer.stage.getAirframe().fov.visible = on;
    els["toggle-fov"].classList.toggle("active", on);
  };
  els["toggle-guards"].onclick = () => {
    const on = !els["toggle-guards"].classList.contains("active");
    viewer.stage.getAirframe().guards.visible = on;
    els["toggle-guards"].classList.toggle("active", on);
  };
  els["eye-fx"].onclick = () => {
    const on = !els["eye-fx"].classList.contains("active");
    els.eye0.closest(".eyes")?.classList.toggle("fx", on);
    els["eye-fx"].classList.toggle("active", on);
  };
  document.querySelectorAll<HTMLButtonElement>("[data-target]").forEach((b) => {
    b.onclick = () =>
      viewer.send({
        op: "objects",
        target: [2, b.dataset.target === "left" ? 1.3 : b.dataset.target === "right" ? -1.3 : 0, 1],
      });
  });
  els.loom.onclick = () => {
    const p = viewer.latest?.state.position ?? [0, 0, 1];
    viewer.send({
      op: "objects",
      obstacle: [Math.min(3.5, p[0] + 0.65), p[1], Math.max(0.3, p[2])],
    });
  };
  els.clear.onclick = () => viewer.send({ op: "objects", obstacle: [2, -2, 1] });
  // Free-roam probes. The server validates each one and reports any refusal through #error.
  els["roam-beacon"].onclick = () => {
    const f = viewer.latest;
    if (!f) return;
    const { yaw } = attitudeFromQuaternion(f.state.quaternion);
    const p = f.state.position;
    const distance = 1.5;
    viewer.send({
      op: "place_beacon",
      x: p[0] + Math.cos(yaw) * distance,
      y: p[1] + Math.sin(yaw) * distance,
    });
  };
  els["roam-threat"].onclick = () => viewer.send({ op: "launch_threat" });
  els["roam-silence-sensory"].onclick = () =>
    viewer.send({
      op: "pathway",
      name: "sensory",
      silenced: !viewer.latest?.free_roam?.silenced.includes("sensory"),
    });
  els["roam-silence-light"].onclick = () =>
    viewer.send({
      op: "pathway",
      name: "light",
      silenced: !viewer.latest?.free_roam?.silenced.includes("light"),
    });
  els["roam-silence-loom"].onclick = () =>
    viewer.send({
      op: "pathway",
      name: "loom",
      silenced: !viewer.latest?.free_roam?.silenced.includes("loom"),
    });
  els["roam-ghost"].onclick = () =>
    viewer.send({ op: "ghost", value: !viewer.latest?.free_roam?.ghost });
  els["roam-restore"].onclick = () => viewer.send({ op: "restore" });
  els["roam-codec"].onclick = () => {
    // Switch the declared bridge and reset the sim so the new codec flies from a clean state.
    const next = (viewer.latest?.codec ?? "v2") === "v2" ? "v1" : "v2";
    viewer.send({ op: "codec", value: next });
  };
  els["cb-safe"].onclick = () => {
    const safe = els["cb-safe"].getAttribute("aria-pressed") !== "true";
    localStorage.setItem(CVD_SAFE_KEY, safe ? "1" : "0");
    applyVectorPalette(viewer, safe);
  };
  applyVectorPalette(viewer, localStorage.getItem(CVD_SAFE_KEY) === "1");
  els["help-open"].onclick = () => setHelp(true);
  els["help-close"].onclick = () => setHelp(false);
  window.addEventListener("keydown", (event) => {
    const target = event.target as HTMLElement | null;
    if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
    if (event.key === "?") {
      setHelp(els["help-overlay"].hidden);
      event.preventDefault();
      return;
    }
    if (event.key === "Escape") {
      setHelp(false);
      return;
    }
    switch (event.key.toLowerCase()) {
      case " ":
        event.preventDefault();
        viewer.send({ op: "pause", value: !viewer.latest?.paused });
        break;
      case "r":
        runTrial(viewer);
        break;
      case "f":
        els["cam-follow"].click();
        break;
      case "1":
        setCameraMode(viewer, "orbit");
        break;
      case "2":
        setCameraMode(viewer, "fpv");
        break;
      case "3":
        setCameraMode(viewer, "tpv");
        break;
      case "g":
        if (viewer.latest?.free_roam)
          viewer.send({ op: "ghost", value: !viewer.latest.free_roam.ghost });
        break;
    }
  });
}
