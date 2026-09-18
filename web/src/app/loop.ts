import { applyCameraMode } from "./controls";
import type { Viewer } from "./context";
import { droneTarget, obstacleTarget, POSE_TAU, targetTarget } from "./motion";
import { updateDebugVectors } from "../view/gauges";
import { drawScopes } from "../view/meters";

// The single requestAnimationFrame loop: ease the smoothed poses, draw the scopes and the
// debug vectors, then render the three viewports. Server frames only set targets.
export function startRenderer(viewer: Viewer): void {
  const { stage } = viewer;
  let lastRender = performance.now();
  function render(now = performance.now()): void {
    requestAnimationFrame(render);
    const dt = Math.min(0.1, (now - lastRender) / 1000);
    lastRender = now;
    const k = 1 - Math.exp(-dt / POSE_TAU);
    stage.drone.position.lerp(droneTarget.pos, k);
    stage.drone.quaternion.slerp(droneTarget.quat, k);
    // The beacon teleports when collected/placed; interpolating it made a caught ball appear
    // to be dragged to the next spot. Snap it; only the thrown threat glides.
    stage.target.position.copy(targetTarget);
    stage.obstacle.position.lerp(obstacleTarget, k);
    stage.locator.position.set(stage.drone.position.x, 0.003, stage.drone.position.z);
    stage.glowDecal.position.set(stage.drone.position.x, 0.002, stage.drone.position.z);
    if (viewer.latest) {
      updateDebugVectors(
        stage.drone,
        {
          heading: stage.headingArrow,
          velocity: stage.velocityArrow,
          command: stage.commandArrow,
        },
        viewer.latest,
      );
    }
    drawScopes();
    applyCameraMode(viewer);
    for (const v of [stage.world, stage.brain, stage.flyview]) {
      v.controls.update();
      v.renderer.render(v.scene, v.camera);
    }
  }
  render();
}
