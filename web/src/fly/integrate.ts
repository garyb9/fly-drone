import type { Vec3, Quat, Pose } from "./types";
import { add, scale, v } from "./types";
import { qIntegrate, qRotate } from "./quat";
import { CONFIG } from "../app/config";
import type { Wrench } from "./wrench";

export interface BodyState {
  position: Vec3;
  orientation: Quat;
  vel: Vec3;
  angVel: Vec3;
}
const P = CONFIG.physics;

export function integrate(
  s: BodyState,
  wrench: Wrench,
  impulse: Vec3 | null,
  dt: number,
): BodyState {
  const gravity = v(0, -P.GRAVITY * P.MASS, 0);
  let vel = add(s.vel, scale(add(wrench.force, gravity), dt / P.MASS));
  if (impulse) vel = add(vel, scale(impulse, 1 / P.MASS));
  vel = scale(vel, Math.exp(-P.LIN_DRAG * dt));
  const position = add(s.position, scale(vel, dt));

  let angVel = add(s.angVel, scale(wrench.torque, dt / P.INERTIA));
  angVel = scale(angVel, Math.exp(-P.ANG_DRAG * dt));
  const orientation = qIntegrate(s.orientation, angVel, dt);
  return { position, orientation, vel, angVel };
}

export function poseOf(s: BodyState): Pose {
  return {
    position: s.position,
    orientation: s.orientation,
    forward: qRotate(s.orientation, v(1, 0, 0)),
    up: qRotate(s.orientation, v(0, 1, 0)),
  };
}
