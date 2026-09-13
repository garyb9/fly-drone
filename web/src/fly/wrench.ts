import type { Pose, Vec3, Readouts } from "./types";
import { v, add, scale, norm } from "./types";
import { CONFIG } from "../app/config";
import type { ValueNoise } from "./noise";

export interface Wrench {
  force: Vec3;
  torque: Vec3;
}
export interface EscapeState {
  armed: boolean;
  lockout: number;
}
export const initEscapeState = (): EscapeState => ({ armed: true, lockout: 0 });

const P = CONFIG.physics;

export function mapReadouts(
  readouts: Readouts,
  pose: Pose,
  esc: EscapeState,
  dt: number,
  noise: ValueNoise,
  tSeconds: number,
  neuralFlight = false,
): { wrench: Wrench; esc: EscapeState; firedImpulse: Vec3 | null } {
  const n = (ch: number) => noise.at(ch, tSeconds) * P.NOISE_AMP;
  // One shared sample for both wings: the ValueNoise channels are not zero-mean
  // over a short window, so independent wing noise left a standing `wl - wr` DC
  // bias that slowly rolled a resting fly and made it spiral off. Correlated
  // noise perturbs the symmetric term `s = (wl+wr)/2` (harmless lift/thrust
  // jitter — the feature) but cancels out of the asymmetric term `a = wl - wr`.
  const wingNoise = n(0);
  const separated = readouts.power_l !== undefined && readouts.power_r !== undefined;
  const wl = (separated ? readouts.power_l! : (readouts.wing_l ?? 0)) + wingNoise;
  const wr = (separated ? readouts.power_r! : (readouts.wing_r ?? 0)) + wingNoise;
  // Power is one actuator; its forward component below shares the same budget.
  const thrust = (separated ? 0 : (readouts.thrust ?? 0)) + n(2);
  // n(3) alone is not zero-mean over a short window, so it fed a slow DC yaw-torque
  // bias → the resting fly wandered ~50°/15s in heading. A finite difference of the
  // same noise channel is the increment of a stationary process: mean-zero, so its
  // integral (heading) stays bounded while still wobbling for life.
  const yawJitter =
    (noise.at(3, tSeconds) - noise.at(3, tSeconds - CONFIG.physics.YAW_JITTER_DT)) * P.NOISE_AMP;
  const yaw = (readouts.yaw_torque ?? 0) + yawJitter;
  const escape = readouts.escape ?? 0;

  let lockout = Math.max(0, esc.lockout - dt);
  let armed = esc.armed;
  let firedImpulse: Vec3 | null = null;

  if (armed && escape >= P.ESCAPE_TH) {
    firedImpulse = scale(norm(add(pose.up, pose.forward)), P.ESCAPE_IMPULSE);
    armed = false;
    lockout = P.ESCAPE_LOCKOUT_S;
  }
  if (!armed && escape < P.ESCAPE_TH - P.ESCAPE_HYST) armed = true;

  const underLockout = lockout > 0;
  const s = (wl + wr) / 2;
  const a = separated ? (readouts.steer_l ?? 0) - (readouts.steer_r ?? 0) : wl - wr;

  let force = v();
  let torque = v();
  if (!underLockout) {
    const lift = neuralFlight
      ? (P.GRAVITY * P.MASS * Math.max(0, s)) / 0.5
      : P.GRAVITY * P.MASS + P.LIFT_K * (s - P.HOVER_S);
    const liftAxis = separated ? norm(add(pose.up, scale(pose.forward, 0.12))) : pose.up;
    force = add(
      scale(liftAxis, lift),
      scale(pose.forward, (neuralFlight ? 0 : P.CRUISE_THRUST) + P.THRUST_K * thrust),
    );
    torque = add(scale(pose.forward, P.ROLL_K * a), scale(pose.up, P.YAW_A_K * a + P.YAW_K * yaw));
  }
  return { wrench: { force, torque }, esc: { armed, lockout }, firedImpulse };
}
