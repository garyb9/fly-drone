import type { Quat, Vec3 } from "./types";

export const qIdentity = (): Quat => ({ x: 0, y: 0, z: 0, w: 1 });

export const qNormalize = (q: Quat): Quat => {
  const l = Math.hypot(q.x, q.y, q.z, q.w) || 1;
  return { x: q.x / l, y: q.y / l, z: q.z / l, w: q.w / l };
};

export const qMul = (a: Quat, b: Quat): Quat => ({
  w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
  y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
  z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
});

export const qFromAxisAngle = (axis: Vec3, rad: number): Quat => {
  const l = Math.hypot(axis.x, axis.y, axis.z) || 1;
  const s = Math.sin(rad / 2);
  return {
    x: (axis.x / l) * s,
    y: (axis.y / l) * s,
    z: (axis.z / l) * s,
    w: Math.cos(rad / 2),
  };
};

export const qRotate = (q: Quat, p: Vec3): Vec3 => {
  const t = qMul(qMul(q, { x: p.x, y: p.y, z: p.z, w: 0 }), {
    x: -q.x,
    y: -q.y,
    z: -q.z,
    w: q.w,
  });
  return { x: t.x, y: t.y, z: t.z };
};

export const qIntegrate = (q: Quat, w: Vec3, dt: number): Quat => {
  const dq = qMul({ x: w.x, y: w.y, z: w.z, w: 0 }, q);
  return qNormalize({
    x: q.x + 0.5 * dq.x * dt,
    y: q.y + 0.5 * dq.y * dt,
    z: q.z + 0.5 * dq.z * dt,
    w: q.w + 0.5 * dq.w * dt,
  });
};
