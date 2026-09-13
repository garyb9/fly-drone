import type { Vec3, WorldQuery, Aabb } from "./types";
import { CONFIG } from "../app/config";

const AXES = ["x", "y", "z"] as const;

export function resolveSphere(pos: Vec3, vel: Vec3, radius: number, world: WorldQuery) {
  const p = { ...pos },
    vv = { ...vel };
  let contact = false;

  for (const box of world.aabbs) {
    const exp: Aabb = {
      min: { x: box.min.x - radius, y: box.min.y - radius, z: box.min.z - radius },
      max: { x: box.max.x + radius, y: box.max.y + radius, z: box.max.z + radius },
    };
    if (
      p.x <= exp.min.x ||
      p.x >= exp.max.x ||
      p.y <= exp.min.y ||
      p.y >= exp.max.y ||
      p.z <= exp.min.z ||
      p.z >= exp.max.z
    )
      continue;
    // inside the expanded box → penetrating; push out on least-penetration axis
    let bestAx: (typeof AXES)[number] = "x",
      bestPen = Infinity,
      bestDir = 1;
    for (const ax of AXES) {
      const penPos = exp.max[ax] - p[ax]; // distance to exit in +ax
      const penNeg = p[ax] - exp.min[ax];
      const [pen, dir] = penPos < penNeg ? [penPos, 1] : [penNeg, -1];
      if (pen < bestPen) {
        bestPen = pen;
        bestAx = ax;
        bestDir = dir;
      }
    }
    p[bestAx] += bestDir * bestPen;
    if (Math.sign(vv[bestAx]) === -bestDir) vv[bestAx] = -CONFIG.physics.BOUNCE * vv[bestAx];
    contact = true;
  }

  for (const ax of AXES) {
    const lo = world.bounds.min[ax] + radius,
      hi = world.bounds.max[ax] - radius;
    if (p[ax] < lo) {
      p[ax] = lo;
      if (vv[ax] < 0) vv[ax] = -CONFIG.physics.BOUNCE * vv[ax];
      contact = true;
    } else if (p[ax] > hi) {
      p[ax] = hi;
      if (vv[ax] > 0) vv[ax] = -CONFIG.physics.BOUNCE * vv[ax];
      contact = true;
    }
  }
  return { position: p, vel: vv, contact };
}
