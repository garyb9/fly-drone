import { describe, it, expect } from "vitest";
import { integrate } from "./integrate";
import { v } from "./types";
describe("reference fly equations", () => {
  it("balances gravity with lift", () => {
    const s = {
      position: v(0, 1, 0),
      orientation: { x: 0, y: 0, z: 0, w: 1 },
      vel: v(),
      angVel: v(),
    };
    const result = integrate(s, { force: v(0, 9.81, 0), torque: v() }, null, 0.005);
    expect(result.position).toEqual(s.position);
  });
});
