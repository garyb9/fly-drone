import { CONFIG } from "../app/config";

const hash = (x: number) => {
  let h = (x ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0xffffffff; // [0,1)
};

export class ValueNoise {
  constructor(private seed: number) {}
  at(channel: number, tSeconds: number): number {
    const phase = tSeconds * CONFIG.physics.NOISE_HZ;
    const i = Math.floor(phase);
    const f = phase - i;
    const key = (n: number) => (this.seed * 2654435761 + channel * 40503 + n * 668265263) | 0;
    const a = hash(key(i)) * 2 - 1;
    const b = hash(key(i + 1)) * 2 - 1;
    const u = (1 - Math.cos(f * Math.PI)) * 0.5;
    return a + (b - a) * u;
  }
}
