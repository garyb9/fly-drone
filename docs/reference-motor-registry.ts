import type { Cell } from "../data/dataset";
import legMappings from "./leg-mappings.json";

export const MOTOR_MODEL_VERSION = "power-steering-legs-v1";
export const LEG_NAMES = ["lf", "lm", "lh", "rf", "rm", "rh"] as const;
const steeringTypes = new Set([
  "b1 MN",
  "b2 MN",
  "b3 MN",
  "i1 MN",
  "i2 MN",
  "iii1 MN",
  "iii3 MN",
  "hg1 MN",
  "hg2 MN",
  "hg3 MN",
  "hg4 MN",
]);

/** Anatomical type selection; output signs/gains and soma-side laterality are
 * explicit engineering assumptions, not measured muscle transfer functions. */
export function motorRegistry(cells: Cell[]) {
  const readouts: Record<string, number[]> = {
    power_l: [],
    power_r: [],
    steer_l: [],
    steer_r: [],
    dnp03_l: [],
    dnp03_r: [],
  };
  const inputs: Record<string, number[]> = { looming_l: [], looming_r: [], flow_l: [], flow_r: [] };
  for (const name of LEG_NAMES) readouts[`leg_${name}`] = [];
  const legs = new Map(legMappings.cells.map((c) => [c.id, c.leg]));
  cells.forEach((cell, i) => {
    const leg = legs.get(cell.id);
    if (leg) readouts[`leg_${leg}`]!.push(i);
    const side = cell.side === "L" ? "l" : cell.side === "R" ? "r" : null;
    if (!side) return;
    if (/^(DLMn|DVMn) /.test(cell.type)) readouts[`power_${side}`]!.push(i);
    if (steeringTypes.has(cell.type)) readouts[`steer_${side}`]!.push(i);
    if (cell.type === "DNp03") readouts[`dnp03_${side}`]!.push(i);
    if (["LC4", "LPLC2"].includes(cell.type)) inputs[`looming_${side}`]!.push(i);
    if (["HSE", "HSN", "HSS"].includes(cell.type)) inputs[`flow_${side}`]!.push(i);
  });
  const tonic = [...readouts.power_l!, ...readouts.power_r!];
  return { version: MOTOR_MODEL_VERSION, readouts, inputs, tonic };
}
