import paletteJson from "./palette.json";

export const PALETTE = paletteJson;

export function hexToInt(hex: string): number {
  return parseInt(hex.replace("#", ""), 16);
}
