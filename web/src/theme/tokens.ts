import paletteJson from "./palette.json";

export const PALETTE = paletteJson as Record<string, string>;

export function hexToInt(hex: string): number {
  return parseInt(hex.replace("#", ""), 16);
}
