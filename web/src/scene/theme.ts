// Blueprint/schematic palette for the Three.js scenes. The source of truth is
// web/src/theme/palette.json. Do not edit colors here — update palette.json instead.
import { hexToInt, PALETTE } from "../theme/tokens";

export const THEME = {
  bg: hexToInt(PALETTE.navyDeep),
  grid: hexToInt(PALETTE.cyanGrid),
  gridMinor: hexToInt(PALETTE.gridMinor),
  wall: hexToInt(PALETTE.cyanLine),
  wallFill: hexToInt(PALETTE.obstacleFill),
  line: hexToInt(PALETTE.cyanLine),
  ink: hexToInt(PALETTE.inkBright),
  amber: hexToInt(PALETTE.amber),
  obstacleFill: hexToInt(PALETTE.obstacleFill),
  obstacleStroke: hexToInt(PALETTE.obstacleStroke),
  targetEmissive: hexToInt(PALETTE.targetEmissive),
  velocity: hexToInt(PALETTE.velocity),
  command: hexToInt(PALETTE.command),
  axisX: hexToInt(PALETTE.axisX),
  axisY: hexToInt(PALETTE.axisY),
  axisZ: hexToInt(PALETTE.axisZ),
} as const;
