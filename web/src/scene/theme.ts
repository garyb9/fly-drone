// Blueprint/schematic palette for the Three.js scenes. Keep this in sync with the
// --bp-* custom properties in web/src/style.css — one is CSS-side chrome, this is
// scene materials; a palette tweak should update both.
export const THEME = {
  bg: 0x071b26,
  grid: 0x92ccdc,
  gridMinor: 0x4d7f93,
  wall: 0xcfeaf0,
  wallFill: 0x0f3747,
  line: 0xcfeaf0,
  ink: 0xeaf6f9,
  amber: 0xffb15c,
  obstacleFill: 0x0f3747,
  obstacleStroke: 0x6fa9bd,
  targetEmissive: 0xa16b15,
  velocity: 0x6fe2ff,
  command: 0xff6fd8,
  axisX: 0xff5c5c,
  axisY: 0x5cff7a,
  axisZ: 0x5cb0ff,
} as const;
