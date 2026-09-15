import { PALETTE } from "./tokens";

// Maps each CSS custom property to its palette.json key. Add a mapping here
// whenever a new --bp-* variable is introduced instead of hardcoding it in
// style.css's :root block, which now only holds fallback values.
const CSS_VAR_MAP: Record<string, string> = {
  "--bp-bg": "navyDeep",
  "--bp-surface": "navySurface",
  "--bp-border": "navyBorder",
  "--bp-grid": "cyanGrid",
  "--bp-line": "cyanLine",
  "--bp-ink": "inkBright",
  "--bp-ink-dim": "inkDim",
  "--bp-amber": "amber",
  "--bp-obstacle-fill": "obstacleFill",
  "--bp-obstacle-stroke": "obstacleStroke",
  "--bp-velocity": "velocity",
  "--bp-command": "command",
  "--bp-axis-x": "axisX",
  "--bp-axis-y": "axisY",
  "--bp-axis-z": "axisZ",
  "--bp-success": "successTeal",
  "--bp-card": "card",
};

export function applyCssTokens(root: HTMLElement = document.documentElement): void {
  for (const [cssVar, key] of Object.entries(CSS_VAR_MAP)) {
    root.style.setProperty(cssVar, PALETTE[key]);
  }
}
