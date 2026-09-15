import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const palette = JSON.parse(
  readFileSync(new URL("../web/src/theme/palette.json", import.meta.url)),
);
const html = readFileSync(new URL("../docs/overview/architecture.html", import.meta.url), "utf8");

// Values architecture.html's dark-mode block is expected to copy verbatim from
// palette.json. Add an entry here whenever a shared primitive is added to a new
// doc custom property — this only covers properties meant to be identical, not
// architecture.html's docs-only semantic colors (--sheet, --rule, --sim, etc.).
const SHARED = [
  ["--bg", "navyDeep"],
  ["--ink", "inkBright"],
  ["--learned", "amber"],
  ["--frozen", "successTeal"],
];

function extractDarkBlockValue(cssVar) {
  // Matches ":root[data-theme=\"dark\"] { ... --bg: #071b26; ... }" — the
  // explicit dark override block, not the prefers-color-scheme media query
  // (both must exist and agree, but this checks the always-present explicit one).
  const blockMatch = html.match(/:root\[data-theme="dark"\]\s*\{([^}]*)\}/);
  if (!blockMatch) return null;
  const propMatch = blockMatch[1].match(new RegExp(`${cssVar}:\\s*(#[0-9a-fA-F]{6})`));
  return propMatch ? propMatch[1].toLowerCase() : null;
}

let failures = [];
for (const [cssVar, paletteKey] of SHARED) {
  const expected = palette[paletteKey].toLowerCase();
  const actual = extractDarkBlockValue(cssVar);
  if (actual !== expected) {
    failures.push(`${cssVar}: architecture.html has ${actual ?? "(missing)"}, palette.json's ${paletteKey} is ${expected}`);
  }
}

if (failures.length) {
  console.error("architecture.html's dark-mode tokens have drifted from web/src/theme/palette.json:");
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`OK: architecture.html's ${SHARED.length} shared dark-mode tokens match palette.json`);
