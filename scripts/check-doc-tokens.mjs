import { readFileSync } from "node:fs";

const palette = JSON.parse(readFileSync(new URL("../web/src/theme/palette.json", import.meta.url)));
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

function extractBlockValue(block, cssVar) {
  const propMatch = block.match(new RegExp(`${cssVar}:\\s*(#[0-9a-fA-F]{6})`));
  return propMatch ? propMatch[1].toLowerCase() : null;
}

// Matches ":root[data-theme=\"dark\"] { ... --bg: #071b26; ... }" — the explicit
// dark override block, chosen by anyone who has picked a theme.
function extractDarkBlockValue(cssVar) {
  const blockMatch = html.match(/:root\[data-theme="dark"\]\s*\{([^}]*)\}/);
  if (!blockMatch) return null;
  return extractBlockValue(blockMatch[1], cssVar);
}

// Matches the prefers-color-scheme media query's own copy of the same values,
// which serves default-dark-mode visitors who haven't made an explicit choice —
// "@media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) { ... } }".
// Hand-duplicated from the explicit block above, so it drifts independently.
function extractMediaQueryBlockValue(cssVar) {
  const mediaMatch = html.match(/@media \(prefers-color-scheme:\s*dark\)\s*\{([\s\S]*?)\n\}/);
  if (!mediaMatch) return null;
  const blockMatch = mediaMatch[1].match(/:root:not\(\[data-theme="light"\]\)\s*\{([^}]*)\}/);
  if (!blockMatch) return null;
  return extractBlockValue(blockMatch[1], cssVar);
}

let failures = [];
for (const [cssVar, paletteKey] of SHARED) {
  const expected = palette[paletteKey].toLowerCase();
  const explicit = extractDarkBlockValue(cssVar);
  const media = extractMediaQueryBlockValue(cssVar);
  if (explicit !== expected) {
    failures.push(
      `${cssVar}: architecture.html's :root[data-theme="dark"] block has ${explicit ?? "(missing)"}, palette.json's ${paletteKey} is ${expected}`,
    );
  }
  if (media !== expected) {
    failures.push(
      `${cssVar}: architecture.html's @media (prefers-color-scheme: dark) block has ${media ?? "(missing)"}, palette.json's ${paletteKey} is ${expected}`,
    );
  }
}

if (failures.length) {
  console.error(
    "architecture.html's dark-mode tokens have drifted from web/src/theme/palette.json:",
  );
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(
  `OK: architecture.html's ${SHARED.length} shared dark-mode tokens match palette.json in both the explicit :root[data-theme="dark"] block and the @media (prefers-color-scheme: dark) block`,
);
