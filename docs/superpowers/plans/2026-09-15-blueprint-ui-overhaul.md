# Blueprint UI Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate the fly-drone live viewer (`web/`) and the architecture docs page
(`docs/overview/architecture.html`) under one coherent "blueprint/schematic" visual
system, without unifying away their intentional dark-cockpit / light-drafting-sheet
pairing.

**Architecture:** Extract the blueprint palette into one JSON source of truth
(`web/src/theme/palette.json`) that both the CSS custom properties and the Three.js
`THEME` object derive from at runtime, add a staleness guard for the values
`architecture.html` duplicates by hand, then work outward: fold two already-pervasive
but unnamed literal colors into the token system, componentize the single
`main.ts` template string into per-panel modules, and layer on the remaining
polish (form controls, focus states, scrollbars, one boot-sequence animation,
a real mobile drawer).

**Tech Stack:** Vite + TypeScript + Three.js (vanilla DOM, no framework), plain CSS
with custom properties, Vitest for unit tests, Playwright (`scripts/browser-check.mjs`)
for end-to-end smoke testing, Node ESM scripts for the doc-token guard.

**Spec:** `/home/gb/.claude/plans/lets-plan-a-ui-modular-clock.md`

## Global Constraints

- The dark live viewer and the light-first `architecture.html` stay an intentional
  pair — do not add a light mode to the viewer or make the docs page dark-first.
- `IBM Plex Mono` is added to the viewer's font stack for numeric readouts only
  (altitude, speed, RPM, neuron IDs) — never for labels, which stay `DM Sans`.
- Uppercase labels are kept only on true annotations (dimension callouts, section
  tags), not applied to every button/heading.
- The corner-bracket/dimension motif (`.world-frame`, `.world-dims`) stays unique to
  the flight-view frame — do not copy it onto other panels.
- Exactly one orchestrated motion sequence is added (the HUD power-on boot); every
  other animation stays as quiet, functional feedback (hover/press/drawer open).
- `scripts/browser-check.mjs` is the acceptance test for the whole live UI and
  asserts on specific DOM ids, ARIA roles/accessible names, and literal text
  (e.g. `#task`, `#tick`, `#trial` containing `"STEER TO TARGET"` /
  `"DODGE OBSTACLE"`, buttons named "Run trial", "Signals", "Brain", "Trials",
  "Pulse"/"Hold"/"Silence"/"Restore", "Left", "Place ahead", "Move aside", "Pause"/
  "Resume", "Reset trial") and a no-horizontal-scroll check at 390×844. Every task
  that touches `main.ts`'s markup must preserve every id/role/string it uses,
  verified by running `yarn browser:check` before committing.
- `web/src/main.ts:460-477`'s `TASK_LABELS`/`TASK_TRIAL_LABEL` comment already warns:
  the uppercase strings feed `#trial` and `scripts/browser-check.mjs` asserts on them
  literally — never change them incidentally while restyling.

---

## File Structure

New files:
- `web/src/theme/palette.json` — the single source of truth: named hex primitives
  (plain JSON so both the Vite/TS side and the plain-Node doc-token script can read
  it without a TS toolchain).
- `web/src/theme/tokens.ts` — typed wrapper around `palette.json` plus a
  `hexToInt()` helper for Three.js.
- `web/src/theme/apply-css-tokens.ts` — sets the `--bp-*` CSS custom properties from
  the palette at runtime.
- `web/src/theme/tokens.test.ts` — Vitest coverage for `hexToInt` and
  `applyCssTokens`.
- `scripts/check-doc-tokens.mjs` — Node script asserting `architecture.html`'s
  hand-written dark-mode custom properties still match `palette.json` for the
  values that are supposed to be identical.
- `web/src/ui/flightView.ts` — flight-view + `.world-frame`/`.world-dims`/
  `.world-bottom` markup (template function).
- `web/src/ui/hudPinned.ts` — brain-mini / fly-mini / eyes / instruments markup.
- `web/src/ui/drawer.ts` — drawer rail + all four tab panels' markup.
- `web/src/ui/footer.ts` — footer markup.

Modified files:
- `web/src/scene/theme.ts` — `THEME` derives from `web/src/theme/tokens.ts` instead
  of hardcoding hex numbers.
- `web/src/style.css` — token block becomes fallback values with a comment pointing
  at the real source; literal `#8be6d5`/`#101c23` occurrences replaced with new
  named custom properties; new rules for form controls, focus states, scrollbars,
  boot-sequence keyframes, and redesigned mobile-drawer compact variants.
- `web/src/main.ts` — drops the giant `app.innerHTML` template literal in favor of
  composing the four new `web/src/ui/*.ts` template functions; `ACTIVITY_HOT`/
  `LEGACY_VIEWPORT_BG` become named palette lookups; calls `applyCssTokens()` once
  at startup.
- `docs/overview/architecture.html` — dark-mode block comment updated to name
  `palette.json` as the source of truth for the values it duplicates.
- `package.json` — new `"test:tokens"` script, wired into `"ci"`.

No existing DOM/component test harness exists for `web/src/main.ts` (Vitest only
covers `web/src/fly/*.test.ts` physics logic). Tasks that change markup/CSS are
verified via `yarn build` (typecheck + bundle), `yarn web:dev` manual inspection,
and `yarn browser:check` (the existing Playwright smoke test) — not fabricated unit
tests for DOM structure.

---

### Task 1: Shared token source (palette.json + tokens.ts + runtime CSS apply)

**Files:**
- Create: `web/src/theme/palette.json`
- Create: `web/src/theme/tokens.ts`
- Create: `web/src/theme/apply-css-tokens.ts`
- Create: `web/src/theme/tokens.test.ts`
- Modify: `web/src/scene/theme.ts`
- Modify: `web/src/style.css:1-20`
- Modify: `web/src/main.ts:1-6` (import + call `applyCssTokens()`)

**Interfaces:**
- Produces: `PALETTE: Record<string, string>` (hex strings) from `tokens.ts`,
  `hexToInt(hex: string): number`, `applyCssTokens(root?: HTMLElement): void`.
  Later tasks (2, 7-10) read `PALETTE` for any color they need to reference from
  TypeScript.

- [ ] **Step 1: Create the palette JSON**

Extract the current `--bp-*` values from `web/src/style.css:4-20` verbatim (no
value changes in this task — task 2 fixes the one real drift found below):

```json
{
  "navyDeep": "#071b26",
  "navySurface": "#0d222c",
  "navyBorder": "#1c3947",
  "cyanGrid": "#92cce0",
  "cyanLine": "#cfeaf0",
  "inkBright": "#eaf6f9",
  "inkDim": "#7fa8b3",
  "amber": "#ffb15c",
  "obstacleFill": "#0f3747",
  "obstacleStroke": "#6fa9bd",
  "velocity": "#6fe2ff",
  "command": "#ff6fd8",
  "axisX": "#ff5c5c",
  "axisY": "#5cff7a",
  "axisZ": "#5cb0ff",
  "gridMinor": "#4d7f93",
  "targetEmissive": "#a16b15"
}
```

Save as `web/src/theme/palette.json`. `gridMinor` and `targetEmissive` are scene-only
values currently hardcoded in `web/src/scene/theme.ts` (`gridMinor: 0x4d7f93`,
`targetEmissive: 0xa16b15`) with no CSS counterpart — included here so they stop
being magic numbers too.

- [ ] **Step 2: Write `tokens.ts`**

```typescript
// web/src/theme/tokens.ts
import paletteJson from "./palette.json";

export const PALETTE = paletteJson as Record<string, string>;

export function hexToInt(hex: string): number {
  return parseInt(hex.replace("#", ""), 16);
}
```

- [ ] **Step 3: Write the failing test for `hexToInt`**

```typescript
// web/src/theme/tokens.test.ts
import { describe, expect, it } from "vitest";
import { hexToInt, PALETTE } from "./tokens";

describe("hexToInt", () => {
  it("converts a hex string to the matching integer", () => {
    expect(hexToInt("#ffb15c")).toBe(0xffb15c);
    expect(hexToInt("071b26")).toBe(0x071b26);
  });

  it("exposes every palette entry as a 6-digit hex string", () => {
    for (const value of Object.values(PALETTE)) {
      expect(value).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });
});
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `env -u PYTHONPATH yarn test tokens.test.ts` (from repo root; `yarn test` is
`vitest run --root web`)
Expected: FAIL — `apply-css-tokens.ts` doesn't exist yet, but `tokens.ts` itself
should already pass since it has no other dependency. If `tokens.test.ts` passes
immediately, that's fine — move to step 5's test instead, which does start failing
(no `apply-css-tokens.ts` yet).

- [ ] **Step 5: Write `apply-css-tokens.ts`**

```typescript
// web/src/theme/apply-css-tokens.ts
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
};

export function applyCssTokens(root: HTMLElement = document.documentElement): void {
  for (const [cssVar, key] of Object.entries(CSS_VAR_MAP)) {
    root.style.setProperty(cssVar, PALETTE[key]);
  }
}
```

Add the matching test to `tokens.test.ts`:

```typescript
import { applyCssTokens } from "./apply-css-tokens";

describe("applyCssTokens", () => {
  it("sets every --bp-* custom property to its palette value", () => {
    const el = document.createElement("div");
    applyCssTokens(el);
    expect(el.style.getPropertyValue("--bp-bg")).toBe(PALETTE.navyDeep);
    expect(el.style.getPropertyValue("--bp-amber")).toBe(PALETTE.amber);
  });
});
```

(Vitest's default environment is Node; if `document` is unavailable, add
`// @vitest-environment jsdom` as the first line of `tokens.test.ts` — check
`web/vitest.config.ts` / `vite.config.ts` first, since a jsdom environment may
already be configured for other tests.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `env -u PYTHONPATH yarn test tokens.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 7: Wire `applyCssTokens()` into `main.ts` and derive `THEME` from `tokens.ts`**

In `web/src/main.ts`, add near the top (after the existing imports at line 6):

```typescript
import { applyCssTokens } from "./theme/apply-css-tokens";
applyCssTokens();
```

Call this before `app.innerHTML = ...` (line 105) so the CSS variables are correct
before first paint.

Rewrite `web/src/scene/theme.ts` to derive every matching value from the palette,
keeping only the values that have no CSS counterpart as local constants:

```typescript
// web/src/scene/theme.ts
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
```

Note this fixes a real drift: the old `theme.ts` had `grid: 0x92ccdc`, but
`style.css`'s `--bp-grid` was `#92cce0` — the last two hex digits disagreed
(`dc` vs `e0`). Deriving both from one `cyanGrid` value in `palette.json` makes
them match; `#92cce0` (the CSS value) is treated as canonical since it's the one
visible on more surfaces.

Update the header comments in both files: `style.css:1-3` and the old
`scene/theme.ts:1-3` "keep in sync" comments are replaced with a single comment in
each pointing at `web/src/theme/palette.json` as the source of truth. In
`style.css`, change the `:root { --bp-bg: #071b26; ... }` block's leading comment
to:

```css
/* Fallback values only — the source of truth is web/src/theme/palette.json,
   applied at runtime by applyCssTokens() (web/src/theme/apply-css-tokens.ts).
   These are also what architecture.html's dark-mode block copies by hand;
   scripts/check-doc-tokens.mjs (Task 3) guards that copy against drift. */
```

- [ ] **Step 8: Verify the build and visual output**

Run: `env -u PYTHONPATH yarn typecheck && env -u PYTHONPATH yarn build`
Expected: no type errors; bundle succeeds.

Run: `env -u PYTHONPATH yarn web:dev`, open the viewer, confirm the flight view,
grid, drone, brain graph, and HUD panels render with the same colors as before
(this task changes no values except the one `grid` hex-drift fix, which is a
one-digit shift not visible to the eye).

- [ ] **Step 9: Commit**

```bash
git add web/src/theme web/src/scene/theme.ts web/src/style.css web/src/main.ts
git commit -m "Add a single palette source of truth for the blueprint theme"
```

---

### Task 2: Fold the pervasive-but-unnamed literal colors into named tokens

Two colors are already used repeatedly across `style.css` and `main.ts` as raw hex
literals with no name: `#8be6d5` (a teal used for `button.primary`, active states,
pass/ok outcomes, and — under the comment "outside the blueprint-schematic scope" —
the brain-activity pulse `ACTIVITY_HOT`) and `#101c23` (the card/panel background,
also reused verbatim as `LEGACY_VIEWPORT_BG` for the fly mini-panel). Both
"off-theme" values called out in `main.ts:156-159` turn out to already be exactly
the site's existing accent/surface colors — the code comment overstated the
divergence. Naming them removes the last unnamed magic literals and lets Task 1's
guard cover them too.

**Files:**
- Modify: `web/src/theme/palette.json` (add `successTeal`, `card`)
- Modify: `web/src/style.css` (16 occurrences: lines 59, 149, 151, 174, 216, 218,
  410, 427, 428, 474, 512, 536, 556, 587, 589, 629 per the grep below)
- Modify: `web/src/main.ts:156-159, 350, 699` (`ACTIVITY_HOT`, `LEGACY_VIEWPORT_BG`)

**Interfaces:**
- Consumes: `PALETTE` from Task 1.
- Produces: `--bp-success` and `--bp-card` CSS custom properties, added to
  `CSS_VAR_MAP` in `apply-css-tokens.ts`.

- [ ] **Step 1: Add the two entries to `palette.json`**

```json
  "successTeal": "#8be6d5",
  "card": "#101c23"
```

(append to the existing object from Task 1)

- [ ] **Step 2: Add them to `CSS_VAR_MAP` in `apply-css-tokens.ts`**

```typescript
  "--bp-success": "successTeal",
  "--bp-card": "card",
```

Add the matching fallback declarations to `style.css`'s `:root` block:

```css
  --bp-success: #8be6d5;
  --bp-card: #101c23;
```

- [ ] **Step 3: Replace the literals in `style.css`**

Run to find every occurrence first: `grep -n "8be6d5\|101c23" web/src/style.css`.
Replace each `background: #8be6d5` / `border-color: #8be6d5` / `color: #8be6d5`
with the same property using `var(--bp-success)`, and each `background: #101c23`
with `var(--bp-card)`. This touches: `.card` (line 59), `.camera-controls
button.active` (149, 151), `.brain-legend i` (174), `button.primary` (216, 218),
`.outcome.pass` (410), `.seed-grid .seed.pass` (427, 428), `.roam-log-entry.ok`
(474), `.hud-pinned .brain-mini` (512), `.fly-mini-panel` (536), `.eyes-panel`
(556), `.drawer-rail button.active` (587, 589), `.instruments` (629).

- [ ] **Step 4: Replace the literals in `main.ts`**

```typescript
import { PALETTE, hexToInt } from "./theme/tokens";
// ...
const ACTIVITY_HOT = hexToInt(PALETTE.successTeal);
const LEGACY_VIEWPORT_BG = hexToInt(PALETTE.card);
```

Update the comment above them (`main.ts:156-157`) since it's no longer accurate:

```typescript
// The brain-activity pulse and the fly viewport's background share the site's
// existing accent/surface tokens rather than introducing new colors.
```

- [ ] **Step 5: Verify**

Run: `env -u PYTHONPATH yarn typecheck && env -u PYTHONPATH yarn build`
Run: `env -u PYTHONPATH yarn web:dev`, visually confirm every button/active-state/
pass-outcome/card background is pixel-identical to before (values are unchanged,
only their source moved).

- [ ] **Step 6: Commit**

```bash
git add web/src/theme/palette.json web/src/theme/apply-css-tokens.ts web/src/style.css web/src/main.ts
git commit -m "Name the two pervasive accent/surface colors instead of hardcoding them"
```

---

### Task 3: Doc-token staleness guard for architecture.html

`architecture.html`'s `:root[data-theme="dark"]` block (and its
`prefers-color-scheme: dark` twin) already duplicates several `palette.json`
values by hand: `--bg` (`#071b26` = `navyDeep`), `--ink` (`#eaf6f9` = `inkBright`),
`--learned` (`#ffb15c` = `amber`), `--frozen` (`#8be6d5` = `successTeal`, from
Task 2). Nothing enforces that they stay equal. Rather than inventing a generated
CSS include for a semantically different token set (docs uses `--frozen`/
`--learned`/`--sim`/`--pass`/`--fail`, roles that don't map 1:1 onto every `--bp-*`
token), add a guard script that fails loudly the moment one of these four shared
values drifts.

**Files:**
- Create: `scripts/check-doc-tokens.mjs`
- Modify: `package.json` (new `"test:tokens"` script, add to `"ci"`)
- Modify: `docs/overview/architecture.html:32-36, 38-43` (comment only)

**Interfaces:**
- Consumes: `web/src/theme/palette.json` (read directly as JSON — no TS
  compilation needed, keeping this script dependency-free).
- Produces: exit code 0 / non-zero for `yarn test:tokens`.

- [ ] **Step 1: Write the script**

```javascript
// scripts/check-doc-tokens.mjs
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
```

- [ ] **Step 2: Add the npm script**

In `package.json`, add alongside the existing `"test"` entry:

```json
    "test:tokens": "node scripts/check-doc-tokens.mjs",
```

Add `&& yarn test:tokens` to the `"ci"` script, right after the existing
`&& yarn test` segment.

- [ ] **Step 3: Update the architecture.html comment**

Change `docs/overview/architecture.html:10` (and the dark-block's own leading
comment if any) from "Tokens follow web/src/style.css (--bp-*)" to:

```html
<!-- --bg/--ink/--learned/--frozen in the dark-mode block below must match
     web/src/theme/palette.json's navyDeep/inkBright/amber/successTeal —
     scripts/check-doc-tokens.mjs (yarn test:tokens) enforces this. -->
```

placed directly above the `:root[data-theme="dark"]` rule (`architecture.html:38`).

- [ ] **Step 4: Run it**

Run: `node scripts/check-doc-tokens.mjs`
Expected: `OK: architecture.html's 4 shared dark-mode tokens match palette.json`
(all four already agree today, per the audit above — this is a regression guard,
not a fix).

- [ ] **Step 5: Commit**

```bash
git add scripts/check-doc-tokens.mjs package.json docs/overview/architecture.html
git commit -m "Guard architecture.html's dark-mode tokens against drift from palette.json"
```

---

### Task 4: Custom form controls

Native `<select>` (`#task`, `#ablation`, `#report`, `#neuron`) and
`<input type="number">` (`#seed`) currently keep the browser's default dropdown
arrow and number spinner (`web/src/style.css:190-199, 384-403`), which breaks the
otherwise fully custom instrument-panel look.

**Files:**
- Modify: `web/src/style.css:190-199` (`select`), `370-403` (`.replay-form select,
  input`)

**Interfaces:** none (pure CSS; no markup or TS changes).

- [ ] **Step 1: Restyle the base `select` rule**

Replace `web/src/style.css:190-199`:

```css
select {
  font-family: inherit;
  font-size: 9px;
  background-color: #13232c;
  color: #a7bdc6;
  border: 1px solid #2f444e;
  border-radius: 4px;
  width: 100%;
  padding: 4px 20px 4px 6px;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='6' viewBox='0 0 8 6'%3E%3Cpath d='M0 0 L8 0 L4 6 Z' fill='%2392cce0'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 7px center;
}
```

- [ ] **Step 2: Restyle `.replay-form select, input` and the number input specifically**

Replace `web/src/style.css:384-396`:

```css
.replay-form select,
.replay-form input {
  min-width: 0;
  max-width: 100%;
  background-color: #0d171e;
  color: inherit;
  border: 1px solid #2c4550;
  border-radius: 6px;
  padding: 6px 8px;
  transition:
    border-color 0.2s,
    box-shadow 0.2s;
}
.replay-form input[type="number"] {
  appearance: textfield;
}
.replay-form input[type="number"]::-webkit-inner-spin-button,
.replay-form input[type="number"]::-webkit-outer-spin-button {
  appearance: none;
  margin: 0;
}
```

(`.replay-form select` inherits the new chevron background from the base `select`
rule above since it doesn't override `background-image`; only `background-color`
is set here.)

- [ ] **Step 3: Verify**

Run: `env -u PYTHONPATH yarn web:dev`, open the Trials & Replay panel, confirm the
task/ablation/report dropdowns show the new chevron and no native arrow, and the
seed number input has no spinner but still accepts typed numbers and the seed-grid
buttons still populate it correctly.

Run: `env -u PYTHONPATH yarn browser:check` — the existing test types into `#seed`
via `page.fill("#seed", "1003")` and selects via `page.selectOption`, both of which
are unaffected by `appearance`/`background-image` styling, so this should still
pass unchanged.

- [ ] **Step 4: Commit**

```bash
git add web/src/style.css
git commit -m "Restyle native select/number-input chrome to match the instrument panel"
```

---

### Task 5: Consistent focus-visible states

Only `.replay-form select/input` (and the bare `select` selector, redundantly) have
a `focus-visible` ring today (`web/src/style.css:397-403`). Buttons, tab buttons,
camera-mode controls, and the drawer-collapse toggle have no defined focus style.

**Files:**
- Modify: `web/src/style.css:397-403` (consolidate), add a new global rule near the
  top-level `button`/`select` rules (~line 214, after the existing `button:hover`).

**Interfaces:** none.

- [ ] **Step 1: Add one token-driven rule covering every interactive element**

Insert after `web/src/style.css:214` (`button:hover { ... }`):

```css
button:focus-visible,
select:focus-visible,
input:focus-visible {
  outline: 2px solid var(--bp-grid);
  outline-offset: 2px;
}
```

- [ ] **Step 2: Remove the now-redundant narrower rule**

Delete the old `web/src/style.css:397-403` block:

```css
.replay-form select:focus-visible,
.replay-form input:focus-visible,
select:focus-visible {
  outline: none;
  border-color: var(--bp-grid);
  box-shadow: 0 0 0 2px rgba(146, 204, 224, 0.18);
}
```

(The new global rule already covers `.replay-form select`/`input` and every other
`select`/`input`/`button`; this block's `border-color`+`box-shadow` treatment is
replaced by the same `outline` treatment everywhere for consistency.)

- [ ] **Step 3: Verify**

Run: `env -u PYTHONPATH yarn web:dev`. Tab through the page with the keyboard:
camera-mode buttons, the drawer-collapse toggle, drawer-rail tab buttons, every
form control, and the neuron-inspector buttons should all show a visible cyan
outline on focus, and it should disappear on mouse click (that's what
`:focus-visible` vs `:focus` gives you for free).

- [ ] **Step 4: Commit**

```bash
git add web/src/style.css
git commit -m "Give every interactive element a consistent focus-visible ring"
```

---

### Task 6: Themed scrollbars

`.roam-log` (`web/src/style.css:457-464`) and `.drawer-body`
(`web/src/style.css:615-624`) scroll with the OS-default scrollbar.

**Files:**
- Modify: `web/src/style.css:457-464`, `615-624`

**Interfaces:** none.

- [ ] **Step 1: Add scrollbar styling to both rules**

Append to `.roam-log` (line 464, before the closing brace) and to `.drawer-body`
(line 624, before the closing brace):

```css
  scrollbar-width: thin;
  scrollbar-color: var(--bp-border) var(--bp-surface);
```

Add matching WebKit rules right after each block:

```css
.roam-log::-webkit-scrollbar,
.drawer-body::-webkit-scrollbar {
  width: 8px;
}
.roam-log::-webkit-scrollbar-track,
.drawer-body::-webkit-scrollbar-track {
  background: var(--bp-surface);
}
.roam-log::-webkit-scrollbar-thumb,
.drawer-body::-webkit-scrollbar-thumb {
  background: var(--bp-border);
  border-radius: 4px;
}
```

- [ ] **Step 2: Verify**

Run: `env -u PYTHONPATH yarn web:dev`, open the Free Roam tab with enough events to
overflow `.roam-log`, and resize the drawer body to overflow — confirm the
scrollbar tracks/thumbs are dark blueprint colors, not the OS default, in a
Chromium browser (WebKit rules) and that `scrollbar-width`/`scrollbar-color` apply
in Firefox.

- [ ] **Step 3: Commit**

```bash
git add web/src/style.css
git commit -m "Theme the drawer body and roam log scrollbars"
```

---

### Task 7: Componentize main.ts — flight view + pinned HUD stack

Extract the flight-view (`.world`) and pinned HUD stack (`.hud-pinned`) markup out
of `main.ts`'s single template literal (currently `web/src/main.ts:106-136`) into
their own template modules, each a plain function returning an HTML string, so
future styling touches one focused file instead of the 883-line `main.ts`.

**Files:**
- Create: `web/src/ui/flightView.ts`
- Create: `web/src/ui/hudPinned.ts`
- Modify: `web/src/main.ts:104-154` (composition), no changes to any `el(...)`
  lookups or update functions — every id these templates emit must stay identical.

**Interfaces:**
- Produces: `renderFlightView(): string`, `renderHudPinned(): string` — pure
  functions, no arguments, no DOM access, called once at startup.
- Consumes: nothing (they emit static markup; all dynamic values are filled in
  later by the existing `el(id)`-based update functions, unchanged).

- [ ] **Step 1: Extract the flight-view template**

```typescript
// web/src/ui/flightView.ts
export function renderFlightView(): string {
  return `
<div class="world">
<div id="world" class="viewport"></div>
<div class="connection"><i id="dot"></i><span id="status">Connecting to simulation</span></div>
<div class="world-bottom"><div><span>ALTITUDE</span><strong id="altitude">—<small> m</small></strong></div><div><span>SPEED</span><strong id="speed">—<small> m/s</small></strong></div><div><span>VX / VY / VZ</span><strong id="velocity-axes">—<small> m/s</small></strong></div><div><span>SIMULATION</span><strong id="simtime">0.00<small> s</small></strong></div><div><span>REAL TIME</span><strong id="rtf">—<small> ×</small></strong></div><div class="camera-controls"><button id="cam-follow" title="Keep the camera target locked to the drone">Follow</button><button id="cam-recenter" title="Snap the camera target to the drone once">Recenter</button><button id="cam-reset" title="Restore the default orbit view">Reset view</button><button id="cam-fpv" title="Ride along in the drone's cockpit">1st person</button><button id="cam-tpv" title="Chase camera behind the drone">3rd person</button></div></div>
<div class="world-frame"><i></i><i></i><i></i><i></i></div>
<div class="world-dims" id="dims"></div>
<div class="world-hint">DRAG TO ORBIT · SCROLL TO ZOOM</div>
</div>`;
}
```

(Copied verbatim from `web/src/main.ts:107-114` — no id, class, or text changes.)

- [ ] **Step 2: Extract the pinned HUD stack template**

```typescript
// web/src/ui/hudPinned.ts
export function renderHudPinned(): string {
  return `
<section class="hud-pinned" id="hud-pinned">
<div class="brain-mini"><div class="brain-mini-head"><span id="mode">INITIALIZING</span><span id="tick">TICK 0</span></div><div id="brain" class="viewport"></div><div class="brain-legend"><span><i></i> measured activity</span><span>connections</span></div></div>
<div class="fly-mini-panel" title="Same neural readouts, independent trajectory — illustrative fly dynamics, not calibrated biomechanics."><div class="fly-mini-head"><span>FLY BODY</span></div><div id="fly" class="viewport"></div></div>
<div class="eyes-panel"><div class="eyes"><figure><img id="eye0" alt="Left simulated eye"><figcaption>LEFT EYE</figcaption></figure><figure><img id="eye1" alt="Right simulated eye"><figcaption>RIGHT EYE</figcaption></figure></div></div>
<div class="instruments">
<div class="attitude-gauges">
<div class="attitude-gauge"><div class="track"><i class="fill" id="roll-fill"></i><i class="needle" id="roll-needle"></i></div><span>ROLL</span><em id="roll-value">—</em></div>
<div class="attitude-gauge"><div class="track"><i class="fill" id="pitch-fill"></i><i class="needle" id="pitch-needle"></i></div><span>PITCH</span><em id="pitch-value">—</em></div>
<div class="attitude-gauge"><div class="track"><i class="fill" id="yaw-fill"></i><i class="needle" id="yaw-needle"></i></div><span>YAW</span><em id="yaw-value">—</em></div>
</div>
<div class="turn-rate"><span>TURN RATE</span><em id="turn-rate">—</em></div>
<div class="vector-legend">
<span><i style="background:#ffb15c"></i><span class="legend-label">heading</span></span>
<span><i style="background:#6fe2ff"></i><span class="legend-label">velocity</span></span>
<span><i style="background:#ff6fd8"></i><span class="legend-label">command</span></span>
<span><i style="background:#ff5c5c"></i><span class="legend-label">world X</span></span>
<span><i style="background:#5cff7a"></i><span class="legend-label">world Z↑</span></span>
<span><i style="background:#5cb0ff"></i><span class="legend-label">world -Y</span></span>
</div>
</div>
</section>`;
}
```

(Copied verbatim from `web/src/main.ts:116-136`. The inline vector-legend swatch
colors stay as inline styles for now — Task 8 is a natural place to revisit them,
but this task's scope is extraction only, not re-theming, per the plan's rule that
each task should be independently reviewable for exactly what it claims to do.)

- [ ] **Step 3: Compose them in `main.ts`**

Replace `web/src/main.ts:104-154`'s single template literal with:

```typescript
import { renderFlightView } from "./ui/flightView";
import { renderHudPinned } from "./ui/hudPinned";
// ... (drawer/footer imports land in Task 8/9's edits to this same block)

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
<main>
${renderFlightView()}
<div class="hud-layer">
${renderHudPinned()}
<div class="drawer" id="drawer">
<!-- drawer body markup stays inline until Task 8 extracts it -->
${DRAWER_BODY_PLACEHOLDER_FROM_ORIGINAL_LINES_137_144}
</div>
<div class="drawer-rail" id="drawer-rail">
<!-- drawer rail markup stays inline until Task 8 extracts it -->
${DRAWER_RAIL_PLACEHOLDER_FROM_ORIGINAL_LINES_145_151}
</div>
</div>
</main>
<footer><span>ANATOMICAL WIRING · MODELED NEURONS · LEARNED DECODING</span><span>MaleCNS v1.0 · FlyEM / Cambridge / MRC LMB / Google Research · CC-BY 4.0</span></footer>`;
```

Concretely: keep `web/src/main.ts:137-151`'s drawer/drawer-rail markup exactly as
it is inline in this task (that's Task 8's extraction target) — only swap the
flight-view and hud-pinned chunks for the two new function calls, and add the two
new imports. The rest of `main.ts` (everything from `const el = ...` at line 155
onward) is untouched by this task.

- [ ] **Step 4: Verify**

Run: `env -u PYTHONPATH yarn typecheck && env -u PYTHONPATH yarn build`
Run: `env -u PYTHONPATH yarn web:dev` and confirm the flight view, camera controls,
brain graph, fly panel, eyes, and attitude gauges all render and update exactly as
before (this task is a pure extraction — the rendered DOM must be byte-identical).
Run: `env -u PYTHONPATH yarn browser:check` — must still fully pass, since none of
the ids/roles/text it asserts on changed.

- [ ] **Step 5: Commit**

```bash
git add web/src/ui/flightView.ts web/src/ui/hudPinned.ts web/src/main.ts
git commit -m "Extract flight-view and pinned-HUD markup into their own modules"
```

---

### Task 8: Componentize main.ts — drawer tabs and footer

Extract the remaining markup: the drawer body (all four tab panels:
Trials & Replay, Neuron Inspector, Free Roam, Signals & Controls), the drawer rail,
and the footer (`web/src/main.ts:137-154` in the pre-Task-7 numbering, now the
inline chunk left in place by Task 7's step 3).

**Files:**
- Create: `web/src/ui/drawer.ts`
- Create: `web/src/ui/footer.ts`
- Modify: `web/src/main.ts` (composition, replacing the two placeholder-inline
  chunks Task 7 left in place)

**Interfaces:**
- Produces: `renderDrawer(): string` (drawer body + drawer rail together, since
  they're structurally coupled — the rail's tab buttons and the body's
  `data-panel` sections must stay in sync), `renderFooter(): string`.
- Consumes: nothing (static markup, same as Task 7).

- [ ] **Step 1: Extract the drawer template**

```typescript
// web/src/ui/drawer.ts
export function renderDrawer(): string {
  return `
<div class="drawer" id="drawer">
<div class="drawer-body">
<section class="card replay" data-panel="replay"><div class="panel-head"><span class="panel-title">Trials &amp; Replay</span><span class="panel-meta" id="trial">SEED 42 · VISUAL · INTACT</span></div><div class="replay-grid"><div><h3>RUN A TRIAL</h3><div class="replay-form"><label>Task<select id="task"></select></label><label>Brain<select id="ablation"><option value="none">Intact</option><option value="zero">Zeroed features</option><option value="sensory">Vision silenced</option><option value="shuffle">Shuffled features</option></select></label><label>Seed<input id="seed" type="number" value="1000" min="0" step="1"></label><button id="run" class="primary">Run trial</button></div><p id="outcome" class="outcome">Outcome: —</p></div><div><h3>REPLAY AN EVALUATION</h3><div class="replay-form"><label>Report<select id="report"><option value="">No reports loaded</option></select></label></div><p id="report-summary" class="outcome"></p><div id="seeds" class="seed-grid" aria-label="Evaluation seeds"></div></div></div></section>
<section class="card brain-card" data-panel="brain" hidden><div class="panel-head"><span class="panel-title">Neuron Inspector</span></div><div class="inspect"><select id="neuron" aria-label="Neuron to inspect"><option>Loading neurons…</option></select><div class="button-row"><button data-op="pulse">Pulse</button><button data-op="hold">Hold</button><button data-op="silence">Silence</button><button data-op="restore">Restore</button></div></div></section>
<section class="card roam-hud" id="roam-hud" data-panel="roam" hidden><div class="panel-head"><span class="panel-title">Free Roam</span><span class="panel-meta" id="roam-level">LEVEL —</span></div><div class="roam-stats"><div><span>BEACONS / MIN</span><strong id="roam-beacons">—</strong></div><div><span>COLLISIONS / MIN</span><strong id="roam-collisions">—</strong></div><div><span>THREATS DODGED</span><strong id="roam-dodged">—</strong></div><div><span>THREATS HIT</span><strong id="roam-hit">—</strong></div><div><span>CELLS VISITED</span><strong id="roam-cells">—</strong></div></div><div class="roam-log" id="roam-log" aria-label="Free roam event log"></div></section>
<section class="card signals" data-panel="signals" hidden><div class="panel-head"><span class="panel-title">Signals &amp; Controls</span><span class="panel-meta" id="episode">EPISODE 0</span></div><div class="signal-grid"><div><h3>SENSORY CURRENT</h3><div id="cues" class="meters"></div></div><div><h3>NEURAL READOUT</h3><div id="readouts" class="meters"></div></div><div><h3>ACTUAL / COMMANDED RPM</h3><div id="motors" class="meters"></div></div></div><div class="controls"><div><h3>EXPERIMENT</h3><div class="button-row"><button id="pause" class="primary">Pause</button><button id="reset">Reset trial</button></div></div><div><h3>VISUAL TARGET</h3><div class="button-row"><button data-target="left">Left</button><button data-target="center">Center</button><button data-target="right">Right</button></div></div><div><h3>OBSTACLE</h3><div class="button-row"><button id="loom">Place ahead</button><button id="clear">Move aside</button></div></div></div><div class="notes"><span id="command">Motion command: —</span><span id="error">Waiting for the local Rust + MuJoCo service.</span></div></section>
</div>
</div>
<div class="drawer-rail" id="drawer-rail">
<button class="drawer-collapse" id="drawer-collapse" aria-label="Toggle panel drawer" title="Toggle panel"><i>‹</i></button>
<button data-tab="replay" class="tab-btn active">Trials</button>
<button data-tab="brain" class="tab-btn">Brain</button>
<button data-tab="roam" class="tab-btn" id="roam-tab" hidden>Roam</button>
<button data-tab="signals" class="tab-btn">Signals</button>
</div>`;
}
```

(Copied verbatim from `web/src/main.ts:139-151` in the current, pre-Task-7 file.
Every id/data-attribute/aria-label/button text is unchanged, since
`scripts/browser-check.mjs` and the update functions in `main.ts` depend on all of
them exactly.)

- [ ] **Step 2: Extract the footer**

```typescript
// web/src/ui/footer.ts
export function renderFooter(): string {
  return `<footer><span>ANATOMICAL WIRING · MODELED NEURONS · LEARNED DECODING</span><span>MaleCNS v1.0 · FlyEM / Cambridge / MRC LMB / Google Research · CC-BY 4.0</span></footer>`;
}
```

- [ ] **Step 3: Compose the final `main.ts` template**

```typescript
import { renderFlightView } from "./ui/flightView";
import { renderHudPinned } from "./ui/hudPinned";
import { renderDrawer } from "./ui/drawer";
import { renderFooter } from "./ui/footer";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
<main>
${renderFlightView()}
<div class="hud-layer">
${renderHudPinned()}
${renderDrawer()}
</div>
</main>
${renderFooter()}`;
```

This fully replaces the original `web/src/main.ts:104-154` block. Every id lookup
via `el(...)` later in the file (e.g. `el("task")`, `el("drawer")`,
`el("roam-tab")`) resolves to the same element it did before, since no id changed.

- [ ] **Step 4: Verify**

Run: `env -u PYTHONPATH yarn typecheck && env -u PYTHONPATH yarn build`
Run: `env -u PYTHONPATH yarn web:dev`, click through all four drawer tabs, run a
trial, replay a report, pulse/hold/silence/restore a neuron, pause/reset, place/
clear an obstacle — every control should behave exactly as before.
Run: `env -u PYTHONPATH yarn browser:check` — must fully pass; this is the task
most likely to silently break a selector, since it moves the largest markup chunk.

- [ ] **Step 5: Commit**

```bash
git add web/src/ui/drawer.ts web/src/ui/footer.ts web/src/main.ts
git commit -m "Extract drawer tabs and footer markup into their own modules"
```

---

### Task 9: HUD power-on boot sequence

Add the single deliberate motion moment: on load, the flight-view frame,
instruments, and brain/eyes stack sweep into view in sequence (flight frame →
instruments → brain/eyes → drawer), evoking avionics warm-up, instead of
appearing instantly.

**Files:**
- Modify: `web/src/style.css` (new keyframes + boot classes, add near the
  `.world-frame` block at the end of the file)
- Modify: `web/src/main.ts` (trigger the sequence once, after `app.innerHTML` is
  set)

**Interfaces:**
- Produces: a `runBootSequence(): void` function in `main.ts`, called once at
  startup after the DOM is built (no other module needs to call it — it's a
  fire-and-forget presentation effect, not app state).

- [ ] **Step 1: Add the boot keyframes and classes to `style.css`**

Append at the end of `web/src/style.css`:

```css
/* One-time HUD power-on: elements start dim/offset and settle into place in a
   staggered sequence, the single orchestrated motion moment (see AGENTS.md-level
   design constraint: everything else stays quiet, functional feedback only).
   Respect reduced-motion by skipping straight to the settled state. */
@keyframes bp-boot-in {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
.bp-boot {
  opacity: 0;
}
.bp-boot.bp-boot-run {
  animation: bp-boot-in 0.45s ease-out forwards;
}
@media (prefers-reduced-motion: reduce) {
  .bp-boot {
    opacity: 1;
    animation: none;
  }
}
```

- [ ] **Step 2: Mark the four boot targets and trigger the sequence in `main.ts`**

Add the `bp-boot` class to four root elements at the point they're built:
`.world-frame` (in `renderFlightView`), `.instruments` (in `renderHudPinned`),
`.hud-pinned .brain-mini` + `.hud-pinned .eyes-panel` (also in `renderHudPinned`,
sharing the same stagger group as instruments' surrounding section), and
`.drawer` (in `renderDrawer`). Concretely, in `web/src/ui/flightView.ts` change
`<div class="world-frame">` to `<div class="world-frame bp-boot">`; in
`web/src/ui/hudPinned.ts` change `<div class="instruments">` to
`<div class="instruments bp-boot">` and the outer `<section class="hud-pinned"
id="hud-pinned">` gets `bp-boot` added too (covering brain-mini/fly-mini/eyes as
one group so they settle together); in `web/src/ui/drawer.ts` change
`<div class="drawer" id="drawer">` to `<div class="drawer bp-boot" id="drawer">`.

Add to `web/src/main.ts`, right after the `app.innerHTML = ...` assignment:

```typescript
function runBootSequence(): void {
  const sequence = [".world-frame", "#hud-pinned", "#drawer"];
  sequence.forEach((selector, i) => {
    const el = document.querySelector<HTMLElement>(selector);
    if (!el) return;
    setTimeout(() => el.classList.add("bp-boot-run"), i * 120);
  });
}
runBootSequence();
```

- [ ] **Step 3: Verify**

Run: `env -u PYTHONPATH yarn web:dev`, reload the page a few times, confirm the
flight-view frame brackets appear first, then the pinned brain/eyes/instruments
stack, then the drawer, each with a brief fade/rise — and confirm nothing else
(buttons, meters, tab switches) got a stray animation. Enable "reduce motion" in
OS/browser settings and reload — everything should appear instantly instead.
Run: `env -u PYTHONPATH yarn browser:check` — the test screenshots and asserts
after `connected(page)` resolves, which happens well after the ~360ms boot
sequence finishes, so this shouldn't introduce flakiness; run it a couple of times
to confirm.

- [ ] **Step 4: Commit**

```bash
git add web/src/style.css web/src/main.ts web/src/ui/flightView.ts web/src/ui/hudPinned.ts web/src/ui/drawer.ts
git commit -m "Add a one-time HUD power-on boot sequence"
```

---

### Task 10: Redesigned mobile drawer

At `max-width: 700px`, `web/src/style.css:765-768` currently just hides turn-rate
and the vector legend outright, and `753-755` hides the fly-mini-panel entirely,
rather than giving them a real compact layout.

**Files:**
- Modify: `web/src/style.css:728-811` (the existing `@media (max-width: 700px)`
  block)

**Interfaces:** none.

- [ ] **Step 1: Replace the turn-rate/vector-legend hide rules with compact variants**

Replace `web/src/style.css:765-768`:

```css
  .turn-rate,
  .vector-legend {
    display: none;
  }
```

with:

```css
  .turn-rate {
    margin-top: 5px;
    font-size: 6px;
  }
  .vector-legend {
    flex-direction: row;
    flex-wrap: wrap;
    gap: 4px 8px;
    margin-top: 6px;
  }
  .vector-legend .legend-label {
    display: none;
  }
```

(The `.vector-legend .legend-label { display: none; }` rule already existed at
`web/src/style.css:805-807` for a wider mobile breakpoint's swatch-only mode —
remove that now-duplicate rule since it's folded in here.)

- [ ] **Step 2: Give the fly-mini-panel a compact variant instead of hiding it**

Replace `web/src/style.css:753-755`:

```css
  .fly-mini-panel {
    display: none;
  }
```

with:

```css
  .fly-mini-panel {
    width: 100%;
  }
  .fly-mini-panel #fly {
    height: 54px;
  }
  .fly-mini-head {
    height: 18px;
    font-size: 6px;
  }
```

Update the comment above it (`web/src/style.css:749-752`, "Only the brain graph and
eyes are load-bearing...") to reflect the new behavior:

```css
  /* The fly body panel shrinks instead of disappearing — AGENTS.md only requires
     the brain-activity graph and eyes stay pinned and visible; giving the fly
     panel a real compact size (rather than display:none) keeps the pinned stack
     from feeling like a smaller, degraded version of the desktop layout. */
```

- [ ] **Step 3: Verify the pinned stack still fits and there's no horizontal scroll**

Run: `env -u PYTHONPATH yarn web:dev`, resize the browser to 390×844 (the same
viewport `scripts/browser-check.mjs` uses), confirm the pinned stack (brain, fly,
eyes, instruments with turn-rate and vector legend) all fit within `width: 130px`
(`web/src/style.css:731-736`'s existing mobile width) without overflowing or
causing horizontal scroll on the page.

Run: `env -u PYTHONPATH yarn browser:check` — it already asserts
`document.documentElement.scrollWidth <= window.innerWidth` at 390×844
(`scripts/browser-check.mjs`'s last check before `assert.deepEqual(errors, [])`);
this must still pass, since it's the exact regression this task risks.

- [ ] **Step 4: Commit**

```bash
git add web/src/style.css
git commit -m "Give the mobile pinned HUD real compact layouts instead of hiding panels"
```

---

## Self-Review

**Spec coverage:** All 8 upgrade points from the approved design map to tasks —
(1) shared token source → Task 1; (2) reconcile off-theme colors → Task 2;
(3) componentize `main.ts` → Tasks 7-8; (4) custom form controls → Task 4;
(5) focus states → Task 5; (6) themed scrollbars → Task 6; (7) boot sequence →
Task 9; (8) redesigned mobile drawer → Task 10. The design's "lighter-touch"
`architecture.html` token-pointing item is Task 3. Every Global Constraint
(dark/light pairing, mono-for-numerals restriction, uppercase-labels restriction,
corner-bracket exclusivity, single motion beat) is either directly implemented
(Task 9 for the motion beat) or stated as a constraint no task violates (the
others are properties the plan's tasks were checked against, not separate
deliverables — none of them required new code).

**Placeholder scan:** No task contains "TBD", "add appropriate styling", or
similar. Every step has literal code or an exact verification command. (Task 7's
step 3 code block does contain two ALL-CAPS `_PLACEHOLDER_` template markers, but
they are explicitly explained in prose immediately below the block as "keep the
original markup inline for now" — not an unresolved TBD; the concrete instruction
is "keep lines 137-151 exactly as they are.")

**Type consistency:** `PALETTE`/`hexToInt` (Task 1) are used with identical
signatures in Task 2 (`main.ts`) and Task 7 (`scene/theme.ts`, already done in
Task 1 itself). `renderFlightView`/`renderHudPinned`/`renderDrawer`/`renderFooter`
(Tasks 7-8) all share the `(): string` signature and are imported with matching
names in Task 8's final `main.ts` composition. `applyCssTokens(root?: HTMLElement)`
is called with no arguments in Task 1's `main.ts` wiring and with an explicit
element in its own test — both valid given the optional parameter.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-15-blueprint-ui-overhaul.md`. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
