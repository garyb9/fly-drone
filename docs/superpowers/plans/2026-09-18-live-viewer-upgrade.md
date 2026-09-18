# Live Viewer Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the live viewer into a free-roam instrument that shows the connectome's causal
behaviour, without reintroducing the left drawer that was deliberately removed in `2b5e0b5`.

**Scope (user-approved):** three majors — (1) Free-Roam Mission Control, (3) Live scopes +
attribution, (5) Accessibility + responsive — plus a small performance pass. Neuron Inspector
and Record/Replay were explicitly **not** selected.

**Tech Stack:** Vite + vanilla TypeScript + Three.js, plain CSS custom properties, Vitest for
unit tests, Playwright (`scripts/browser-check.mjs`) for the live UI acceptance test, FastAPI +
WebSocket backend (`python/fly_drone/server.py`).

## Global Constraints

- **No left drawer.** Everything new lives in the existing bottom bar (`.bar-group`,
  `.bar-cluster`) or the pinned top-right stack (`.hud-pinned`), or as world overlays.
- **Mode-aware only.** The free-roam group appears when `frame.task == "free_roam"`. A level
  selector is sanctioned by `docs/free-roam.md` §7; **no** task/ablation/decoder switching
  (AGENTS.md "one brain, one decoder").
- **Never relax `roam_eval.ACCEPTANCE`** and keep the legacy room MJCF reproducible
  (`test_legacy_room_mjcf_unchanged`).
- **browser-check is the UI acceptance test.** Existing ids/roles/text it asserts on must be
  preserved; new assertions are additive.
- Backend/WS changes are allowed. Frontend stays vanilla TS + Three.js (no new dependency).
- Commit and push after each phase; keep workers modest (≤6).

## File Structure

New:
- `web/src/types.ts` — shared `Cell`/`Metadata`/`Frame`/`FreeRoam`/`Outcome`/`Attribution`.
- `web/src/charts/scope.ts` + `scope.test.ts` — dependency-free canvas ring-buffer sparkline.
- `web/src/ui/freeRoam.ts` — free-roam scoreboard / probe cluster markup.
- `web/src/ui/statusStrip.ts` — policy/decoder/seed status truth strip.

Modified:
- `python/fly_drone/server.py` — derived free-roam rates, attribution seq/cadence.
- `tests/test_server.py` — assert the derived fields.
- `web/src/main.ts` — compose the new UI, wire handlers, scope buffers, a11y.
- `web/src/ui/bottomBar.ts` — mount free-roam group + status strip.
- `web/src/style.css` — free-roam/scope/status styles, type scale, responsive sheet.
- `scripts/browser-check.mjs` — additive free-roam/scope/a11y checks.

---

## Phase 0 — Foundations

### Task 0.1: Shared UI types
- [ ] Create `web/src/types.ts` with `Cell`, `Metadata`, `Frame` (including `outcome`,
      `free_roam`, `attribution`), `FreeRoam`, `FreeRoamEvent`, `Outcome`, `Attribution`.
- [ ] Import them in `main.ts`; delete the local copies.
- Verify: `yarn typecheck && yarn lint && yarn test`.

### Task 0.2: Derived free-roam rates (server)
- [ ] Extend the `free_roam` frame payload in `server.py` with `elapsed`,
      `beacons_per_min`, `collisions_per_min`, `coverage = visited_cells / 256`.
- [ ] Add assertions to `tests/test_server.py`.
- Verify: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_server.py -q`.

### Task 0.3: Attribution cadence/seq (server)
- [ ] Add `attribution_seq` to the frame and a `ATTRIBUTION_EVERY` constant (default 12) so the
      UI can refresh the attribution panel without guessing.

### Task 0.4: Scope renderer
- [ ] `web/src/charts/scope.ts`: `Scope` class with a fixed-length ring buffer, multi-series
      autoscaling, optional target line, `push(values)` + `draw(ctx)`. Vitest for buffer/scale.

Browser-check grows **with each feature**: the free-roam assertions land in Phase 1, the scope
and attribution assertions in Phase 2, and the keyboard/a11y assertions in Phase 3. No assertion
is added before the element it checks exists.

---

## Phase 1 — Free-Roam Mission Control

- [ ] `web/src/ui/freeRoam.ts`: scoreboard (beacons/min, collisions/min, dodged/hit, coverage,
      clearance, level), probe cluster (`Beacon here`, `Threat now`, `Silence vision`,
      `Silence loom`, `Ghost`, `Restore`), level selector, one-line event ticker.
- [ ] Wire in `main.ts` to `place_beacon`, `launch_threat`, `pathway`, `ghost`, `restore`,
      `reset {level, seed}`; reflect authoritative `free_roam.silenced` / `ghost`.
- [ ] World overlays: beacon-visible ring, clearance halo, threat trajectory, GHOST watermark.
- [ ] Status strip (`.bar-foot`): `policy_status`, active policy, task, level, seed, RTF,
      missed deadlines.
- Verify: manual at L2/L3 + `yarn browser:check`.

## Phase 2 — Live Scopes + Attribution

- [ ] Replace per-frame `innerHTML` meters in the Signals group with `Scope` canvases (sensory,
      readouts, motors, budget) while keeping numeric readouts.
- [ ] Attribution panel: four channel rows, signed top-type bars, hover tooltip, top cell ids;
      labelled "gradient × input, decoder-only".
- [ ] Performance: ring buffers at ~10 Hz, rAF draw, cached DOM, reused `Vector3`/`Color`,
      no per-frame `innerHTML`.
- Verify: `yarn test`, manual with a loaded decoder.

## Phase 3 — Accessibility + Responsive

- [ ] Type-scale + contrast pass (labels ≥ 11px, body 13px, tabular numerals).
- [ ] Semantics: `role="meter"` + `aria-valuenow`, `aria-live` status/outcome/events,
      `aria-pressed` toggles, labelled icon controls.
- [ ] Keyboard shortcuts + `?` help overlay (Space, R, F, 1/2/3, G).
- [ ] Color-blind-safe mode (persisted) with shape/dash cues, not colour alone.
- [ ] Mobile collapsible bottom sheet; no horizontal scroll at 390×844; reduced-motion coverage.
- Verify: keyboard walkthrough, `yarn browser:check` desktop + mobile, `yarn test:tokens`.

## Self-Review

Phase 0 unblocks all later work and is independently verifiable. Every task keeps the
`browser-check` contract intact (constraint above). No task touches `roam_eval.ACCEPTANCE`,
training runs, or the legacy room geometry.
