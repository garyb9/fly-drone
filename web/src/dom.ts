/**
 * Typed registry of every DOM id the viewer touches.
 *
 * The markup lives in static template strings (`web/src/ui/*`), so a renamed or duplicated
 * id would otherwise only surface as a null dereference in the middle of a live frame.
 * Declaring the contract here lets `initDom()` resolve each node once after mount, fail
 * loudly at boot on a missing / duplicated / mis-typed id, and hand callers concrete
 * element types instead of `HTMLElement | null`.
 */

export const DOM_IDS = {
  // Flight view
  world: "div",
  dot: "i",
  status: "span",
  dims: "div",
  "ghost-banner": "div",
  // Pinned HUD
  altitude: "strong",
  speed: "strong",
  "velocity-axes": "strong",
  simtime: "strong",
  rtf: "strong",
  mode: "span",
  tick: "span",
  brain: "div",
  fly: "div",
  eye0: "img",
  eye1: "img",
  "roll-fill": "i",
  "roll-needle": "i",
  "roll-value": "em",
  "pitch-fill": "i",
  "pitch-needle": "i",
  "pitch-value": "em",
  "yaw-fill": "i",
  "yaw-needle": "i",
  "yaw-value": "em",
  "turn-rate": "em",
  "legend-heading": "i",
  "legend-velocity": "i",
  "legend-command": "i",
  // Bottom bar: tabs, panels, controls, telemetry
  "tab-controls": "button",
  "tab-signals": "button",
  "tab-attribution": "button",
  episode: "b",
  "panel-controls": "div",
  "panel-signals": "div",
  "panel-attribution": "div",
  pause: "button",
  reset: "button",
  loom: "button",
  clear: "button",
  "af-a": "button",
  "af-b": "button",
  "toggle-fov": "button",
  "toggle-guards": "button",
  "eye-fx": "button",
  "cam-orbit": "button",
  "cam-follow": "button",
  "cam-recenter": "button",
  "cam-reset": "button",
  "cb-safe": "button",
  "help-open": "button",
  "cam-fpv": "button",
  "cam-tpv": "button",
  "cues-scope": "canvas",
  "readouts-scope": "canvas",
  "motors-scope": "canvas",
  "budget-scope": "canvas",
  cues: "div",
  readouts: "div",
  motors: "div",
  budget: "div",
  "budget-note": "em",
  command: "span",
  "status-strip": "span",
  error: "span",
  // Free roam
  "roam-buttons": "div",
  "roam-readout": "aside",
  "roam-beacons": "strong",
  "roam-collisions": "strong",
  "roam-threats": "strong",
  "roam-coverage": "strong",
  "roam-clearance": "strong",
  "roam-level": "strong",
  "roam-beacon": "button",
  "roam-threat": "button",
  "roam-silence-sensory": "button",
  "roam-silence-light": "button",
  "roam-silence-loom": "button",
  "roam-ghost": "button",
  "roam-restore": "button",
  "roam-levels": "div",
  "roam-event": "div",
  "roam-visibility": "span",
  // Attribution + help
  attribution: "div",
  "help-overlay": "div",
  "help-close": "button",
} as const;

export type Dom = {
  readonly [K in keyof typeof DOM_IDS]: HTMLElementTagNameMap[(typeof DOM_IDS)[K]];
};

export function initDom(root: ParentNode = document): Dom {
  const els: Record<string, HTMLElement> = {};
  for (const id of Object.keys(DOM_IDS) as (keyof typeof DOM_IDS)[]) {
    const matches = root.querySelectorAll(`[id="${id}"]`);
    if (matches.length === 0) throw new Error(`DOM id #${id} is missing from the markup`);
    if (matches.length > 1) throw new Error(`DOM id #${id} appears ${matches.length} times`);
    const node = matches[0];
    const expected = DOM_IDS[id];
    const actual = node.tagName.toLowerCase();
    if (actual !== expected) {
      throw new Error(`DOM id #${id} is <${actual}>, expected <${expected}>`);
    }
    els[id] = node as HTMLElement;
  }
  if (typeof document !== "undefined" && root === document) cache = els as Dom;
  return els as Dom;
}

// Lazy module singleton: modules can `import { els }` and read nodes without a boot-order
// dependency. The boot file calls `initDom()` once to fail fast, which also seeds the cache.
let cache: Dom | undefined;
export const els: Dom = new Proxy({} as Dom, {
  get(_target, key: string | symbol) {
    cache ??= initDom();
    return cache[key as keyof Dom];
  },
});

/** Write `value` only when it changed, skipping the layout/paint of a redundant update. */
export function setText(host: HTMLElement, value: string): void {
  if (host.textContent !== value) host.textContent = value;
}

/** Update an element's leading text node while leaving its child elements (e.g. `<small>`) intact. */
export function setLeadingText(host: HTMLElement, value: string): void {
  const first = host.firstChild;
  if (first && first.nodeType === Node.TEXT_NODE) {
    if (first.nodeValue !== value) first.nodeValue = value;
  } else {
    host.insertBefore(document.createTextNode(value), first);
  }
}
