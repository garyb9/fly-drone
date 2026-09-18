import { Scope } from "../charts/scope";
import { PALETTE } from "../theme/tokens";

// Meter rows are rebuilt only when the row count or scale changes; otherwise the cached
// nodes have their text/width patched in place, so 20 Hz telemetry stops thrashing innerHTML.
type MeterRow = {
  root: HTMLElement;
  name: HTMLSpanElement;
  bar: HTMLElement;
  value: HTMLElement;
};

const meterCache = new Map<HTMLElement, { max: number; rows: MeterRow[] }>();

export function meters(host: HTMLElement, values: [string, number, string][], max = 1): void {
  let cache = meterCache.get(host);
  if (!cache || cache.rows.length !== values.length || cache.max !== max) {
    host.innerHTML = values
      .map(() => `<div class="meter"><span></span><div><i></i></div><em></em></div>`)
      .join("");
    cache = {
      max,
      rows: Array.from(host.querySelectorAll<HTMLElement>(".meter")).map((root) => {
        root.setAttribute("role", "meter");
        root.setAttribute("aria-valuemin", "0");
        root.setAttribute("aria-valuemax", String(max));
        return {
          root,
          name: root.querySelector("span")!,
          bar: root.querySelector("i")!,
          value: root.querySelector("em")!,
        };
      }),
    };
    meterCache.set(host, cache);
  }
  values.forEach(([name, value, label], i) => {
    const row = cache!.rows[i];
    row.name.textContent = name;
    row.bar.style.width = `${Math.min(100, Math.max(0, (value / max) * 100))}%`;
    row.value.textContent = label;
    row.root.setAttribute("aria-valuenow", value.toFixed(3));
    row.root.setAttribute("aria-label", `${name} ${label}`);
  });
}

// Live history for the same signals the meters show: a series count change (e.g. encoder v4's
// 4 cues vs v5/v6's population groups) rebuilds the scope.
const scopeCache = new Map<HTMLCanvasElement, { canvas: HTMLCanvasElement; scope: Scope }>();
const SCOPE_COLORS = [
  PALETTE.successTeal,
  PALETTE.velocity,
  PALETTE.amber,
  PALETTE.command,
  PALETTE.axisX,
  PALETTE.axisY,
  PALETTE.axisZ,
];

export function pushScope(
  canvas: HTMLCanvasElement,
  values: number[],
  options: { min?: number; max?: number; target?: number } = {},
): void {
  let entry = scopeCache.get(canvas);
  if (!entry || entry.scope.series !== values.length) {
    entry = {
      canvas,
      scope: new Scope(values.length, 300, {
        colors: values.map((_, i) => SCOPE_COLORS[i % SCOPE_COLORS.length]),
        ...options,
      }),
    };
    scopeCache.set(canvas, entry);
  }
  entry.scope.push(values);
}

export function drawScopes(): void {
  for (const { canvas, scope } of scopeCache.values()) {
    const dpr = Math.min(devicePixelRatio, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (ctx) scope.draw(ctx, width, height);
  }
}
