/**
 * Dependency-free rolling scope for live telemetry.
 *
 * The buffer/scale logic is plain data so it can be unit-tested without a DOM; only
 * `draw()` touches a canvas. Kept deliberately small: the viewer runs at 20 Hz frame
 * pushes against a rAF draw loop, so allocation per sample is O(1) and per draw is O(n).
 */

export class RingBuffer {
  private data: number[];
  private head = 0;
  private filled = 0;

  constructor(readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error("RingBuffer capacity must be a positive integer");
    }
    this.data = new Array<number>(capacity);
  }

  push(value: number): void {
    this.data[this.head] = value;
    this.head = (this.head + 1) % this.capacity;
    this.filled = Math.min(this.capacity, this.filled + 1);
  }

  get length(): number {
    return this.filled;
  }

  get last(): number | undefined {
    if (this.filled === 0) return undefined;
    return this.data[(this.head - 1 + this.capacity) % this.capacity];
  }

  /** Samples ordered oldest -> newest. */
  toArray(): number[] {
    const out = new Array<number>(this.filled);
    const start = (this.head - this.filled + this.capacity) % this.capacity;
    for (let i = 0; i < this.filled; i++) {
      out[i] = this.data[(start + i) % this.capacity];
    }
    return out;
  }
}

/**
 * Vertical bounds for a signal that always keep the target line and a flat series
 * visible (a flat signal would otherwise collapse to a zero-height range).
 */
export function seriesBounds(values: number[], targetLine = 0): { min: number; max: number } {
  let min = targetLine;
  let max = targetLine;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }
  if (min === max) {
    const pad = Math.max(Math.abs(min) * 0.1, 0.001);
    return { min: min - pad, max: max + pad };
  }
  return { min, max };
}

export type ScopeStyle = {
  colors: string[];
  /** Dashed horizontal reference line (e.g. the 5 ms tick budget). */
  target?: number;
  /** Fixed bounds; when omitted the scope autoscales to the visible history. */
  min?: number;
  max?: number;
};

export class Scope {
  readonly buffers: RingBuffer[];

  constructor(
    readonly series: number,
    capacity: number,
    private readonly style: ScopeStyle,
  ) {
    if (series <= 0) throw new Error("Scope needs at least one series");
    if (style.colors.length !== series) {
      throw new Error("Scope needs one colour per series");
    }
    this.buffers = Array.from({ length: series }, () => new RingBuffer(capacity));
  }

  push(values: number[]): void {
    if (values.length !== this.series) {
      throw new Error(`Scope expects ${this.series} values, got ${values.length}`);
    }
    this.buffers.forEach((buffer, i) => buffer.push(values[i]));
  }

  bounds(): { min: number; max: number } {
    if (this.style.min !== undefined && this.style.max !== undefined) {
      return { min: this.style.min, max: this.style.max };
    }
    const all = this.buffers.flatMap((buffer) => buffer.toArray());
    return seriesBounds(all, this.style.target ?? 0);
  }

  draw(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const { min, max } = this.bounds();
    const span = max - min || 1;
    const y = (value: number) => height - ((value - min) / span) * height;
    const capacity = this.buffers[0].capacity;
    const x = (i: number) => (i / Math.max(1, capacity - 1)) * width;

    ctx.clearRect(0, 0, width, height);

    if (this.style.target !== undefined) {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 177, 92, 0.55)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, y(this.style.target));
      ctx.lineTo(width, y(this.style.target));
      ctx.stroke();
      ctx.restore();
    }

    this.buffers.forEach((buffer, s) => {
      const values = buffer.toArray();
      if (values.length < 2) return;
      const offset = capacity - values.length;
      ctx.strokeStyle = this.style.colors[s];
      ctx.lineWidth = 1.25;
      ctx.lineJoin = "round";
      ctx.beginPath();
      values.forEach((value, i) => {
        if (!Number.isFinite(value)) return;
        const px = x(offset + i);
        const py = y(value);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });
  }
}
