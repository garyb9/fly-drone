import type { FreeRoamEvent } from "../types";

// Frame-rate formatting that is pure and therefore unit-testable without a DOM.

// Exaggerates one second of travel so low speeds (a few tenths of a m/s) are still visible as
// short arrows rather than points, while clamping so a fast dodge doesn't dwarf the arena.
export const VECTOR_SCALE = 1.2;
export const MIN_ARROW_LEN = 0.12;
export const MAX_ARROW_LEN = 1.0;

export function scaledArrowLength(magnitude: number): number {
  return Math.min(MAX_ARROW_LEN, Math.max(MIN_ARROW_LEN, magnitude * VECTOR_SCALE));
}

// Roll/pitch/yaw in degrees from the MuJoCo world-frame quaternion (w, x, y, z), standard
// aerospace convention. Shared by the debug command-arrow rotation and the attitude gauges.
export function attitudeFromQuaternion(q: number[]): {
  roll: number;
  pitch: number;
  yaw: number;
} {
  const [w, x, y, z] = q;
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const deg = (r: number) => (r * 180) / Math.PI;
  return { roll: deg(roll), pitch: deg(pitch), yaw: deg(yaw) };
}

export function formatRoamEvent(event: FreeRoamEvent): string {
  const when = event.time !== undefined ? `t${event.time.toFixed(1)}s ` : "";
  switch (event.type) {
    case "beacon_collected":
      return `${when}beacon #${event.count} collected`;
    case "threat_launched":
      return `${when}threat launched (${(event.side ?? 0) > 0 ? "left" : "right"})`;
    case "threat_passed":
      return `${when}threat dodged · min ${event.min_distance?.toFixed(2) ?? "—"} m`;
    case "threat_hit":
      return `${when}threat hit`;
    case "collision":
      return `${when}collision: ${event.kinds?.join(", ") ?? ""}`;
    default:
      return `${when}${event.type.replace(/_/g, " ")}`;
  }
}
