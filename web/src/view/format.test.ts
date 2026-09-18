import { describe, expect, it } from "vitest";
import { attitudeFromQuaternion, formatRoamEvent, scaledArrowLength } from "./format";

const identity = [1, 0, 0, 0];

describe("attitudeFromQuaternion", () => {
  it("reads identity as level and aligned", () => {
    expect(attitudeFromQuaternion(identity)).toEqual({ roll: 0, pitch: 0, yaw: 0 });
  });

  it("reads a +90 degree yaw about world Z", () => {
    const half = Math.SQRT1_2; // cos(45deg) = sin(45deg)
    const { roll, pitch, yaw } = attitudeFromQuaternion([half, 0, 0, half]);
    expect(roll).toBeCloseTo(0, 6);
    expect(pitch).toBeCloseTo(0, 6);
    expect(yaw).toBeCloseTo(90, 6);
  });
});

describe("scaledArrowLength", () => {
  it("floors slow speeds and caps fast ones", () => {
    expect(scaledArrowLength(0)).toBeCloseTo(0.12, 6);
    expect(scaledArrowLength(10)).toBeCloseTo(1.0, 6);
  });

  it("scales the mid range", () => {
    expect(scaledArrowLength(0.2)).toBeCloseTo(0.24, 6);
  });
});

describe("formatRoamEvent", () => {
  it("names collected beacons with their count and time", () => {
    expect(formatRoamEvent({ type: "beacon_collected", time: 1.24, count: 3 })).toBe(
      "t1.2s beacon #3 collected",
    );
  });

  it("keeps the raw event type when it has no special case", () => {
    expect(formatRoamEvent({ type: "some_new_event" })).toBe("some new event");
  });
});
