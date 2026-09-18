import { describe, expect, it } from "vitest";
import { RingBuffer, Scope, seriesBounds } from "./scope";

describe("RingBuffer", () => {
  it("returns samples oldest to newest", () => {
    const ring = new RingBuffer(3);
    ring.push(1);
    ring.push(2);
    ring.push(3);
    expect(ring.toArray()).toEqual([1, 2, 3]);
    expect(ring.length).toBe(3);
    expect(ring.last).toBe(3);
  });

  it("overwrites the oldest sample once full", () => {
    const ring = new RingBuffer(3);
    for (const value of [1, 2, 3, 4, 5]) ring.push(value);
    expect(ring.toArray()).toEqual([3, 4, 5]);
    expect(ring.length).toBe(3);
  });

  it("reports an empty last before the first push", () => {
    expect(new RingBuffer(2).last).toBeUndefined();
  });

  it("rejects a non-positive capacity", () => {
    expect(() => new RingBuffer(0)).toThrow();
  });
});

describe("seriesBounds", () => {
  it("includes the target line and data", () => {
    expect(seriesBounds([1, 2, 3], 0)).toEqual({ min: 0, max: 3 });
  });

  it("pads a flat signal so it stays visible", () => {
    const { min, max } = seriesBounds([5, 5, 5], 5);
    expect(min).toBeLessThan(5);
    expect(max).toBeGreaterThan(5);
  });
});

describe("Scope", () => {
  it("tracks one buffer per series and exposes the last sample", () => {
    const scope = new Scope(2, 4, { colors: ["#f00", "#0f0"] });
    scope.push([1, 2]);
    scope.push([3, 4]);
    expect(scope.buffers[0].last).toBe(3);
    expect(scope.buffers[1].last).toBe(4);
  });

  it("uses fixed bounds when supplied", () => {
    const scope = new Scope(1, 4, { colors: ["#f00"], min: 0, max: 10 });
    expect(scope.bounds()).toEqual({ min: 0, max: 10 });
  });

  it("rejects a value count that does not match its series", () => {
    const scope = new Scope(2, 4, { colors: ["#f00", "#0f0"] });
    expect(() => scope.push([1])).toThrow();
  });
});
