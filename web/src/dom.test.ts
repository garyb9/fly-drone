import { describe, expect, it } from "vitest";
import { DOM_IDS, initDom } from "./dom";

// A dependency-free stand-in for `document`: `initDom` only needs `querySelectorAll`,
// so the boot-time validation can be unit-tested without a DOM environment.
function stubRoot(counts: Record<string, number> = {}, tagOverrides: Record<string, string> = {}) {
  return {
    querySelectorAll(selector: string) {
      const id = selector.slice(5, -2);
      const tag = (tagOverrides[id] ?? DOM_IDS[id as keyof typeof DOM_IDS] ?? "DIV").toUpperCase();
      return Array.from({ length: counts[id] ?? 1 }, (_, i) => ({
        tagName: i === 0 ? tag : "DIV",
      }));
    },
  } as unknown as ParentNode;
}

describe("initDom", () => {
  it("resolves every declared id and types canvases/images concretely", () => {
    const els = initDom(stubRoot());
    expect(Object.keys(els)).toHaveLength(Object.keys(DOM_IDS).length);
    expect(els["cues-scope"]).toBeInstanceOf(Object);
    expect(els.pause.tagName).toBe("BUTTON");
  });

  it("throws when a declared id is missing", () => {
    expect(() => initDom(stubRoot({ world: 0 }))).toThrow(/#world is missing/);
  });

  it("throws when an id is duplicated", () => {
    expect(() => initDom(stubRoot({ world: 2 }))).toThrow(/#world appears 2 times/);
  });

  it("throws when an id resolves to the wrong tag", () => {
    expect(() => initDom(stubRoot({}, { world: "SPAN" }))).toThrow(
      /#world is <span>, expected <div>/,
    );
  });
});
