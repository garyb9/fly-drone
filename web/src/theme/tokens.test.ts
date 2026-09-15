import { describe, expect, it } from "vitest";
import { hexToInt, PALETTE } from "./tokens";
import { applyCssTokens } from "./apply-css-tokens";

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

describe("applyCssTokens", () => {
  it("sets every --bp-* custom property to its palette value", () => {
    // Mock HTMLElement with a style object that tracks setProperty calls
    const properties: Record<string, string> = {};
    const mockEl = {
      style: {
        setProperty: (prop: string, value: string) => {
          properties[prop] = value;
        },
        getPropertyValue: (prop: string) => properties[prop] || "",
      },
    } as unknown as HTMLElement;

    applyCssTokens(mockEl);
    expect(properties["--bp-bg"]).toBe(PALETTE.navyDeep);
    expect(properties["--bp-amber"]).toBe(PALETTE.amber);
  });
});
