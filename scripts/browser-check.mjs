import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.env.FLY_DRONE_URL || "http://127.0.0.1:8000";
const chrome = process.env.CHROME_BIN || "/usr/bin/google-chrome";
const browser = await chromium.launch({
  // Fall back to Playwright's bundled Chromium when system Chrome is absent.
  executablePath: existsSync(chrome) ? chrome : undefined,
  headless: true,
  args: [
    "--no-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});

const text = (page, selector) => page.locator(selector).textContent();
const tickOf = async (page) => Number((await text(page, "#tick")).replace(/[^0-9]/g, ""));
const connected = (page) =>
  page.waitForFunction(
    () => {
      const status = document.querySelector("#status")?.textContent ?? "";
      return status === "Local simulation connected" || status === "Simulation paused";
    },
    null,
    { timeout: 60000 },
  );
const passed = [];

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1060 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto(url);
  await connected(page);
  await page.waitForFunction(() => document.querySelectorAll("#motors .meter").length === 4);
  await page.screenshot({ path: join(tmpdir(), "fly-drone-desktop.png") });
  passed.push("connect + render");

  const motors = await page.locator("#motors .meter em").allTextContents();
  assert.equal(motors.length, 4);
  for (const m of motors) {
    const [actual, commanded] = m.split("/").map((v) => Number(v.trim()));
    assert.ok(actual > 10000 && actual < 22000, `actual RPM ${m}`);
    assert.ok(commanded > 10000 && commanded < 22000, `commanded RPM ${m}`);
  }
  passed.push("motor telemetry (actual/commanded RPM near hover)");

  // The bottom control bar is the only controls surface now (the left drawer is gone).
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent === "Simulation paused",
  );
  const frozen = await text(page, "#tick");
  await page.waitForTimeout(600);
  assert.equal(await text(page, "#tick"), frozen);
  passed.push("pause freezes tick");

  await page.getByRole("button", { name: "Resume", exact: true }).click();
  const resumedFrom = await tickOf(page);
  await page.waitForFunction(
    (t) => Number(document.querySelector("#tick").textContent.replace(/[^0-9]/g, "")) > t,
    resumedFrom,
  );
  passed.push("resume advances tick");

  const episodeBefore = await text(page, "#episode");
  await page.getByRole("button", { name: "Reset trial" }).click();
  await page.waitForFunction(
    (before) => document.querySelector("#episode")?.textContent !== before,
    episodeBefore,
  );
  const afterReset = await tickOf(page);
  assert.ok(afterReset < resumedFrom + 400, `tick restarted after reset (${afterReset})`);
  passed.push("reset increments episode and restarts ticks");

  await page.getByRole("button", { name: "Left", exact: true }).click();
  await page.getByRole("button", { name: "Place ahead" }).click();
  await page.getByRole("button", { name: "Move aside" }).click();
  passed.push("target and obstacle placement");

  await page.getByRole("button", { name: "B · 450 mm" }).click();
  await page.getByRole("button", { name: "Eye FOV" }).click();
  await page.getByRole("button", { name: "Guards" }).click();
  await page.getByRole("button", { name: "3rd person" }).click();
  await page.getByRole("button", { name: "1st person" }).click();
  await page.getByRole("button", { name: "Reset view" }).click();
  passed.push("airframe and camera controls");

  // The status strip is the one place that explains a decoder holding the drone still.
  const statusStrip = await text(page, "#status-strip");
  assert.ok(statusStrip && statusStrip.length > 0, "status strip populated");
  passed.push("decoder status strip");

  // Free-roam mission control is mode-aware: present only when a free-roam decoder is loaded.
  if (await page.locator("#roam-group").isVisible()) {
    await page.waitForFunction(() => {
      const value = document.querySelector("#roam-beacons")?.textContent ?? "";
      return value !== "" && value !== "—";
    });
    assert.match(await text(page, "#roam-beacons"), /^\d+(\.\d+)?$/);
    assert.match(await text(page, "#roam-collisions"), /^\d+(\.\d+)?$/);
    assert.ok((await page.locator("#roam-levels button").count()) >= 2);

    const pressed = (id) =>
      page.waitForFunction(
        (selector) => document.querySelector(selector)?.getAttribute("aria-pressed") === "true",
        id,
      );
    const released = (id) =>
      page.waitForFunction(
        (selector) => document.querySelector(selector)?.getAttribute("aria-pressed") === "false",
        id,
      );

    await page.getByRole("button", { name: "Silence loom", exact: true }).click();
    await pressed("#roam-silence-loom");
    await page.getByRole("button", { name: "Silence vision", exact: true }).click();
    await pressed("#roam-silence-sensory");
    await page.getByRole("button", { name: "Ghost", exact: true }).click();
    await pressed("#roam-ghost");
    await page.waitForFunction(
      () => getComputedStyle(document.querySelector("#ghost-banner")).display !== "none",
    );
    await page.getByRole("button", { name: "Restore", exact: true }).click();
    await released("#roam-silence-loom");
    await released("#roam-silence-sensory");
    await released("#roam-ghost");
    await page.waitForFunction(
      () => getComputedStyle(document.querySelector("#ghost-banner")).display === "none",
    );
    await page.getByRole("button", { name: "Threat now", exact: true }).click();
    passed.push("free-roam mission control (scoreboard, causal probes, ghost)");
  } else {
    passed.push("free-roam group hidden (no free-roam decoder loaded)");
  }

  // Rolling scopes draw into backing canvases; attribution explains the decoder's commands.
  const cuesScope = page.locator("#cues-scope");
  assert.ok(await cuesScope.isVisible(), "sensory scope visible");
  assert.ok(
    await cuesScope.evaluate((canvas) => canvas.width > 0 && canvas.height > 0),
    "sensory scope has backing pixels",
  );
  passed.push("rolling signal scopes");
  if (await page.locator("#attribution-group").isVisible()) {
    await page.waitForFunction(
      () => document.querySelectorAll("#attribution .attr-channel").length >= 4,
    );
    const attribution = await text(page, "#attribution");
    assert.match(attribution, /forward/);
    assert.match(attribution, /lateral/);
    passed.push("attribution panel (per-channel decoder contributions)");
  } else {
    passed.push("attribution hidden (no decoder loaded)");
  }

  const episode = await text(page, "#episode");
  const beforeReload = await tickOf(page);
  await page.reload();
  await connected(page);
  await page.waitForFunction(
    (t) => Number(document.querySelector("#tick").textContent.replace(/[^0-9]/g, "")) > t,
    beforeReload,
  );
  assert.equal(await text(page, "#episode"), episode);
  passed.push("reconnect keeps the running simulation (same episode, tick continues)");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: join(tmpdir(), "fly-drone-mobile.png"), fullPage: true });
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    true,
  );
  passed.push("mobile layout without horizontal scroll");

  assert.deepEqual(errors, []);
  console.log(`PASS (${passed.length}): ${passed.join("; ")}; no page errors`);
} finally {
  await browser.close();
}
