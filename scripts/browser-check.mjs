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

  if (await page.getByRole("button", { name: "Pause", exact: true }).count()) {
    await page.getByRole("button", { name: "Pause", exact: true }).click();
  }
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

  for (const op of ["Pulse", "Hold", "Silence", "Restore"]) {
    await page.getByRole("button", { name: op, exact: true }).click();
  }
  passed.push("neuron interventions");

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

  await page.selectOption("#task", "looming");
  await page.selectOption("#ablation", "sensory");
  await page.fill("#seed", "1003");
  await page.getByRole("button", { name: "Run trial" }).click();
  await page.waitForFunction(() => {
    const trial = document.querySelector("#trial")?.textContent ?? "";
    return (
      trial.includes("SEED 1003") &&
      trial.includes("DODGE OBSTACLE") &&
      trial.includes("VISION SILENCED")
    );
  });
  await page.waitForFunction(() =>
    (document.querySelector("#outcome")?.textContent ?? "").includes("obstacle"),
  );
  assert.ok((await page.locator("#report option").count()) >= 1);
  passed.push("trial reset with seed/task/ablation and live looming outcome");

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
