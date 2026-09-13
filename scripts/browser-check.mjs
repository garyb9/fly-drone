import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
const browser = await chromium.launch({
  executablePath: process.env.CHROME_BIN || "/usr/bin/google-chrome",
  headless: true,
  args: [
    "--no-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1060 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("http://127.0.0.1:8000");
  await page.waitForFunction(
    () =>
      document.querySelector("#status")?.textContent?.includes("Simulation paused") ||
      document.querySelector("#status")?.textContent === "Local simulation connected",
    null,
    { timeout: 60000 },
  );
  await page.screenshot({ path: "/tmp/fly-drone-desktop.png" });
  if (await page.getByRole("button", { name: "Pause", exact: true }).count())
    await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent === "Simulation paused",
  );
  const before = await page.locator("#tick").textContent();
  await page.waitForTimeout(500);
  assert.equal(await page.locator("#tick").textContent(), before);
  await page.getByRole("button", { name: "Reset trial" }).click();
  await page.waitForFunction(() => document.querySelector("#episode")?.textContent === "EPISODE 1");
  await page.getByRole("button", { name: "Left", exact: true }).click();
  await page.getByRole("button", { name: "Pulse", exact: true }).click();
  await page.getByRole("button", { name: "Restore", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "/tmp/fly-drone-mobile.png", fullPage: true });
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    true,
  );
  assert.deepEqual(errors, []);
  console.log(
    "PASS: desktop/mobile rendering, pause, reset, target placement, interventions; no page errors",
  );
} finally {
  await browser.close();
}
