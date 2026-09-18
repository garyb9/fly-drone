import "./style.css";
import { initDom } from "./dom";
import { applyCssTokens } from "./theme/apply-css-tokens";
import { renderFlightView } from "./ui/flightView";
import { renderHudPinned } from "./ui/hudPinned";
import { renderBottomBar } from "./ui/bottomBar";
import { renderFooter } from "./ui/footer";
import { createStage } from "./scene/stage";
import { createViewer } from "./app/context";
import { initControls } from "./app/controls";
import { startRenderer } from "./app/loop";
import { connect, send } from "./net/socket";
import { setupBrain, updateFrame, updateRoom } from "./view/frame";

applyCssTokens();

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
<main>
${renderFlightView()}
<div class="hud-layer">
${renderHudPinned()}
${renderBottomBar()}
</div>
</main>
${renderFooter()}`;
// Resolve and validate every id up front so a markup/registry mismatch fails at boot.
initDom();

// Derived from the DOM (every .bp-boot element, in document order) rather than a hardcoded
// selector list, so adding bp-boot to a new element can't silently leave it out of the sequence
// (and stuck invisible). Drop both classes once the animation finishes so its transform stops
// shadowing the element's real state.
function runBootSequence(): void {
  const targets = Array.from(document.querySelectorAll<HTMLElement>(".bp-boot"));
  targets.forEach((target, i) => {
    setTimeout(() => {
      target.classList.add("bp-boot-run");
      target.addEventListener(
        "animationend",
        () => target.classList.remove("bp-boot", "bp-boot-run"),
        { once: true },
      );
    }, i * 120);
  });
}
runBootSequence();

const stage = createStage();
const viewer = createViewer(stage, send);
initControls(viewer);
connect({
  onMetadata: (message) => setupBrain(viewer, message),
  onFrame: (message) => updateFrame(viewer, message),
  onRoom: (message) => updateRoom(viewer, message),
});
startRenderer(viewer);
