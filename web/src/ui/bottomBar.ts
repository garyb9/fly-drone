import { renderAttribution } from "./attribution";
import { renderFreeRoam } from "./freeRoam";
import { renderHelp } from "./help";

export function renderBottomBar(): string {
  return `
<div class="bottom-bar bp-boot" id="bottom-bar">
<section class="bar-group bar-group-signals">
<div class="bar-head"><h3>Signals</h3><span class="bar-head-meta">full graph</span></div>
<div class="bar-grid">
<div><span class="bar-label">sensory current</span><canvas class="scope" id="cues-scope" aria-hidden="true"></canvas><div id="cues" class="meters"></div></div>
<div><span class="bar-label">neural readout</span><canvas class="scope" id="readouts-scope" aria-hidden="true"></canvas><div id="readouts" class="meters"></div></div>
<div><span class="bar-label">actual / commanded rpm</span><canvas class="scope" id="motors-scope" aria-hidden="true"></canvas><div id="motors" class="meters"></div></div>
<div><span class="bar-label">compute budget</span><canvas class="scope" id="budget-scope" aria-hidden="true"></canvas><div id="budget" class="meters"></div><em class="bar-note" id="budget-note">—</em></div>
</div>
</section>
${renderFreeRoam()}
${renderAttribution()}
<section class="bar-group bar-group-console">
<div class="bar-head"><h3>Controls &amp; camera</h3><span class="bar-head-meta">episode <b id="episode">0</b></span></div>
<div class="bar-row bar-row-controls">
<div class="bar-cell"><span class="bar-label">experiment</span><div class="button-row"><button id="pause" class="primary">Pause</button><button id="reset">Reset trial</button></div></div>
<div class="bar-cell"><span class="bar-label">visual target</span><div class="button-row"><button data-target="left">Left</button><button data-target="center">Center</button><button data-target="right">Right</button></div></div>
<div class="bar-cell"><span class="bar-label">obstacle</span><div class="button-row"><button id="loom">Place ahead</button><button id="clear">Move aside</button></div></div>
</div>
<div class="bar-row bar-row-camera">
<span class="bar-label">camera &amp; model</span>
<div class="bar-clusters">
<div class="bar-cluster"><em>airframe</em><div class="button-row"><button id="af-a" class="active" title="250 mm airframe — Option A, connectome runs on the ground">A · 250 mm</button><button id="af-b" title="450 mm airframe — Option B, onboard companion computer">B · 450 mm</button></div></div>
<div class="bar-cluster"><em>eyes</em><div class="button-row"><button id="toggle-fov" class="active" title="Show the two eye fields of view (simulated camera splay and fov)">Eye FOV</button><button id="toggle-guards" class="active" title="Show propeller guards">Guards</button><button id="eye-fx" title="Display-only low-light/blur preview of the eye feed — does not alter the simulated camera input">Eye FX</button></div></div>
<div class="bar-cluster"><em>view</em><div class="button-row"><button id="cam-orbit" title="Slow automatic orbit for demos">Auto-orbit</button><button id="cam-follow" class="active" title="Keep the camera target locked to the drone">Follow</button><button id="cam-recenter" title="Snap the camera target to the drone once">Recenter</button><button id="cam-reset" title="Restore the default orbit view">Reset view</button><button id="cb-safe" aria-pressed="false" title="Switch the heading / velocity / command arrows to a colour-blind-safe palette">CVD-safe</button><button id="help-open" title="Keyboard shortcuts (?)">?</button></div></div>
<div class="bar-cluster"><em>mode</em><div class="button-row"><button id="cam-fpv" title="Ride along in the drone's cockpit">1st person</button><button id="cam-tpv" title="Chase camera behind the drone">3rd person</button></div></div>
</div>
</div>
<div class="bar-foot"><span id="command">Motion command: —</span><span class="status-strip" id="status-strip" data-state="warn" role="status" aria-live="polite">DECODER —</span><span id="error">Waiting for the local Rust + MuJoCo service.</span></div>
</section>
</div>
${renderHelp()}`;
}
