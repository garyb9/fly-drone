export function renderFlightView(): string {
  return `
<div class="world">
<div id="world" class="viewport"></div>
<div class="connection"><i id="dot"></i><span id="status">Connecting to simulation</span></div>
<div class="world-bottom"><div><span>ALTITUDE</span><strong id="altitude">—<small> m</small></strong></div><div><span>SPEED</span><strong id="speed">—<small> m/s</small></strong></div><div><span>VX / VY / VZ</span><strong id="velocity-axes">—<small> m/s</small></strong></div><div><span>SIMULATION</span><strong id="simtime">0.00<small> s</small></strong></div><div><span>REAL TIME</span><strong id="rtf">—<small> ×</small></strong></div><div class="camera-controls"><button id="cam-follow" title="Keep the camera target locked to the drone">Follow</button><button id="cam-recenter" title="Snap the camera target to the drone once">Recenter</button><button id="cam-reset" title="Restore the default orbit view">Reset view</button><button id="cam-fpv" title="Ride along in the drone's cockpit">1st person</button><button id="cam-tpv" title="Chase camera behind the drone">3rd person</button></div></div>
<div class="world-frame bp-boot"><i></i><i></i><i></i><i></i></div>
<div class="world-dims" id="dims"></div>
<div class="world-hint">DRAG TO ORBIT · SCROLL TO ZOOM</div>
</div>`;
}
