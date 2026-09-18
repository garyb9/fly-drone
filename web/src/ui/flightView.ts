export function renderFlightView(): string {
  return `
<div class="world">
<div id="world" class="viewport"></div>
<div class="connection"><i id="dot"></i><span id="status">Connecting to simulation</span></div>
<div class="world-frame bp-boot"><i></i><i></i><i></i><i></i></div>
<div class="world-dims" id="dims"></div>
<div class="ghost-banner" id="ghost-banner" hidden role="status">GHOST · OBJECTS INVISIBLE TO THE EYES · STILL COLLIDABLE</div>
</div>`;
}
