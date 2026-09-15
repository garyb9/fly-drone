export function renderHudPinned(): string {
  return `
<section class="hud-pinned bp-boot" id="hud-pinned">
<div class="brain-mini"><div class="brain-mini-head"><span id="mode">INITIALIZING</span><span id="tick">TICK 0</span></div><div id="brain" class="viewport"></div><div class="brain-legend"><span><i></i> measured activity</span><span>connections</span></div></div>
<div class="fly-mini-panel" title="Same neural readouts, independent trajectory — illustrative fly dynamics, not calibrated biomechanics."><div class="fly-mini-head"><span>FLY BODY</span></div><div id="fly" class="viewport"></div></div>
<div class="eyes-panel"><div class="eyes"><figure><img id="eye0" alt="Left simulated eye"><figcaption>LEFT EYE</figcaption></figure><figure><img id="eye1" alt="Right simulated eye"><figcaption>RIGHT EYE</figcaption></figure></div></div>
<div class="instruments">
<div class="attitude-gauges">
<div class="attitude-gauge"><div class="track"><i class="fill" id="roll-fill"></i><i class="needle" id="roll-needle"></i></div><span>ROLL</span><em id="roll-value">—</em></div>
<div class="attitude-gauge"><div class="track"><i class="fill" id="pitch-fill"></i><i class="needle" id="pitch-needle"></i></div><span>PITCH</span><em id="pitch-value">—</em></div>
<div class="attitude-gauge"><div class="track"><i class="fill" id="yaw-fill"></i><i class="needle" id="yaw-needle"></i></div><span>YAW</span><em id="yaw-value">—</em></div>
</div>
<div class="turn-rate"><span>TURN RATE</span><em id="turn-rate">—</em></div>
<div class="vector-legend">
<span><i style="background:#ffb15c"></i><span class="legend-label">heading</span></span>
<span><i style="background:#6fe2ff"></i><span class="legend-label">velocity</span></span>
<span><i style="background:#ff6fd8"></i><span class="legend-label">command</span></span>
<span><i style="background:#ff5c5c"></i><span class="legend-label">world X</span></span>
<span><i style="background:#5cff7a"></i><span class="legend-label">world Z↑</span></span>
<span><i style="background:#5cb0ff"></i><span class="legend-label">world -Y</span></span>
</div>
</div>
</section>`;
}
