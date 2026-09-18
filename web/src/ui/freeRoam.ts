export function renderFreeRoamControls(): string {
  return `
<div class="bar-roam" id="roam-group" hidden>
<div class="bar-roam-head">
<span class="bar-label">Free roam</span>
<span id="roam-event" role="status" aria-live="polite">No free-roam events yet.</span>
<span id="roam-visibility">Beacon —</span>
</div>
<div class="roam-stats">
<div><span>BEACONS / MIN</span><strong id="roam-beacons">—</strong></div>
<div><span>COLLISIONS / MIN</span><strong id="roam-collisions">—</strong></div>
<div><span>DODGED / HIT</span><strong id="roam-threats">—</strong></div>
<div><span>COVERAGE</span><strong id="roam-coverage">—</strong></div>
<div><span>CLEARANCE</span><strong id="roam-clearance">—</strong></div>
<div><span>LEVEL</span><strong id="roam-level">—</strong></div>
</div>
<div class="bar-row bar-row-roam">
<div class="bar-cell"><span class="bar-label">probe</span><div class="button-row"><button id="roam-beacon" title="Place a beacon on free floor ahead; the drone has to find it.">Beacon here</button><button id="roam-threat" title="Launch a threat on an intercept course; dodging proves the looming pathway.">Threat now</button></div></div>
<div class="bar-cell"><span class="bar-label">causal probes</span><div class="button-row"><button id="roam-silence-sensory" aria-pressed="false" title="Silence all visual input live. Beacons should stop being found and collisions should rise.">Silence vision</button><button id="roam-silence-light" aria-pressed="false" title="Silence only the beacon-light pathway; search should degrade.">Silence light</button><button id="roam-silence-loom" aria-pressed="false" title="Silence only the looming pathway; dodging should fail.">Silence loom</button><button id="roam-ghost" aria-pressed="false" title="Objects become invisible to the eyes but stay collidable: a blind control that must not pass.">Ghost</button><button id="roam-restore" title="Restore every silenced pathway and clear the probes.">Restore</button></div></div>
<div class="bar-cell"><span class="bar-label">level</span><div class="button-row" id="roam-levels"></div></div>
</div>
</div>`;
}
