export function renderHelp(): string {
  return `
<div class="help-overlay" id="help-overlay" hidden>
<div class="help-card" role="dialog" aria-modal="true" aria-labelledby="help-title">
<div class="help-head"><h2 id="help-title">Keyboard shortcuts</h2><button id="help-close" aria-label="Close help">Close</button></div>
<ul>
<li><kbd>Space</kbd><span>Pause / resume</span></li>
<li><kbd>R</kbd><span>Reset trial</span></li>
<li><kbd>F</kbd><span>Toggle follow</span></li>
<li><kbd>1</kbd><span>Orbit · <kbd>2</kbd> first person · <kbd>3</kbd> third person</span></li>
<li><kbd>G</kbd><span>Ghost mode (free roam)</span></li>
<li><kbd>?</kbd><span>Show / hide this help</span></li>
<li><kbd>Esc</kbd><span>Close</span></li>
</ul>
<p class="help-note">The connectome is frozen; only the decoder learns. The drone mesh is drawn larger than life and the fly body is illustrative.</p>
</div>
</div>`;
}
