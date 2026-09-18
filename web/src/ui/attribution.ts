export function renderAttribution(): string {
  return `
<section class="bar-group bar-group-attribution" id="attribution-group" hidden>
<div class="bar-head"><h3>Command attribution</h3><span class="bar-head-meta">gradient × input · decoder only</span></div>
<div class="attribution-grid" id="attribution"></div>
</section>`;
}
