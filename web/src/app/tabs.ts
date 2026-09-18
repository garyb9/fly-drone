import { els } from "../dom";

// One bottom panel, several tabs. Mode-aware tabs (free roam, attribution) are hidden when
// they carry no data; if the active tab disappears the view falls back to the controls.
export const TABS = ["controls", "signals", "attribution"] as const;
export type TabName = (typeof TABS)[number];

export function setTab(name: TabName): void {
  for (const tab of TABS) {
    const selected = tab === name;
    const button = els[`tab-${tab}`];
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    els[`panel-${tab}`].hidden = !selected;
  }
}

export function setTabAvailable(tab: TabName, available: boolean): void {
  const button = els[`tab-${tab}`];
  button.hidden = !available;
  if (!available && button.classList.contains("active")) setTab("controls");
}
