/* Theme controller: light, dark, or follow the operating system. The <head> snippet sets the first paint. */
(function () {
  "use strict";
  const KEY = "modelfit-theme";
  const ORDER = ["system", "light", "dark"];
  const LABEL = { system: "Match system", light: "Light", dark: "Dark" };
  const ICON = { system: "monitor", light: "sun", dark: "moon" };
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  const root = document.documentElement;

  function stored() {
    try {
      const value = localStorage.getItem(KEY);
      return ORDER.includes(value) ? value : "system";
    } catch {
      return "system";
    }
  }

  function resolve(choice) {
    return choice === "system" ? (media && media.matches ? "dark" : "light") : choice;
  }

  function paint(choice) {
    const theme = resolve(choice);
    root.dataset.theme = theme;
    root.dataset.themeChoice = choice;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", getComputedStyle(root).getPropertyValue("--paper").trim() || (theme === "dark" ? "#101512" : "#f5f5ef"));
    const button = document.getElementById("theme-toggle");
    if (button) {
      const next = ORDER[(ORDER.indexOf(choice) + 1) % ORDER.length];
      button.setAttribute("aria-label", `Theme: ${LABEL[choice]}. Switch to ${LABEL[next]}.`);
      button.title = `Theme: ${LABEL[choice]}`;
      button.dataset.choice = choice;
      button.innerHTML = `<i data-lucide="${ICON[choice]}"></i>`;
      if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.7 } });
    }
    document.dispatchEvent(new CustomEvent("modelfit:theme", { detail: { theme, choice } }));
  }

  function set(choice) {
    const value = ORDER.includes(choice) ? choice : "system";
    try {
      if (value === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, value);
    } catch { /* storage blocked: the choice still applies for this visit */ }
    paint(value);
  }

  function cycle() {
    const current = root.dataset.themeChoice || stored();
    set(ORDER[(ORDER.indexOf(current) + 1) % ORDER.length]);
  }

  function init() {
    const button = document.getElementById("theme-toggle");
    if (button) button.addEventListener("click", cycle);
    paint(stored());
  }

  if (media) {
    const onChange = () => { if ((root.dataset.themeChoice || stored()) === "system") paint("system"); };
    if (media.addEventListener) media.addEventListener("change", onChange);
    else if (media.addListener) media.addListener(onChange);
  }
  window.addEventListener("storage", (event) => { if (event.key === KEY) paint(stored()); });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.ModelFitTheme = { set, cycle, get choice() { return root.dataset.themeChoice || stored(); }, get theme() { return root.dataset.theme; } };
})();
