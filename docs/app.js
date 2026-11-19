const state = {
  data: null,
  model: null,
  bits: 4.85,
  context: 8192,
  users: 1,
  target: 20,
  vendor: "all",
  legacy: false,
  view: "gpus",
  filter: "fits",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const byId = (id) => document.getElementById(id);
const cleanNumber = (value) => Number.isFinite(Number(value)) ? Number(value) : null;
const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
const decimal = new Intl.NumberFormat("en", { maximumFractionDigits: 1, minimumFractionDigits: 0 });

function iconize() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.7 } });
}

function urlList(value) {
  return String(value || "").split(/;\s*/).map((item) => item.trim()).filter((item) => /^https?:\/\//i.test(item));
}

function sourceLabel(url) {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    if (host.includes("huggingface")) return "Hugging Face";
    if (host.includes("intel")) return "Intel ARK";
    if (host.includes("nvidia")) return "NVIDIA";
    if (host.includes("amd")) return "AMD";
    if (host.includes("apple")) return "Apple";
    return host;
  } catch {
    return "Source";
  }
}

function evidenceClass(text) {
  const value = String(text || "");
  const verified = Number((value.match(/(\d+) verified/) || [0, 0])[1]);
  if (/corrected/.test(value) && Number((value.match(/(\d+) corrected/) || [0, 0])[1]) > 0) return "red";
  return verified > 0 ? "" : "amber";
}

function evidenceShort(text) {
  const value = String(text || "unverified");
  const verified = (value.match(/(\d+) verified/) || [])[1];
  const corrected = (value.match(/(\d+) corrected/) || [])[1];
  if (verified && corrected === "0") return `${verified} verified`;
  if (corrected && corrected !== "0") return `${corrected} corrected`;
  return "Review source";
}

function kvPerTokenGb(model) {
  return 2 * Number(model.Layers) * Number(model["KV heads"]) * Number(model["Head dim"]) * 2 / 1e9;
}

function memoryNeed(model) {
  const weights = Number(model["Total params (B)"]) * state.bits / 8 * 1.03;
  const kv = kvPerTokenGb(model) * state.context * state.users;
  return { weights, kv, runtime: 1.5, total: weights + kv + 1.5 };
}

function efficiency(vendor, cpu = false) {
  if (cpu || vendor === "Intel") return 0.55;
  if (vendor === "Apple") return 0.65;
  return 0.6;
}

function tokPerSecond(hardware, model, cpu = false) {
  const bandwidth = Number(cpu ? hardware["Memory BW (GB/s)"] : hardware["Bandwidth (GB/s)"]);
  const perTokenWork = Number(model["Active params (B)"]) * state.bits / 8 + kvPerTokenGb(model);
  return efficiency(cpu ? hardware.Vendor || "" : hardware.Vendor, cpu) * bandwidth / perTokenWork / state.users;
}

function gpuUsable(hardware) {
  return Number(hardware["Memory (GB)"]) * 0.92;
}

function cpuUsable(hardware) {
  const memory = Number(hardware["Max RAM (GB)"]);
  return memory - Math.min(8, memory * 0.25);
}

function hardwareRows() {
  const need = memoryNeed(state.model).total;
  if (state.view === "cpus") {
    return state.data.cpus
      .filter((row) => cleanNumber(row["Max RAM (GB)"]) && cleanNumber(row["Memory BW (GB/s)"]))
      .map((row) => ({
        raw: row,
        name: row["CPU / platform"],
        vendor: row.Class,
        subtitle: `${row.Cores} cores · ${row.Memory}`,
        memory: cpuUsable(row),
        physicalMemory: Number(row["Max RAM (GB)"]),
        bandwidth: Number(row["Memory BW (GB/s)"]),
        power: row["Max turbo / max power (W)"],
        speed: tokPerSecond(row, state.model, true),
        fits: cpuUsable(row) >= need,
        evidence: row["Evidence status"],
        source: row["Source URL"],
        kind: "cpu",
      }));
  }
  return state.data.gpus
    .filter((row) => cleanNumber(row["Memory (GB)"]) && cleanNumber(row["Bandwidth (GB/s)"]))
    .filter((row) => state.vendor === "all" || row.Vendor === state.vendor)
    .filter((row) => state.legacy || String(row["Used / legacy?"]).toLowerCase() !== "yes")
    .map((row) => ({
      raw: row,
      name: row.Accelerator,
      vendor: row.Vendor,
      subtitle: `${row.Architecture} · ${row.Interface}`,
      memory: gpuUsable(row),
      physicalMemory: Number(row["Memory (GB)"]),
      bandwidth: Number(row["Bandwidth (GB/s)"]),
      power: row["TDP / board power (W)"],
      speed: tokPerSecond(row, state.model, false),
      fits: gpuUsable(row) >= need,
      evidence: row["Evidence status"],
      source: row["Source URL"],
      kind: "gpu",
    }));
}

function filteredRows() {
  const rows = hardwareRows();
  return rows
    .filter((row) => state.filter === "all" || (state.filter === "fits" ? row.fits : row.fits && row.speed >= state.target))
    .sort((a, b) => {
      const aBucket = a.fits ? 0 : 1;
      const bBucket = b.fits ? 0 : 1;
      if (aBucket !== bBucket) return aBucket - bBucket;
      if (state.filter === "all" && !a.fits && !b.fits) return b.memory - a.memory;
      return b.speed - a.speed;
    });
}

function setModel(model) {
  state.model = model;
  byId("model-search").value = model.Model;
  closeSearch();
  render();
}

function renderSummary() {
  const need = memoryNeed(state.model);
  const allRows = hardwareRows();
  const fits = allRows.filter((row) => row.fits).length;
  byId("model-summary-title").textContent = state.model.Model;
  byId("total-params").textContent = `${decimal.format(state.model["Total params (B)"])}B`;
  byId("active-params").textContent = `${decimal.format(state.model["Active params (B)"])}B`;
  byId("need-memory").textContent = `${decimal.format(need.total)} GB`;
  byId("fits-count").textContent = fits;
  byId("weights-label").textContent = `${decimal.format(need.weights)} GB`;
  byId("kv-label").textContent = `${decimal.format(need.kv)} GB`;
  byId("runtime-label").textContent = `${decimal.format(need.runtime)} GB`;
  byId("weights-segment").style.width = `${need.weights / need.total * 100}%`;
  byId("kv-segment").style.width = `${need.kv / need.total * 100}%`;
  byId("runtime-segment").style.width = `${need.runtime / need.total * 100}%`;
  byId("gpu-count").textContent = hardwareRowsForView("gpus").filter((row) => row.fits).length;
  byId("cpu-count").textContent = hardwareRowsForView("cpus").filter((row) => row.fits).length;
}

function hardwareRowsForView(view) {
  const current = state.view;
  state.view = view;
  const rows = hardwareRows();
  state.view = current;
  return rows;
}

function vendorInitial(vendor, name) {
  if (vendor === "NVIDIA") return "NV";
  if (vendor === "Intel") return "IN";
  if (vendor === "AMD") return "AM";
  if (vendor === "Apple") return "AP";
  if (/Mac/.test(name)) return "AP";
  if (/Jetson|NVIDIA/.test(name)) return "NV";
  if (/Ryzen|EPYC|Threadripper/.test(name)) return "AM";
  if (/Xeon|Core/.test(name)) return "IN";
  return String(vendor || "CPU").slice(0, 2).toUpperCase();
}

function renderRows() {
  const host = byId("results");
  const rows = filteredRows();
  host.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = "<h3>No hardware clears this filter.</h3><p>Try a lower quantization, fewer users, shorter context, or show all hardware.</p>";
    host.append(empty);
    return;
  }

  const need = memoryNeed(state.model).total;
  for (const [index, row] of rows.entries()) {
    const item = document.createElement("article");
    item.className = "result-row";
    item.style.setProperty("--delay", `${Math.min(index, 12) * 22}ms`);
    const ratio = row.memory / need;
    const bar = Math.min(100, row.memory / Math.max(need, row.memory) * 100);
    const speedClass = row.fits && row.speed >= state.target ? "fast" : "slow";
    const speedLabel = row.fits ? (row.speed >= state.target ? "Meets target" : "Fits, slower") : "Memory shortfall";
    const evidenceLevel = evidenceClass(row.evidence);
    item.innerHTML = `
      <div class="hardware-cell">
        <span class="vendor-chip">${vendorInitial(row.vendor, row.name)}</span>
        <div class="hardware-name"><strong>${escapeHtml(row.name)}</strong><small>${escapeHtml(row.subtitle)}</small></div>
      </div>
      <div class="fit-cell">
        <div class="fit-meta"><b>${decimal.format(row.memory)} GB usable</b><span>${row.fits ? `${decimal.format(ratio * 100)}%` : `short ${decimal.format(need - row.memory)} GB`}</span></div>
        <div class="fit-bar ${row.fits ? "" : "over"}"><span style="width:${bar}%"></span></div>
        <div class="fit-copy">Need <strong>${decimal.format(need)} GB</strong> from ${decimal.format(row.physicalMemory)} GB ${row.kind === "gpu" ? "device memory" : "system RAM"}</div>
      </div>
      <div class="speed-cell"><strong class="${speedClass}">${decimal.format(row.speed)}</strong><small>tok/s est. · ${speedLabel}</small></div>
      <div class="power-cell">${row.power ?? "—"} W</div>
      <div class="evidence-cell"><span class="status-pill ${evidenceLevel}">${evidenceShort(row.evidence)}</span><button class="row-detail" aria-label="Fact check ${escapeHtml(row.name)}" title="Fact check"><i data-lucide="arrow-up-right"></i></button></div>`;
    item.querySelector(".row-detail").addEventListener("click", () => openEvidence(row.raw, row.kind));
    host.append(item);
  }
  iconize();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

function render() {
  if (!state.model) return;
  renderSummary();
  renderRows();
}

function showSearch() {
  const input = byId("model-search");
  const query = input.value.toLowerCase().trim();
  const rows = state.data.models.filter((row) => row.Model.toLowerCase().includes(query)).slice(0, 18);
  const menu = byId("model-menu");
  menu.replaceChildren();
  for (const row of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "option";
    button.innerHTML = `<span>${escapeHtml(row.Model)}</span><small>${decimal.format(row["Total params (B)"])}B · ${escapeHtml(row.Vendor)}</small>`;
    button.addEventListener("click", () => setModel(row));
    menu.append(button);
  }
  menu.hidden = rows.length === 0;
  input.setAttribute("aria-expanded", String(!menu.hidden));
}

function closeSearch() {
  byId("model-menu").hidden = true;
  byId("model-search").setAttribute("aria-expanded", "false");
}

function evidenceFacts(sheet, item) {
  const key = `${sheet}|${item}`;
  return state.data.factsByItem[key] || [];
}

function openEvidence(row, kind) {
  const sheet = kind === "model" ? "Models" : kind === "cpu" ? "CPUs" : "GPUs";
  const item = kind === "model" ? row.Model : kind === "cpu" ? row["CPU / platform"] : row.Accelerator;
  const facts = evidenceFacts(sheet, item);
  const title = byId("drawer-title");
  const content = byId("drawer-content");
  title.textContent = item;
  content.replaceChildren();

  const intro = document.createElement("p");
  intro.textContent = facts.length
    ? `${facts.length} source checks are recorded for this row. Status reflects the public evidence log, not a claim of benchmark validation.`
    : "No parameter-level evidence records were found for this public row.";
  content.append(intro);

  const sourceUrls = urlList(row["Source URL"]);
  if (sourceUrls.length) {
    const links = document.createElement("div");
    links.className = "drawer-links";
    for (const url of sourceUrls) {
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.innerHTML = `${escapeHtml(sourceLabel(url))} <i data-lucide="external-link"></i>`;
      links.append(link);
    }
    content.append(links);
  }

  if (facts.length) {
    const list = document.createElement("div");
    list.className = "fact-list";
    for (const fact of facts) {
      const block = document.createElement("article");
      block.className = "fact-row";
      const heading = document.createElement("h3");
      heading.textContent = fact.Parameter || "Source";
      const body = document.createElement("div");
      const value = document.createElement("div");
      value.className = "fact-value";
      value.textContent = `${fact["Value in sheet (before)"] || "—"}${fact["Corrected to"] ? ` → ${fact["Corrected to"]}` : ""}`;
      const status = document.createElement("span");
      status.className = "fact-status";
      status.textContent = fact.Status || "—";
      const note = document.createElement("div");
      note.className = "fact-note";
      note.textContent = fact.Note || fact["Source value / quote"] || "No note.";
      body.append(value, status, note);
      block.append(heading, body);
      list.append(block);
    }
    content.append(list);
  }

  byId("drawer-scrim").hidden = false;
  byId("evidence-drawer").classList.add("open");
  byId("evidence-drawer").setAttribute("aria-hidden", "false");
  iconize();
  byId("close-drawer").focus();
}

function closeDrawer() {
  byId("drawer-scrim").hidden = true;
  byId("evidence-drawer").classList.remove("open");
  byId("evidence-drawer").setAttribute("aria-hidden", "true");
}

function bindEvents() {
  byId("model-search").addEventListener("focus", showSearch);
  byId("model-search").addEventListener("input", showSearch);
  byId("model-search").addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSearch();
    if (event.key === "Enter") {
      const exact = state.data.models.find((row) => row.Model.toLowerCase() === event.target.value.toLowerCase());
      if (exact) setModel(exact);
      else byId("model-menu").querySelector("button")?.click();
    }
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".model-control")) closeSearch();
  });
  byId("quant-select").addEventListener("change", (event) => {
    state.bits = Number(event.target.selectedOptions[0].dataset.bits);
    render();
  });
  byId("context-select").addEventListener("change", (event) => { state.context = Number(event.target.value); render(); });
  byId("user-select").addEventListener("change", (event) => { state.users = Number(event.target.value); render(); });
  byId("target-input").addEventListener("input", (event) => { state.target = Number(event.target.value) || 20; renderRows(); });
  byId("vendor-select").addEventListener("change", (event) => { state.vendor = event.target.value; render(); });
  byId("legacy-toggle").addEventListener("click", () => { state.legacy = !state.legacy; byId("legacy-toggle").setAttribute("aria-pressed", String(state.legacy)); render(); });
  byId("fit-filter").addEventListener("change", (event) => { state.filter = event.target.value; renderRows(); });
  $$(".view-switcher button").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    $$(".view-switcher button").forEach((item) => item.classList.toggle("active", item === button));
    render();
  }));
  byId("show-model-evidence").addEventListener("click", () => openEvidence(state.model, "model"));
  byId("close-drawer").addEventListener("click", closeDrawer);
  byId("drawer-scrim").addEventListener("click", closeDrawer);
  byId("open-about").addEventListener("click", () => byId("about-dialog").showModal());
  byId("close-about").addEventListener("click", () => byId("about-dialog").close());
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
}

async function boot() {
  try {
    const response = await fetch("data.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`data.json returned ${response.status}`);
    state.data = await response.json();
    byId("dataset-count").textContent = `${state.data.models.length} models · ${state.data.gpus.length} GPUs · ${state.data.cpus.length} CPUs · ${state.data.evidence.length} checks`;
    const quantSelect = byId("quant-select");
    for (const quant of state.data.quants) {
      const option = document.createElement("option");
      option.value = quant.Quantization;
      option.dataset.bits = quant["Bits/weight (eff.)"];
      option.textContent = quant.Quantization;
      if (quant.Quantization.includes("Q4_K_M")) option.selected = true;
      quantSelect.append(option);
    }
    state.bits = Number(quantSelect.selectedOptions[0].dataset.bits);
    state.model = state.data.models.find((row) => row.Model === "Qwen3 32B") || state.data.models[0];
    bindEvents();
    render();
    iconize();
  } catch (error) {
    byId("dataset-count").textContent = "Data could not be loaded";
    byId("model-summary-title").textContent = "Serve this page from a local or GitHub Pages web server.";
    byId("results").innerHTML = `<div class="empty-state"><h3>${escapeHtml(error.message)}</h3><p>Opening index.html directly blocks data.json. Run <code>python -m http.server 8000 -d docs</code>.</p></div>`;
  }
}

boot();
