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
  page: "model",
  intelOnly: false,
  minMemory: 0,
  maxPower: 0,
  modelVendor: "all",
  modelType: "all",
  hardware: null,
  hardwareKind: "gpu",
  hardwareBits: 4.85,
  hardwareContext: 8192,
  hardwareUsers: 1,
  hardwareTarget: 20,
  evidenceStatus: "all",
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

function excelMemoryNeed(model, bits = state.bits, context = state.context, users = state.users) {
  const weights = Number(model["Total params (B)"]) * bits / 8;
  const kv = kvPerTokenGb(model) * context * users;
  const runtime = Math.max(1, 0.1 * (weights + kv));
  return { weights, kv, runtime, total: weights + kv + runtime };
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

function isIntelHardware(row, kind = "gpu") {
  if (kind === "gpu") return row.Vendor === "Intel";
  return /^(Intel |Core |Core i|Xeon )|\bCore Ultra\b|\bXeon\b/.test(String(row["CPU / platform"] || ""));
}

function modelMatchesFilters(row, prefix = "model") {
  const vendorId = prefix === "hardware" ? "hardware-model-vendor-filter" : "model-vendor-filter";
  const typeId = prefix === "hardware" ? "hardware-model-type-filter" : "model-type-filter";
  const vendor = byId(vendorId)?.value || "all";
  const type = byId(typeId)?.value || "all";
  return (vendor === "all" || row.Vendor === vendor) && (type === "all" || row.Type === type);
}

function hardwareRows() {
  const need = memoryNeed(state.model).total;
  if (state.view === "cpus") {
    return state.data.cpus
      .filter((row) => cleanNumber(row["Max RAM (GB)"]) && cleanNumber(row["Memory BW (GB/s)"]))
      .filter((row) => !state.intelOnly || isIntelHardware(row, "cpu"))
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
    .filter((row) => !state.intelOnly || isIntelHardware(row, "gpu"))
    .filter((row) => state.vendor === "all" || row.Vendor === state.vendor)
    .filter((row) => !state.minMemory || Number(row["Memory (GB)"]) >= state.minMemory)
    .filter((row) => !state.maxPower || Number(row["TDP / board power (W)"]) <= state.maxPower)
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
  const excel = excelMemoryNeed(state.model);
  const parityDelta = (need.total / excel.total - 1) * 100;
  byId("memory-parity").innerHTML = `<span><b>Excel fit estimate</b> ${decimal.format(excel.total)} GB</span><span><b>Web estimate</b> ${decimal.format(need.total)} GB</span><span class="${Math.abs(parityDelta) > 2 ? "delta-hot" : ""}"><b>Delta</b> ${parityDelta >= 0 ? "+" : ""}${decimal.format(parityDelta)}%</span>`;
  byId("cpu-count").textContent = hardwareRowsForView("cpus").filter((row) => row.fits).length;
}

function hardwareRowsForView(view) {
  const current = state.view;
  state.view = view;
  const rows = hardwareRows();
  state.view = current;
  return rows;
}

function excelEfficiency(hardware, cpu = false) {
  if (!cpu) {
    if (hardware.Vendor === "Intel") return 0.5;
    if (hardware.Vendor === "Apple") return 0.65;
    if (hardware.Vendor === "NVIDIA" && String(hardware.Accelerator) === "GT 710 1GB") return 0.45;
    return hardware.Vendor === "NVIDIA" ? 0.65 : 0.6;
  }
  const name = String(hardware["CPU / platform"] || "");
  if (/^Mac |Apple M4/.test(name)) return 0.65;
  if (/Jetson/.test(name)) return 0.6;
  return 0.55;
}

function excelTokPerSecond(hardware, model, cpu = false) {
  const bandwidth = Number(cpu ? hardware["Memory BW (GB/s)"] : hardware["Bandwidth (GB/s)"]);
  const kv = kvPerTokenGb(model) * state.context * state.users;
  const denominator = Number(model["Active params (B)"]) * state.bits / 8 + kv;
  return excelEfficiency(hardware, cpu) * bandwidth / denominator / state.users;
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
    const excelSpeed = excelTokPerSecond(row.raw, state.model, row.kind === "cpu");
    const speedDelta = excelSpeed ? (row.speed / excelSpeed - 1) * 100 : 0;
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
      <div class="speed-cell"><strong class="${speedClass}">${decimal.format(row.speed)}</strong><small>tok/s web · ${speedLabel}</small><small>Excel ${decimal.format(excelSpeed)} tok/s</small><small class="delta">${speedDelta >= 0 ? "+" : ""}${decimal.format(speedDelta)}% web vs Excel</small></div>
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
  const rows = state.data.models.filter((row) => row.Model.toLowerCase().includes(query) && modelMatchesFilters(row, "model")).slice(0, 18);
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

function renderParityAudit() {
  const parity = state.parity;
  const host = byId("parity-audit-content");
  if (!parity) { host.textContent = "Parity audit unavailable."; return; }
  const a = parity["8k"];
  const b = parity["32k"];
  const signed = (value) => `${value >= 0 ? "+" : ""}${decimal.format(value)}%`;
  host.innerHTML = `<div class="parity-metrics"><div><strong>${signed(a.memory_delta_pct.median)}</strong><small>Median 8K memory delta</small></div><div><strong>${signed(a.gpu_decode_delta_pct.median)}</strong><small>Median 8K GPU decode delta</small></div><div><strong>${signed(b.gpu_decode_delta_pct.median)}</strong><small>Median 32K GPU decode delta</small></div><div><strong>${signed(b.cpu_decode_delta_pct.median)}</strong><small>Median 32K CPU decode delta</small></div></div><p class="parity-caption">Positive means web is higher. The gap grows with context because Excel counts the full KV allocation in per-token traffic; web counts KV bytes read per token. The 8K medians are +3.08% memory, +12.74% GPU decode and +13.86% CPU decode. At 32K they are -0.02% memory, +58.58% GPU decode and +60.88% CPU decode.</p>`;
}

function hardwarePool() {
  const gpus = state.data.gpus.map((row) => ({ raw: row, kind: "gpu", name: row.Accelerator }));
  const cpus = state.data.cpus.map((row) => ({ row, raw: row, kind: "cpu", name: row["CPU / platform"] }));
  return [...gpus, ...cpus];
}

function hardwareMemory(item) {
  return item.kind === "gpu" ? gpuUsable(item.raw) : cpuUsable(item.raw);
}

function hardwareBandwidth(item) {
  return Number(item.kind === "gpu" ? item.raw["Bandwidth (GB/s)"] : item.raw["Memory BW (GB/s)"]);
}

function hardwareNeed(model) {
  const weights = Number(model["Total params (B)"]) * state.hardwareBits / 8 * 1.03;
  const kv = kvPerTokenGb(model) * state.hardwareContext * state.hardwareUsers;
  return { weights, kv, total: weights + kv + 1.5 };
}

function hardwareSpeed(item, model) {
  const denominator = Number(model["Active params (B)"]) * state.hardwareBits / 8 + kvPerTokenGb(model);
  return efficiency(item.kind === "gpu" ? item.raw.Vendor : "", item.kind === "cpu") * hardwareBandwidth(item) / denominator / state.hardwareUsers;
}

function setHardware(item) {
  state.hardware = item.raw;
  state.hardwareKind = item.kind;
  byId("hardware-search").value = item.name;
  closeHardwareSearch();
  renderHardwareView();
}

function showHardwareSearch() {
  const query = byId("hardware-search").value.toLowerCase().trim();
  const rows = hardwarePool()
    .filter((item) => item.name.toLowerCase().includes(query))
    .filter((item) => !state.intelOnly || isIntelHardware(item.raw, item.kind))
    .slice(0, 22);
  const menu = byId("hardware-menu");
  menu.replaceChildren();
  for (const item of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "option";
    const memory = item.kind === "gpu" ? item.raw["Memory (GB)"] : item.raw["Max RAM (GB)"];
    button.innerHTML = `<span>${escapeHtml(item.name)}</span><small>${item.kind.toUpperCase()} · ${escapeHtml(memory)} GB</small>`;
    button.addEventListener("click", () => setHardware(item));
    menu.append(button);
  }
  menu.hidden = rows.length === 0;
  byId("hardware-search").setAttribute("aria-expanded", String(!menu.hidden));
}

function closeHardwareSearch() {
  byId("hardware-menu").hidden = true;
  byId("hardware-search").setAttribute("aria-expanded", "false");
}

function currentHardwareItem() {
  return { raw: state.hardware, kind: state.hardwareKind, name: state.hardwareKind === "gpu" ? state.hardware.Accelerator : state.hardware["CPU / platform"] };
}

function compatibleModels(item) {
  const capacity = hardwareMemory(item);
  const filter = byId("hardware-model-filter").value;
  return state.data.models
    .filter((model) => model["Active params (B)"] != null && model.Layers != null && model["KV heads"] != null && model["Head dim"] != null)
    .filter((model) => modelMatchesFilters(model, "hardware"))
    .map((model) => ({ model, need: hardwareNeed(model).total, speed: hardwareSpeed(item, model) }))
    .map((row) => ({ ...row, fits: row.need <= capacity }))
    .filter((row) => filter === "all" || (filter === "fits" ? row.fits : row.fits && row.speed >= state.hardwareTarget))
    .sort((a, b) => Number(b.fits) - Number(a.fits) || b.speed - a.speed);
}

function renderHardwareView() {
  if (!state.hardware) return;
  const item = currentHardwareItem();
  const capacity = hardwareMemory(item);
  const rows = compatibleModels(item);
  const fits = compatibleModels(item).filter((row) => row.fits).length;
  const source = item.raw["Source URL"] || "";
  const summary = byId("hardware-summary");
  summary.innerHTML = `<div><p class="section-label">Selected ${item.kind === "gpu" ? "accelerator" : "CPU platform"}</p><h2>${escapeHtml(item.name)}</h2><button class="text-button" id="show-hardware-evidence"><i data-lucide="badge-check"></i><span>Fact check</span></button></div><div class="metric"><span>${decimal.format(capacity)} GB</span><small>Usable memory</small></div><div class="metric"><span>${decimal.format(hardwareBandwidth(item))} GB/s</span><small>Bandwidth</small></div><div class="metric"><span>${fits}</span><small>Models that fit</small></div>`;
  summary.querySelector("#show-hardware-evidence").addEventListener("click", () => openEvidence(item.raw, item.kind));
  const host = byId("hardware-model-results");
  host.replaceChildren();
  if (!rows.length) {
    host.innerHTML = `<div class="empty-state"><h3>No model matches these filters.</h3><p>Lower the quantization, shorten the context, or select a different model family.</p></div>`;
    return;
  }
  for (const row of rows.slice(0, 150)) {
    const model = row.model;
    const ratio = row.need / capacity * 100;
    const fast = row.fits && row.speed >= state.hardwareTarget;
    const article = document.createElement("article");
    article.className = "result-row model-result-row";
    const excelNeed = excelMemoryNeed(model, state.hardwareBits, state.hardwareContext, state.hardwareUsers).total;
    article.innerHTML = `<div class="hardware-cell"><span class="vendor-chip">${escapeHtml(String(model.Vendor || "LLM").slice(0, 2).toUpperCase())}</span><div class="hardware-name"><strong>${escapeHtml(model.Model)}</strong><small>${escapeHtml(model.Type)} · ${decimal.format(model["Total params (B)"])}B parameters</small></div></div><div class="fit-cell"><div class="fit-meta"><b>${decimal.format(row.need)} GB needed</b><span>${row.fits ? `${decimal.format(ratio)}%` : `short ${decimal.format(row.need - capacity)} GB`}</span></div><div class="fit-bar ${row.fits ? "" : "over"}"><span style="width:${Math.min(100, ratio)}%"></span></div><div class="fit-copy">${decimal.format(hardwareNeed(model).weights)} GB weights · ${decimal.format(hardwareNeed(model).kv)} GB KV · Excel ${decimal.format(excelNeed)} GB (${(row.need / excelNeed - 1) * 100 >= 0 ? "+" : ""}${decimal.format((row.need / excelNeed - 1) * 100)}%)</div></div><div class="speed-cell"><strong class="${fast ? "fast" : "slow"}">${decimal.format(row.speed)}</strong><small>tok/s est. · ${fast ? "Meets target" : row.fits ? "Fits, slower" : "Memory shortfall"}</small></div><div class="hardware-arch"><code>${escapeHtml(model["Runtime arch"] || "—")}</code><small>${model.Layers}L × ${model["KV heads"]} KV × ${model["Head dim"]}d</small></div><div class="evidence-cell"><span class="status-pill ${evidenceClass(model["Evidence status"])}">${evidenceShort(model["Evidence status"])}</span><button class="row-detail" aria-label="Fact check ${escapeHtml(model.Model)}"><i data-lucide="arrow-up-right"></i></button></div>`;
    article.querySelector(".row-detail").addEventListener("click", () => openEvidence(model, "model"));
    host.append(article);
  }
  iconize();
}

function showPage(page) {
  state.page = page;
  $$(".page-view").forEach((view) => {
    const active = view.id === `${page}-page`;
    view.hidden = !active;
    view.classList.toggle("active", active);
  });
  $$(".nav-page").forEach((button) => button.classList.toggle("active", button.dataset.page === page));
  if (page === "hardware") renderHardwareView();
  if (page === "database") renderDatabase();
  if (page === "evidence") renderEvidence();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function populateFilterOptions() {
  const vendors = [...new Set(state.data.models.map((row) => row.Vendor).filter(Boolean))].sort();
  const types = [...new Set(state.data.models.map((row) => row.Type).filter(Boolean))].sort();
  for (const id of ["model-vendor-filter", "hardware-model-vendor-filter"]) for (const value of vendors) byId(id).add(new Option(value, value));
  for (const id of ["model-type-filter", "hardware-model-type-filter"]) for (const value of types) byId(id).add(new Option(value, value));
  const target = byId("hardware-quant-select");
  for (const quant of state.data.quants) {
    const option = new Option(quant.Quantization, quant.Quantization, quant.Quantization.includes("Q4_K_M"), quant.Quantization.includes("Q4_K_M"));
    option.dataset.bits = quant["Bits/weight (eff.)"];
    target.add(option);
  }
  const defaultHardware = hardwarePool().find((item) => item.name === "Intel Arc Pro B60") || hardwarePool()[0];
  setHardware(defaultHardware);
}

function databaseRows() {
  const name = byId("database-select").value;
  return state.data[name];
}

function renderDatabase() {
  const rows = databaseRows();
  const query = byId("database-search").value.toLowerCase().trim();
  const table = byId("database-table");
  table.replaceChildren();
  const visible = rows.filter((row) => !query || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query))).slice(0, 500);
  if (!visible.length) {
    table.innerHTML = "<tbody><tr><td>No matching rows.</td></tr></tbody>";
    return;
  }
  const cols = Object.keys(visible[0]).filter((key) => !key.startsWith("_"));
  const head = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const col of cols) { const th = document.createElement("th"); th.textContent = col; hr.append(th); }
  head.append(hr);
  const body = document.createElement("tbody");
  for (const row of visible) {
    const tr = document.createElement("tr");
    for (const col of cols) {
      const td = document.createElement("td");
      const value = String(row[col] ?? "");
      if (col === "Source URL" && value) {
        td.textContent = evidenceShort(row["Evidence status"] || "Source");
        td.title = value;
        td.style.cursor = "pointer";
        td.addEventListener("click", () => window.open(urlList(value)[0], "_blank", "noopener"));
      } else {
        td.textContent = value;
        td.title = value;
      }
      tr.append(td);
    }
    body.append(tr);
  }
  table.append(head, body);
}

function renderEvidence() {
  const query = byId("evidence-search").value.toLowerCase().trim();
  let rows = state.data.evidence.filter((row) => {
    const matchesQuery = !query || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query));
    const matchesStatus = state.evidenceStatus === "all" || (state.evidenceStatus === "OTHER" ? !["VERIFIED", "MISMATCH", "DERIVED/ESTIMATE"].includes(row.Status) : row.Status === state.evidenceStatus);
    return matchesQuery && matchesStatus;
  });
  byId("evidence-count").textContent = `${rows.length} matching checks · ${state.data.evidence.length} total`;
  rows = rows.slice(0, 300);
  const host = byId("evidence-results");
  host.replaceChildren();
  for (const row of rows) {
    const article = document.createElement("article");
    article.className = "evidence-entry";
    const group = row.Status === "VERIFIED" || row.Status === "MISMATCH" || row.Status === "DERIVED/ESTIMATE" ? row.Status : "OTHER";
    article.dataset.status = group;
    article.innerHTML = `<div class="evidence-entry-head"><div><h3>${escapeHtml(row.Item)}</h3><code>${escapeHtml(row.Parameter)} · ${escapeHtml(row.Sheet)}</code></div><span class="status-pill ${row.Status === "VERIFIED" ? "" : row.Status === "MISMATCH" ? "red" : "amber"}">${escapeHtml(row.Status)}</span></div><p>${escapeHtml(row.Note || row["Source value / quote"] || "No note")}</p>${row["Source URL"] ? `<a href="${escapeHtml(row["Source URL"])}" target="_blank" rel="noreferrer">${escapeHtml(sourceLabel(row["Source URL"]))} ↗</a>` : ""}`;
    host.append(article);
  }
  if (!rows.length) host.innerHTML = `<div class="empty-state"><h3>No evidence records match.</h3><p>Change the status filter or search text.</p></div>`;
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

  for (const button of $$(".nav-page")) button.addEventListener("click", () => showPage(button.dataset.page));
  for (const button of [byId("intel-only-toggle"), byId("hardware-intel-only-toggle")]) button.addEventListener("click", () => {
    state.intelOnly = !state.intelOnly;
    for (const toggle of [byId("intel-only-toggle"), byId("hardware-intel-only-toggle")]) toggle.setAttribute("aria-pressed", String(state.intelOnly));
    byId("vendor-select").value = state.intelOnly ? "Intel" : "all";
    state.vendor = state.intelOnly ? "Intel" : "all";
    if (state.intelOnly && state.hardware && !isIntelHardware(state.hardware, state.hardwareKind)) setHardware(hardwarePool().find((item) => isIntelHardware(item.raw, item.kind)));
    render();
    if (state.page === "hardware") renderHardwareView();
  });
  for (const id of ["model-vendor-filter", "model-type-filter"]) byId(id).addEventListener("change", showSearch);
  byId("min-memory-filter").addEventListener("change", (event) => { state.minMemory = Number(event.target.value); render(); });
  byId("max-power-filter").addEventListener("change", (event) => { state.maxPower = Number(event.target.value); render(); });

  byId("hardware-search").addEventListener("focus", showHardwareSearch);
  byId("hardware-search").addEventListener("input", showHardwareSearch);
  byId("hardware-search").addEventListener("keydown", (event) => { if (event.key === "Enter") byId("hardware-menu").querySelector("button")?.click(); if (event.key === "Escape") closeHardwareSearch(); });
  document.addEventListener("click", (event) => { if (!event.target.closest(".hardware-search-control")) closeHardwareSearch(); });
  byId("hardware-quant-select").addEventListener("change", (event) => { state.hardwareBits = Number(event.target.selectedOptions[0].dataset.bits); renderHardwareView(); });
  byId("hardware-context-select").addEventListener("change", (event) => { state.hardwareContext = Number(event.target.value); renderHardwareView(); });
  byId("hardware-user-select").addEventListener("change", (event) => { state.hardwareUsers = Number(event.target.value); renderHardwareView(); });
  byId("hardware-target-input").addEventListener("input", (event) => { state.hardwareTarget = Number(event.target.value) || 20; renderHardwareView(); });
  for (const id of ["hardware-model-vendor-filter", "hardware-model-type-filter", "hardware-model-filter"]) byId(id).addEventListener("change", renderHardwareView);

  byId("database-select").addEventListener("change", renderDatabase);
  byId("database-search").addEventListener("input", renderDatabase);
  byId("download-table").addEventListener("click", () => {
    const name = byId("database-select").value;
    const rows = databaseRows();
    const cols = rows.length ? Object.keys(rows[0]).filter((key) => !key.startsWith("_")) : [];
    const csvCell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const csv = [cols.map(csvCell).join(","), ...rows.map((row) => cols.map((col) => csvCell(row[col])).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = `${name}.csv`; link.click(); URL.revokeObjectURL(url);
  });
  byId("evidence-search").addEventListener("input", renderEvidence);
  for (const button of $$(".status-filter")) button.addEventListener("click", () => {
    state.evidenceStatus = button.dataset.status;
    $$(".status-filter").forEach((item) => item.classList.toggle("active", item === button));
    renderEvidence();
  });
}

async function boot() {
  try {
    const response = await fetch("data.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`data.json returned ${response.status}`);
    state.data = await response.json();
    try {
      const parityResponse = await fetch("parity.json", { cache: "no-store" });
      if (parityResponse.ok) state.parity = await parityResponse.json();
    } catch { state.parity = null; }
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
    populateFilterOptions();
    bindEvents();
    render();
    renderEvidence();
    renderParityAudit();
    iconize();
  } catch (error) {
    byId("dataset-count").textContent = "Data could not be loaded";
    byId("model-summary-title").textContent = "Serve this page from a local or GitHub Pages web server.";
    byId("results").innerHTML = `<div class="empty-state"><h3>${escapeHtml(error.message)}</h3><p>Opening index.html directly blocks data.json. Run <code>python -m http.server 8000 -d docs</code>.</p></div>`;
  }
}

boot();
