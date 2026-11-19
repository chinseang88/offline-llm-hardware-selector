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
  hardwareCpu: null,
  hardwareGpu: null,
  hardwareRam: 0,
  hardwareVram: 0,
  hardwareGpuQuantity: 1,
  hardwareRuntime: "llama.cpp",
  hardwareTask: "General chat / Q&A",
  hardwareKvBits: 8,
  hardwareOffload: "N",
  hardwareMemoryHeadroom: 8,
  hardwareBandwidthUtil: 100,
  evidenceStatus: "all",
  runtime: "llama.cpp",
  task: "General chat / Q&A",
  kvBits: 8,
  promptTokens: 2048,
  answerTokens: 256,
  offload: "N",
  gpuQuantity: 1,
  hostCpu: null,
  hostRam: 0,
  maxCpuPower: 500,
  pcieWidth: "Auto",
  preferredGpu: null,
  deviceVram: 0,
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

function kvPerTokenGb(model, bits = state.kvBits) {
  return 2 * Number(model.Layers) * Number(model["KV heads"]) * Number(model["Head dim"]) * bits / 8 / 1e9;
}

function memoryNeed(model) {
  const weights = Number(model["Total params (B)"]) * state.bits / 8 * 1.03;
  const kv = kvPerTokenGb(model, state.kvBits) * state.context * state.users;
  const overhead = weights * 0.07;
  return { weights, kv, runtime: overhead, total: weights + kv + overhead };
}

function excelMemoryNeed(model, bits = state.bits, context = state.context, users = state.users) {
  const weights = Number(model["Total params (B)"]) * bits / 8;
  const kv = kvPerTokenGb(model) * context * users;
  const runtime = Math.max(1, 0.1 * (weights + kv));
  return { weights, kv, runtime, total: weights + kv + runtime };
}

const runtimeEfficiency = {
  "llama.cpp": { NVIDIA: 0.65, AMD: 0.60, Intel: 0.50, Apple: 0.65, CPU: 0.55, Jetson: 0.60 },
  "Ollama": { NVIDIA: 0.62, AMD: 0.58, Intel: 0.45, Apple: 0.62, CPU: 0.52, Jetson: 0.58 },
  "LM Studio": { NVIDIA: 0.62, AMD: 0.58, Intel: 0.45, Apple: 0.65, CPU: 0.52, Jetson: 0 },
  "vLLM": { NVIDIA: 0.75, AMD: 0.70, Intel: 0.55, Apple: 0, CPU: 0.40, Jetson: 0.60 },
  "SGLang": { NVIDIA: 0.75, AMD: 0.70, Intel: 0, Apple: 0, CPU: 0, Jetson: 0 },
  "TensorRT-LLM": { NVIDIA: 0.80, AMD: 0, Intel: 0, Apple: 0, CPU: 0, Jetson: 0.70 },
  "ExLlamaV2 / V3": { NVIDIA: 0.75, AMD: 0.65, Intel: 0, Apple: 0, CPU: 0, Jetson: 0 },
  "MLX / mlx-lm": { NVIDIA: 0, AMD: 0, Intel: 0, Apple: 0.70, CPU: 0, Jetson: 0 },
  "OpenVINO GenAI": { NVIDIA: 0, AMD: 0, Intel: 0.60, Apple: 0, CPU: 0.55, Jetson: 0 },
  "IPEX-LLM": { NVIDIA: 0, AMD: 0, Intel: 0.58, Apple: 0, CPU: 0.50, Jetson: 0 },
};

function efficiency(vendor, cpu = false, hardware = null) {
  const profile = runtimeEfficiency[state.runtime] || runtimeEfficiency["llama.cpp"];
  if (cpu) {
    if (hardware && /Jetson/.test(hardware["CPU / platform"] || "")) return profile.Jetson;
    return profile.CPU;
  }
  return profile[vendor] ?? 0;
}

const taskRules = {
  "General chat / Q&A": { base: 4.85, small: 4.85, large: 4.85, note: "Q4_K_M is the baseline for chat." },
  "RAG / document Q&A": { base: 5.7, small: 8.5, large: 4.85, note: "RAG needs enough precision to preserve grounded facts." },
  "Coding": { base: 5.7, small: 8.5, large: 4.85, note: "Small coders need Q8; >=30B can use Q4." },
  "Maths / reasoning": { base: 5.7, small: 8.5, large: 4.85, note: "Reasoning amplifies small arithmetic and token errors." },
  "Agents / tool calling": { base: 5.7, small: 8.5, large: 4.85, note: "Tool calls require exact JSON and argument spelling." },
  "Vision (image + text)": { base: 4.85, small: 6.6, large: 4.85, note: "Vision requires a VLM and a separately loaded vision encoder." },
  "Classification / extraction": { base: 4.85, small: 4.85, large: 4.85, note: "Short constrained outputs tolerate quantization well." },
  "Translation / multilingual": { base: 5.7, small: 8.5, large: 4.85, note: "Low-resource languages degrade first at lower bits." },
};

function requiredBits(model, task = state.task) {
  const rule = taskRules[task] || taskRules["General chat / Q&A"];
  const total = Number(model["Total params (B)"]);
  return total < 4 ? rule.small : total >= 30 ? rule.large : rule.base;
}

function tokPerSecond(hardware, model, cpu = false) {
  const bandwidth = Number(cpu ? hardware["Memory BW (GB/s)"] : hardware["Bandwidth (GB/s)"]);
  const perTokenWork = Number(model["Active params (B)"]) * state.bits / 8;
  const runtimeEff = efficiency(cpu ? "" : hardware.Vendor, cpu, hardware);
  if (!runtimeEff) return 0;
  const quantityScale = cpu ? 1 : 1 + 0.8 * (state.gpuQuantity - 1);
  return runtimeEff * bandwidth * quantityScale / perTokenWork / state.users;
}

function gpuUsable(hardware) {
  const isPreferred = state.preferredGpu && hardware.Accelerator === state.preferredGpu.Accelerator;
  const memory = isPreferred && state.deviceVram > 0 ? state.deviceVram : Number(hardware["Memory (GB)"]);
  return memory * 0.92 * state.gpuQuantity;
}

function cpuUsable(hardware) {
  const isHost = state.hostCpu && hardware["CPU / platform"] === state.hostCpu["CPU / platform"];
  const memory = isHost && state.hostRam > 0 ? state.hostRam : Number(hardware["Max RAM (GB)"]);
  return memory - Math.min(8, memory * 0.25);
}

function isIntelHardware(row, kind = "gpu") {
  if (kind === "gpu") return row.Vendor === "Intel";
  const name = String(row["CPU / platform"] || "");
  return /^(Intel |Core Ultra|Core i[3579]|Xeon )|\bCore Ultra\b|\bXeon\b/.test(name);
}

function cpuGeneration(name) {
  const map = {
    "Intel N100 mini-PC": "Alder Lake-N",
    "Intel N150 mini-PC": "Twin Lake",
    "Intel N305 mini-PC": "Alder Lake-N",
    "Core Ultra 7 258V (Lunar Lake)": "Lunar Lake",
    "Core Ultra 5 125H (Meteor Lake)": "Meteor Lake",
    "Core Ultra 9 285H": "Arrow Lake-H",
    "Core Ultra 9 275HX": "Arrow Lake-HX",
    "Core i5-12400": "Alder Lake-S",
    "Core i7-14700": "Raptor Lake-S Refresh",
    "Core Ultra 7 265K": "Arrow Lake-S",
    "Core Ultra 9 285K": "Arrow Lake-S",
    "Xeon Gold 6430 (SPR, 1S)": "Sapphire Rapids-SP",
    "Xeon w5-2555X (W-2500)": "Sapphire Rapids W-2500",
    "Xeon w7-2595X (W-2500)": "Sapphire Rapids W-2500",
    "Xeon w9-3595X (W-3500)": "Sapphire Rapids W-3500",
    "Xeon 6 6980P (1S)": "Granite Rapids-SP",
  };
  return map[name] || "";
}

function cpuCatalogLabel(row) {
  const generation = cpuGeneration(row["CPU / platform"]);
  return `${row["CPU / platform"]}${generation ? ` · ${generation}` : ""} · ${row.Class} · ${row.Cores}C · ${row["Base / default power (W)"]}/${row["Max turbo / max power (W)"]}W · ${row.Memory} · ${row["Memory BW (GB/s)"]} GB/s · ${row["Max RAM (GB)"]} GB RAM · ${row["CPU PCIe lanes"]} PCIe lanes`;
}

function gpuCatalogLabel(row) {
  return `${row.Accelerator} · ${row["Memory (GB)"]} GB · ${row["Bandwidth (GB/s)"]} GB/s · ${row["TDP / board power (W)"]}W · ${row.Interface} · ${row.Architecture}`;
}

function rebuildCatalogSelect(id, rows, key, label, placeholder, filterKind) {
  const select = byId(id);
  const previous = select.value;
  select.replaceChildren();
  if (placeholder) select.add(new Option(placeholder.text, placeholder.value));
  const filtered = rows.filter((row) => !state.intelOnly || isIntelHardware(row, filterKind));
  for (const row of filtered) select.add(new Option(label(row), row[key]));
  select.value = filtered.some((row) => row[key] === previous) ? previous : (placeholder ? placeholder.value : filtered[0]?.[key] || "");
}

function selectClosest(select, value) {
  const nums = [...select.options].map((option) => Number(option.value)).filter((number) => Number.isFinite(number) && number > 0).sort((a, b) => a - b);
  const desired = Number(value);
  const valid = nums.filter((number) => number <= desired);
  select.value = String(valid.length ? valid[valid.length - 1] : nums[0] || 0);
  return Number(select.value);
}

function refreshCatalogDropdowns() {
  rebuildCatalogSelect("host-cpu-select", state.data.cpus, "CPU / platform", cpuCatalogLabel, { text: "Auto / none", value: "" }, "cpu");
  rebuildCatalogSelect("hardware-cpu-select", state.data.cpus, "CPU / platform", cpuCatalogLabel, null, "cpu");
  rebuildCatalogSelect("preferred-gpu-select", state.data.gpus, "Accelerator", gpuCatalogLabel, { text: "No preference", value: "" }, "gpu");
  rebuildCatalogSelect("hardware-gpu-select", state.data.gpus, "Accelerator", gpuCatalogLabel, null, "gpu");
  refreshHardwareDropdown();
  for (const id of ["device-vram-input", "hardware-vram-input"]) {
    const select = byId(id);
    const previous = select.value;
    const values = [...new Set(state.data.gpus.map((row) => Number(row["Memory (GB)"])).filter(Number.isFinite))].sort((a, b) => a - b);
    select.replaceChildren();
    if (id === "device-vram-input") select.add(new Option("Catalog VRAM", "0"));
    for (const value of values) select.add(new Option(`${value} GB`, String(value)));
    if ([...select.options].some((option) => option.value === previous)) select.value = previous;
  }
}

function modelMatchesFilters(row, prefix = "model") {
  const vendorId = prefix === "hardware" ? "hardware-model-vendor-filter" : "model-vendor-filter";
  const typeId = prefix === "hardware" ? "hardware-model-type-filter" : "model-type-filter";
  const vendor = byId(vendorId)?.value || "all";
  const type = byId(typeId)?.value || "all";
  return (vendor === "all" || row.Vendor === vendor) && (type === "all" || row.Type === type);
}

function hostOffloadGb() {
  if (state.offload !== "Y" || !state.hostCpu) return 0;
  const raw = state.hostRam > 0 ? state.hostRam : Number(state.hostCpu["Max RAM (GB)"]);
  return Math.max(0, raw - Math.max(8, raw * 0.1));
}

function hardwareRows() {
  const need = memoryNeed(state.model).total;
  if (state.view === "cpus") {
    return state.data.cpus
      .filter((row) => cleanNumber(row["Max RAM (GB)"]) && cleanNumber(row["Memory BW (GB/s)"]))
      .filter((row) => !state.intelOnly || isIntelHardware(row, "cpu"))
      .filter((row) => Number(row["Base / default power (W)"]) <= state.maxCpuPower)
      .filter((row) => state.pcieWidth === "Auto" || Number(row["CPU PCIe lanes"]) >= Number((state.pcieWidth.match(/\d+/) || [0])[0]) * state.gpuQuantity)
      .map((row) => ({
        raw: row,
        name: row["CPU / platform"],
        vendor: row.Class,
        subtitle: `${cpuGeneration(row["CPU / platform"]) || row.Class} · ${row.Cores} cores · ${row.Memory}`,
        memory: cpuUsable(row),
        physicalMemory: Number(row["Max RAM (GB)"]),
        bandwidth: Number(row["Memory BW (GB/s)"]),
        power: row["Max turbo / max power (W)"],
        speed: tokPerSecond(row, state.model, true),
        fits: cpuUsable(row) >= need && efficiency("", true, row) > 0 && state.bits >= requiredBits(state.model) && (state.task !== "Vision (image + text)" || /VLM|Vision/i.test(state.model.Type || "")),
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
      subtitle: `${row.Architecture} · ${row.Interface} · ${state.runtime}`,
      memory: gpuUsable(row) + hostOffloadGb(),
      physicalMemory: Number(row["Memory (GB)"]),
      bandwidth: Number(row["Bandwidth (GB/s)"]),
      power: row["TDP / board power (W)"],
      speed: tokPerSecond(row, state.model, false),
      fits: gpuUsable(row) + hostOffloadGb() >= need && efficiency(row.Vendor, false, row) > 0 && state.bits >= requiredBits(state.model) && (state.task !== "Vision (image + text)" || /VLM|Vision/i.test(state.model.Type || "")),
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
  refreshModelDropdown();
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
  const minBits = requiredBits(state.model);
  byId("profile-note").innerHTML = `<strong>${escapeHtml(state.task)}</strong> requires ≥ ${minBits} bits for this model. Selected quant is ${state.bits} bits${state.bits < minBits ? ' <span class="warning">— below the task-safe floor.</span>' : ' — meets the task-safe floor.'} ${state.offload === "Y" && state.hostCpu ? `Host offload pool: ${decimal.format(hostOffloadGb())} GB from ${escapeHtml(state.hostCpu["CPU / platform"])}.` : ""} Response at target: ~${decimal.format(state.answerTokens / state.target)} s after prefill; prompt/context use ${decimal.format(state.promptTokens / state.context * 100)}%.`;
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
  const kv = kvPerTokenGb(model, state.kvBits) * state.context * state.users;
  const denominator = Number(model["Active params (B)"]) * state.bits / 8 + kv;
  const quantityScale = cpu ? 1 : 1 + 0.8 * (state.gpuQuantity - 1);
  return efficiency(cpu ? "" : hardware.Vendor, cpu, hardware) * bandwidth * quantityScale / denominator / state.users;
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

function refreshModelDropdown() {
  const select = byId("model-search");
  const rows = state.data.models.filter((row) => modelMatchesFilters(row, "model"));
  select.replaceChildren();
  for (const row of rows) select.add(new Option(`${row.Model} · ${row.Vendor} · ${decimal.format(row["Total params (B)"])}B`, row.Model));
  select.value = rows.some((row) => row.Model === state.model?.Model) ? state.model.Model : rows[0]?.Model || "";
  if (state.model && !rows.some((row) => row.Model === state.model.Model) && rows.length) state.model = rows[0];
}

function showSearch() { refreshModelDropdown(); }
function closeSearch() {}

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
  host.innerHTML = `<div class="parity-metrics"><div><strong>${signed(a.memory_delta_pct.median)}</strong><small>Median 8K memory delta</small></div><div><strong>${signed(a.gpu_decode_delta_pct.median)}</strong><small>Median 8K GPU decode delta</small></div><div><strong>${signed(b.gpu_decode_delta_pct.median)}</strong><small>Median 32K GPU decode delta</small></div><div><strong>${signed(b.cpu_decode_delta_pct.median)}</strong><small>Median 32K CPU decode delta</small></div></div><p class="parity-caption">Positive means the app is higher. At 8K, median deltas are ${signed(a.memory_delta_pct.median)} memory, ${signed(a.gpu_decode_delta_pct.median)} GPU decode, and ${signed(a.cpu_decode_delta_pct.median)} CPU decode. At 32K they are ${signed(b.memory_delta_pct.median)} memory, ${signed(b.gpu_decode_delta_pct.median)} GPU decode, and ${signed(b.cpu_decode_delta_pct.median)} CPU decode. The remaining speed gap comes from Excel charging the whole KV allocation into the per-token traffic denominator.</p>`;
}

function hardwarePool() {
  const gpus = state.data.gpus.map((row) => ({ raw: row, kind: "gpu", name: row.Accelerator }));
  const cpus = state.data.cpus.map((row) => ({ row, raw: row, kind: "cpu", name: row["CPU / platform"] }));
  return [...gpus, ...cpus];
}

function hardwareMemory(item) {
  if (item.kind === "gpu") {
    const raw = state.hardwareVram > 0 ? state.hardwareVram : Number(item.raw["Memory (GB)"]);
    const gpuPool = raw * (1 - state.hardwareMemoryHeadroom / 100) * state.hardwareGpuQuantity;
    const hostPool = state.hardwareOffload === "Y" && state.hardwareCpu ? Math.max(0, (state.hardwareRam || Number(state.hardwareCpu["Max RAM (GB)"])) - 8) : 0;
    return gpuPool + hostPool;
  }
  const raw = state.hardwareRam > 0 ? state.hardwareRam : Number(item.raw["Max RAM (GB)"]);
  return raw - Math.min(8, raw * 0.25);
}

function hardwareBandwidth(item) {
  const raw = Number(item.kind === "gpu" ? item.raw["Bandwidth (GB/s)"] : item.raw["Memory BW (GB/s)"]);
  const quantity = item.kind === "gpu" ? 1 + 0.8 * (state.hardwareGpuQuantity - 1) : 1;
  return raw * quantity * state.hardwareBandwidthUtil / 100;
}

function hardwareNeed(model) {
  const weights = Number(model["Total params (B)"]) * state.hardwareBits / 8 * 1.03;
  const kv = kvPerTokenGb(model, state.hardwareKvBits) * state.hardwareContext * state.hardwareUsers;
  return { weights, kv, total: weights + kv + weights * 0.07 };
}

function hardwareSpeed(item, model) {
  const denominator = Number(model["Active params (B)"]) * state.hardwareBits / 8;
  const oldRuntime = state.runtime;
  state.runtime = state.hardwareRuntime;
  const eff = efficiency(item.kind === "gpu" ? item.raw.Vendor : "", item.kind === "cpu", item.raw);
  state.runtime = oldRuntime;
  return eff * hardwareBandwidth(item) / denominator / state.hardwareUsers;
}

function setHardware(item) {
  state.hardware = item.raw;
  state.hardwareKind = item.kind;
  if (item.kind === "gpu") {
    state.hardwareGpu = item.raw;
    byId("hardware-gpu-select").value = item.name;
    state.hardwareVram = Number(item.raw["Memory (GB)"]);
    byId("hardware-vram-input").value = state.hardwareVram;
  } else {
    state.hardwareCpu = item.raw;
    byId("hardware-cpu-select").value = item.name;
    state.hardwareRam = Number(item.raw["Max RAM (GB)"]);
    byId("hardware-ram-input").value = state.hardwareRam;
  }
  byId("hardware-search").value = item.name;
  closeHardwareSearch();
  renderHardwareView();
}

function refreshHardwareDropdown() {
  const select = byId("hardware-search");
  const previous = select.value;
  const rows = hardwarePool().filter((item) => !state.intelOnly || isIntelHardware(item.raw, item.kind));
  select.replaceChildren();
  for (const item of rows) {
    const label = item.kind === "gpu" ? `GPU · ${gpuCatalogLabel(item.raw)}` : `CPU · ${cpuCatalogLabel(item.raw)}`;
    select.add(new Option(label, item.name));
  }
  select.value = rows.some((item) => item.name === previous) ? previous : rows.some((item) => item.name === state.hardware?.Accelerator || item.name === state.hardware?.["CPU / platform"]) ? (state.hardware.Accelerator || state.hardware["CPU / platform"]) : rows[0]?.name || "";
}
function showHardwareSearch() { refreshHardwareDropdown(); }
function closeHardwareSearch() {}

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
    .map((row) => ({ ...row, safeQuant: state.hardwareBits >= requiredBits(row.model, state.hardwareTask) && (state.hardwareTask !== "Vision (image + text)" || /VLM|Vision/i.test(row.model.Type || "")), fits: row.need <= capacity && state.hardwareBits >= requiredBits(row.model, state.hardwareTask) && (state.hardwareTask !== "Vision (image + text)" || /VLM|Vision/i.test(row.model.Type || "")) }))
    .filter((row) => filter === "all" || (filter === "fits" ? row.fits : row.fits && row.safeQuant && row.speed >= state.hardwareTarget))
    .sort((a, b) => Number(b.fits) - Number(a.fits) || b.speed - a.speed);
}

function renderHardwareView() {
  if (!state.hardware) return;
  const item = currentHardwareItem();
  const capacity = hardwareMemory(item);
  const rows = compatibleModels(item);
  const fits = compatibleModels(item).filter((row) => row.fits).length;
  const source = item.raw["Source URL"] || "";
  const taskMinimum = taskRules[state.hardwareTask] || taskRules["General chat / Q&A"];
  byId("hardware-profile-note").textContent = `${state.hardwareTask}: model rows below the task quant floor are not shown as meeting the speed target. ${state.hardwareOffload === "Y" && state.hardwareCpu ? `CPU offload may use ${decimal.format(Number(byId("hardware-ram-input").value || state.hardwareCpu["Max RAM (GB)"]))} GB host RAM.` : "CPU offload disabled."} Runtime ${state.hardwareRuntime}; KV cache ${state.hardwareKvBits}-bit; prompt ${byId("hardware-prompt-input").value}; answer ${byId("hardware-answer-input").value}.`;
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
    if (!row.safeQuant) article.classList.add("quality-warning");
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
  refreshCatalogDropdowns();
  const defaultHardware = hardwarePool().find((item) => item.name === "Intel Arc Pro B60") || hardwarePool()[0];
  const defaultCpu = state.data.cpus.find((row) => row["CPU / platform"] === "Core Ultra 9 285K") || state.data.cpus[0];
  state.hardwareCpu = defaultCpu;
  state.hardwareRam = Number(defaultCpu["Max RAM (GB)"]);
  byId("hardware-cpu-select").value = defaultCpu["CPU / platform"];
  state.hardwareRam = selectClosest(byId("hardware-ram-input"), state.hardwareRam);
  setHardware(defaultHardware);
  refreshModelDropdown();
}

function databaseRows() {
  const name = byId("database-select").value;
  const rows = state.data[name];
  if (!state.intelOnly || !["gpus", "cpus"].includes(name)) return rows;
  return rows.filter((row) => isIntelHardware(row, name === "gpus" ? "gpu" : "cpu"));
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
  const intelGpuNames = new Set(state.data.gpus.filter((row) => row.Vendor === "Intel").map((row) => row.Accelerator));
  const intelCpuNames = new Set(state.data.cpus.filter((row) => isIntelHardware(row, "cpu")).map((row) => row["CPU / platform"]));
  let rows = state.data.evidence.filter((row) => {
    const matchesIntel = !state.intelOnly || !["GPUs", "CPUs"].includes(row.Sheet) || (row.Sheet === "GPUs" ? intelGpuNames.has(row.Item) : intelCpuNames.has(row.Item));
    const matchesQuery = (!query || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query))) && matchesIntel;
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
  byId("model-search").addEventListener("change", (event) => {
    const model = state.data.models.find((row) => row.Model === event.target.value);
    if (model) setModel(model);
  });
  byId("quant-select").addEventListener("change", (event) => {
    state.bits = Number(event.target.selectedOptions[0].dataset.bits);
    render();
  });
  byId("context-select").addEventListener("change", (event) => { state.context = Number(event.target.value); render(); });
  byId("user-select").addEventListener("change", (event) => { state.users = Number(event.target.value); render(); });
  byId("target-input").addEventListener("input", (event) => { state.target = Number(event.target.value) || 20; renderRows(); });
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
    refreshCatalogDropdowns();
    if (state.intelOnly) {
      state.hostCpu = state.data.cpus.find((row) => isIntelHardware(row, "cpu"));
      state.preferredGpu = state.data.gpus.find((row) => row.Vendor === "Intel");
      state.hardwareCpu = state.data.cpus.find((row) => isIntelHardware(row, "cpu"));
      if (!isIntelHardware(state.hardware, state.hardwareKind)) setHardware(hardwarePool().find((item) => isIntelHardware(item.raw, item.kind) && item.kind === "gpu"));
      byId("host-cpu-select").value = state.hostCpu["CPU / platform"];
      byId("hardware-cpu-select").value = state.hardwareCpu["CPU / platform"];
      byId("preferred-gpu-select").value = state.preferredGpu.Accelerator;
      state.hostRam = Number(state.hostCpu["Max RAM (GB)"]);
      state.hardwareRam = Number(state.hardwareCpu["Max RAM (GB)"]);
      state.deviceVram = Number(state.preferredGpu["Memory (GB)"]);
      byId("host-ram-input").value = String(state.hostRam);
      byId("hardware-ram-input").value = String(state.hardwareRam);
      byId("device-vram-input").value = String(state.deviceVram);
    } else {
      state.hostCpu = null;
      state.preferredGpu = null;
      byId("host-cpu-select").value = "";
      byId("preferred-gpu-select").value = "";
    }
    refreshModelDropdown();
    render();
    if (state.page === "hardware") renderHardwareView();
    if (state.page === "database") renderDatabase();
    if (state.page === "evidence") renderEvidence();
  });
  for (const id of ["model-vendor-filter", "model-type-filter"]) byId(id).addEventListener("change", () => {
    refreshModelDropdown();
    const next = state.data.models.find((row) => row.Model === byId("model-search").value);
    if (next) state.model = next;
    render();
  });
  byId("vendor-select").addEventListener("change", (event) => {
    if (state.intelOnly && event.target.value !== "Intel") event.target.value = "Intel";
    state.vendor = event.target.value;
    render();
  });
  const profileInputs = [
    ["runtime-select", "runtime", render], ["task-select", "task", render], ["kv-bits-select", "kvBits", render],
    ["prompt-length-input", "promptTokens", render], ["answer-length-input", "answerTokens", render],
    ["offload-select", "offload", render], ["gpu-quantity-input", "gpuQuantity", render],
    ["cpu-power-input", "maxCpuPower", render], ["pcie-width-select", "pcieWidth", render], ["device-vram-input", "deviceVram", render]
  ];
  for (const [id, key, callback] of profileInputs) byId(id).addEventListener("input", (event) => { state[key] = typeof state[key] === "number" ? Number(event.target.value) : event.target.value; callback(); });
  byId("host-cpu-select").addEventListener("change", (event) => { state.hostCpu = state.data.cpus.find((row) => row["CPU / platform"] === event.target.value) || null; if (state.hostCpu) state.hostRam = selectClosest(byId("host-ram-input"), state.hostCpu["Max RAM (GB)"]); render(); });
  byId("host-ram-input").addEventListener("input", (event) => { state.hostRam = Number(event.target.value); render(); });
  byId("preferred-gpu-select").addEventListener("change", (event) => { state.preferredGpu = state.data.gpus.find((row) => row.Accelerator === event.target.value) || null; if (state.preferredGpu) { state.deviceVram = Number(state.preferredGpu["Memory (GB)"]); byId("device-vram-input").value = state.deviceVram; } render(); });
  byId("min-memory-filter").addEventListener("change", (event) => { state.minMemory = Number(event.target.value); render(); });
  byId("max-power-filter").addEventListener("change", (event) => { state.maxPower = Number(event.target.value); render(); });

  byId("hardware-search").addEventListener("change", (event) => {
    const item = hardwarePool().find((entry) => entry.name === event.target.value);
    if (item) setHardware(item);
  });
  byId("hardware-quant-select").addEventListener("change", (event) => { state.hardwareBits = Number(event.target.selectedOptions[0].dataset.bits); renderHardwareView(); });
  byId("hardware-context-select").addEventListener("change", (event) => { state.hardwareContext = Number(event.target.value); renderHardwareView(); });
  byId("hardware-user-select").addEventListener("change", (event) => { state.hardwareUsers = Number(event.target.value); renderHardwareView(); });
  byId("hardware-target-input").addEventListener("input", (event) => { state.hardwareTarget = Number(event.target.value) || 20; renderHardwareView(); });
  byId("hardware-cpu-select").addEventListener("change", (event) => { state.hardwareCpu = state.data.cpus.find((row) => row["CPU / platform"] === event.target.value) || null; if (state.hardwareCpu) state.hardwareRam = selectClosest(byId("hardware-ram-input"), state.hardwareCpu["Max RAM (GB)"]); renderHardwareView(); });
  byId("hardware-gpu-select").addEventListener("change", (event) => { state.hardwareGpu = state.data.gpus.find((row) => row.Accelerator === event.target.value) || null; if (state.hardwareGpu) { state.hardware = state.hardwareGpu; state.hardwareKind = "gpu"; state.hardwareVram = Number(state.hardwareGpu["Memory (GB)"]); byId("hardware-vram-input").value = state.hardwareVram; byId("hardware-search").value = state.hardwareGpu.Accelerator; } renderHardwareView(); });
  byId("hardware-ram-input").addEventListener("input", (event) => { state.hardwareRam = Number(event.target.value); renderHardwareView(); });
  byId("hardware-vram-input").addEventListener("input", (event) => { state.hardwareVram = Number(event.target.value); renderHardwareView(); });
  byId("hardware-gpu-quantity-input").addEventListener("input", (event) => { state.hardwareGpuQuantity = Number(event.target.value); renderHardwareView(); });
  byId("hardware-runtime-select").addEventListener("change", (event) => { state.hardwareRuntime = event.target.value; renderHardwareView(); });
  byId("hardware-task-select").addEventListener("change", (event) => { state.hardwareTask = event.target.value; renderHardwareView(); });
  byId("hardware-kv-bits-select").addEventListener("change", (event) => { state.hardwareKvBits = Number(event.target.value); renderHardwareView(); });
  byId("hardware-offload-select").addEventListener("change", (event) => { state.hardwareOffload = event.target.value; renderHardwareView(); });
  byId("hardware-prompt-input").addEventListener("input", renderHardwareView);
  byId("hardware-answer-input").addEventListener("input", renderHardwareView);
  byId("memory-headroom-input").addEventListener("input", (event) => { state.hardwareMemoryHeadroom = Number(event.target.value); renderHardwareView(); });
  byId("bandwidth-util-input").addEventListener("input", (event) => { state.hardwareBandwidthUtil = Number(event.target.value); renderHardwareView(); });
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
