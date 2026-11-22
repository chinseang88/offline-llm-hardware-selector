const state = {
  data: null,
  privateReference: null,
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
  hardwareSort: "speed-desc",
  modelSort: "speed-desc",
  databaseSort: { table: "models", column: "", direction: "asc" },
  evidenceSort: { column: "Checked", direction: "desc" },
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
  evidenceSheet: "all",
  evidenceLimit: 300,
  databaseLimit: 500,
  modelQuery: "",
  hardwareModelQuery: "",
  runtime: "llama.cpp",
  task: "General chat / Q&A",
  kvBits: 8,
  promptTokens: 2048,
  answerTokens: 256,
  offload: "N",
  gpuQuantity: 1,
  hostCpu: null,
  hostRam: 0,
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

// All sizing maths lives in formulas.js (window.ModelFit); these wrappers only supply the page state.
const MF = window.ModelFit;
const DATABASE_PAGE = 500;
const EVIDENCE_PAGE = 300;
let efficiencyTable = {};
let taskRules = MF.buildTaskRules([]);

function kvPerTokenGb(model, bits = state.kvBits) {
  return MF.kvPerTokenGb(model, bits);
}

function memoryNeed(model) {
  return MF.memoryNeed(model, { bits: state.bits, kvBits: state.kvBits, context: state.context, users: state.users });
}

function excelMemoryNeed(model, bits = state.bits, context = state.context, users = state.users, kvBits = state.kvBits) {
  return MF.excelMemoryNeed(model, { bits, context, users, kvBits });
}

function gpuEfficiency(gpu, runtime = state.runtime) {
  return MF.gpuRuntimeEfficiency(efficiencyTable, runtime, gpu);
}

function efficiency(hardware, cpu = false, runtime = state.runtime) {
  return cpu ? MF.cpuRuntimeEfficiency(efficiencyTable, runtime, hardware) : gpuEfficiency(hardware, runtime).efficiency;
}

function runtimeNames() {
  const names = (state.data?.libraries || []).map((row) => row.Library).filter(Boolean);
  return names.length ? names : ["llama.cpp"];
}

function requiredBits(model, task = state.task) {
  return MF.requiredBits(model, MF.taskRule(taskRules, task));
}

function archSupport(model, runtime = state.runtime) {
  const row = (state.data.archSupport || []).find((item) => item["Arch key"] === model["Runtime arch"]);
  return row ? String(row[runtime] ?? "") : "";
}

// Returns the first reason a model cannot run, or an empty string when it can.
function fitBlocker(model, { bits, task, runtime, supported, memoryOk }) {
  if (!supported) return `${runtime} not supported`;
  if (archSupport(model, runtime) === "N") return "Runtime cannot load model";
  const rule = MF.taskRule(taskRules, task);
  if (rule.vlm && !MF.isVlm(model)) return "Needs a vision model";
  if (bits < MF.requiredBits(model, rule)) return "Below task quant floor";
  return memoryOk ? "" : "Memory shortfall";
}

function tokPerSecond(hardware, model, cpu = false) {
  return MF.decodeTokPerSecond({
    activeParamsB: model["Active params (B)"], bits: state.bits, users: state.users,
    efficiency: efficiency(hardware, cpu),
    bandwidth: cpu ? hardware["Memory BW (GB/s)"] : hardware["Bandwidth (GB/s)"],
    quantity: cpu ? 1 : state.gpuQuantity,
  });
}

function splitSpeed(model, bits, users, need, gpuPool, gpuRate, cpu, runtime) {
  return MF.splitDecodeTokPerSecond({
    perTokenGb: Number(model["Active params (B)"]) * bits / 8,
    gpuShare: need > 0 ? gpuPool / need : 1,
    gpuRate,
    cpuRate: cpu ? efficiency(cpu, true, runtime) * Number(cpu["Memory BW (GB/s)"]) : 0,
    users,
  });
}

function gpuUsable(hardware) {
  const isPreferred = state.preferredGpu && hardware.Accelerator === state.preferredGpu.Accelerator;
  const memory = isPreferred && state.deviceVram > 0 ? state.deviceVram : Number(hardware["Memory (GB)"]);
  return MF.gpuUsableGb(memory, state.gpuQuantity);
}

function cpuUsable(hardware) {
  const isHost = state.hostCpu && hardware["CPU / platform"] === state.hostCpu["CPU / platform"];
  const memory = isHost && state.hostRam > 0 ? state.hostRam : Number(hardware["Max RAM (GB)"]);
  return MF.cpuUsableGb(memory);
}

// Host RAM that overflow layers may use: installed RAM (capped at the platform limit) minus 8 GB for the OS.
function offloadPoolGb(gpu, cpu, ram, enabled, runtime) {
  if (enabled !== "Y" || !cpu || !MF.offloadSupported(runtime) || MF.isUnifiedMemory(gpu)) return 0;
  return MF.hostOffloadGb(ram > 0 ? ram : cpu["Max RAM (GB)"], cpu["Max RAM (GB)"]);
}

function renderWarnings(id, messages) {
  const host = byId(id);
  host.hidden = !messages.length;
  host.innerHTML = messages.map((message) => `<p class="${message.blocker ? "blocker" : ""}">${escapeHtml(message.text)}</p>`).join("");
}

function platformWarnings({ gpu, cpu, ram, quantity, offload, runtime }) {
  const messages = [];
  if (offload === "Y" && !MF.offloadSupported(runtime)) messages.push({ text: `${runtime} cannot split a model between GPU and CPU, so CPU offload is ignored. llama.cpp, Ollama, LM Studio and IPEX-LLM can.` });
  if (offload === "Y" && !cpu) messages.push({ text: "CPU offload needs a host CPU. Select one to size the overflow pool." });
  if (offload === "Y" && gpu && MF.isUnifiedMemory(gpu)) messages.push({ text: `${gpuDisplayName(gpu.Accelerator)} shares one memory pool between CPU and GPU, so there is nothing to offload to.` });
  if (cpu && ram > Number(cpu["Max RAM (GB)"])) messages.push({ text: `${ram} GB installed RAM is above the ${cpu["Max RAM (GB)"]} GB limit of ${cpu["CPU / platform"]}. Offload is capped at the platform limit.` });
  if (cpu && !(gpu && MF.isUnifiedMemory(gpu)) && Number(cpu["CPU PCIe lanes"]) < 8 * quantity) messages.push({ text: `${cpu["CPU / platform"]} has ${cpu["CPU PCIe lanes"]} CPU PCIe lanes, fewer than x8 for each of ${quantity} GPU${quantity > 1 ? "s" : ""}.` });
  return messages;
}

function isIntelHardware(row, kind = "gpu") {
  if (kind === "gpu") return row.Vendor === "Intel";
  const name = String(row["CPU / platform"] || "");
  return name.startsWith("NVL") || /^(Intel |Core Ultra|Core i[3579]|Xeon )|\bCore Ultra\b|\bXeon\b/.test(name);
}

function cpuGeneration(name) {
  if (name.startsWith("NVL-BGA")) return "Nova Lake-BGA";
  if (name.startsWith("NVL-S")) return "Nova Lake-S";
  if (name.startsWith("NVL-E")) return "Nova Lake-E Xeon";
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

function gpuDisplayName(name) {
  return name.startsWith("Intel CRI") ? name.replace("Intel CRI", "Intel Crescent Island") : name;
}

function gpuCatalogLabel(row) {
  return `${gpuDisplayName(row.Accelerator)} · ${row["Memory (GB)"]} GB · ${row["Bandwidth (GB/s)"]} GB/s · ${row["TDP / board power (W)"]}W · ${row.Interface} · ${row.Architecture}`;
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

function setSkuMemoryOption(select, value, kind) {
  const exact = Number(value);
  if (!Number.isFinite(exact) || exact <= 0) return 0;
  const key = String(exact);
  if (![...select.options].some((option) => option.value === key)) {
    select.add(new Option(`${exact} GB · ${kind} SKU`, key));
  }
  select.value = key;
  return exact;
}

function syncHostCpuMemory(cpu) {
  if (!cpu) return;
  state.hostRam = setSkuMemoryOption(byId("host-ram-input"), cpu["Max RAM (GB)"], "CPU");
}

function syncPreferredGpuMemory(gpu) {
  if (!gpu) return;
  state.deviceVram = setSkuMemoryOption(byId("device-vram-input"), gpu["Memory (GB)"], "GPU");
}

function syncHardwareCpuMemory(cpu) {
  if (!cpu) return;
  state.hardwareRam = setSkuMemoryOption(byId("hardware-ram-input"), cpu["Max RAM (GB)"], "CPU");
}

function syncHardwareGpuMemory(gpu) {
  if (!gpu) return;
  state.hardwareVram = setSkuMemoryOption(byId("hardware-vram-input"), gpu["Memory (GB)"], "GPU");
}

function cpuPowerText(cpu) {
  if (!cpu) return "Select a host CPU";
  return `${cpu["Base / default power (W)"]} W base · ${cpu["Max turbo / max power (W)"]} W max`;
}

function gpuPowerText(gpu, quantity = 1) {
  if (!gpu) return "Select a GPU";
  const perCard = Number(gpu["TDP / board power (W)"]);
  return quantity > 1 ? `${perCard} W each · ${perCard * quantity} W total (${quantity}×)` : `${perCard} W TDP`;
}

function updatePowerReadouts() {
  byId("cpu-power-output").textContent = cpuPowerText(state.hostCpu);
  byId("gpu-power-output").textContent = gpuPowerText(state.preferredGpu, state.gpuQuantity);
  byId("hardware-cpu-power-output").textContent = cpuPowerText(state.hardwareCpu);
  byId("hardware-gpu-power-output").textContent = gpuPowerText(state.hardwareGpu, state.hardwareGpuQuantity);
}

function selectGlobalGpu(gpu, { renderHardware = true } = {}) {
  if (!gpu) {
    state.preferredGpu = null;
    byId("preferred-gpu-select").value = "";
    byId("device-vram-input").value = "0";
    state.deviceVram = 0;
    return;
  }
  state.preferredGpu = gpu;
  state.hardwareGpu = gpu;
  state.hardware = gpu;
  state.hardwareKind = "gpu";
  byId("preferred-gpu-select").value = gpu.Accelerator;
  byId("hardware-gpu-select").value = gpu.Accelerator;
  syncPreferredGpuMemory(gpu);
  syncHardwareGpuMemory(gpu);
  if (renderHardware) renderHardwareView();
}

function selectGlobalCpu(cpu, { renderHardware = true } = {}) {
  state.hostCpu = cpu || null;
  byId("host-cpu-select").value = cpu ? cpu["CPU / platform"] : "";
  // "Auto / none" only clears the model page's host; the hardware page always keeps a CPU selected.
  if (cpu) {
    state.hardwareCpu = cpu;
    byId("hardware-cpu-select").value = cpu["CPU / platform"];
    syncHostCpuMemory(cpu);
    syncHardwareCpuMemory(cpu);
  } else {
    state.hostRam = 0;
    byId("host-ram-input").value = "0";
    if (state.hardwareCpu && byId("hardware-cpu-select").value !== state.hardwareCpu["CPU / platform"]) {
      const options = [...byId("hardware-cpu-select").options].map((option) => option.value);
      if (!options.includes(state.hardwareCpu["CPU / platform"])) state.hardwareCpu = state.data.cpus.find((row) => row["CPU / platform"] === options[0]) || state.hardwareCpu;
      byId("hardware-cpu-select").value = state.hardwareCpu["CPU / platform"];
      syncHardwareCpuMemory(state.hardwareCpu);
    }
  }
  if (renderHardware) renderHardwareView();
}

function refreshCatalogDropdowns() {
  rebuildCatalogSelect("host-cpu-select", state.data.cpus, "CPU / platform", cpuCatalogLabel, { text: "Auto / none", value: "" }, "cpu");
  rebuildCatalogSelect("hardware-cpu-select", state.data.cpus, "CPU / platform", cpuCatalogLabel, null, "cpu");
  rebuildCatalogSelect("preferred-gpu-select", state.data.gpus, "Accelerator", gpuCatalogLabel, { text: "No preference — show all GPUs", value: "" }, "gpu");
  rebuildCatalogSelect("hardware-gpu-select", state.data.gpus, "Accelerator", gpuCatalogLabel, null, "gpu");
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

function hostOffloadGb(gpu = null) {
  return offloadPoolGb(gpu, state.hostCpu, state.hostRam, state.offload, state.runtime);
}

function hardwareRows() {
  const need = memoryNeed(state.model).total;
  const blockerFor = (supported, memoryOk) => fitBlocker(state.model, { bits: state.bits, task: state.task, runtime: state.runtime, supported, memoryOk });
  if (state.view === "cpus") {
    return state.data.cpus
      .filter((row) => cleanNumber(row["Max RAM (GB)"]) && cleanNumber(row["Memory BW (GB/s)"]))
      .filter((row) => !state.intelOnly || isIntelHardware(row, "cpu"))
      .filter((row) => state.pcieWidth === "Auto" || Number(row["CPU PCIe lanes"]) >= Number((state.pcieWidth.match(/\d+/) || [0])[0]) * state.gpuQuantity)
      .map((row) => {
        const memory = cpuUsable(row);
        const blocker = blockerFor(efficiency(row, true) > 0, memory >= need);
        return {
          raw: row,
          name: row["CPU / platform"],
          vendor: row.Class,
          subtitle: `${cpuGeneration(row["CPU / platform"]) || row.Class} · ${row.Cores} cores · ${row.Memory}`,
          memory,
          physicalMemory: Number(row["Max RAM (GB)"]),
          bandwidth: Number(row["Memory BW (GB/s)"]),
          power: row["Max turbo / max power (W)"],
          speed: tokPerSecond(row, state.model, true),
          fits: !blocker,
          blocker,
          gpuShare: null,
          evidence: row["Evidence status"],
          source: row["Source URL"],
          kind: "cpu",
        };
      });
  }
  return state.data.gpus
    .filter((row) => cleanNumber(row["Memory (GB)"]) && cleanNumber(row["Bandwidth (GB/s)"]))
    .filter((row) => !state.intelOnly || isIntelHardware(row, "gpu"))
    .filter((row) => state.vendor === "all" || row.Vendor === state.vendor)
    .filter((row) => !state.preferredGpu || row.Accelerator === state.preferredGpu.Accelerator)
    .filter((row) => !state.minMemory || Number(row["Memory (GB)"]) >= state.minMemory)
    .filter((row) => !state.maxPower || Number(row["TDP / board power (W)"]) <= state.maxPower)
    .filter((row) => state.legacy || String(row["Used / legacy?"]).toLowerCase() !== "yes")
    .map((row) => {
      const backend = gpuEfficiency(row);
      const gpuPool = gpuUsable(row);
      const hostPool = hostOffloadGb(row);
      const offloaded = hostPool > 0 && need > gpuPool;
      const blocker = blockerFor(backend.efficiency > 0, gpuPool + hostPool >= need);
      const gpuRate = backend.efficiency * Number(row["Bandwidth (GB/s)"]) * MF.quantityScale(state.gpuQuantity);
      return {
        raw: row,
        name: gpuDisplayName(row.Accelerator),
        value: row.Accelerator,
        vendor: row.Vendor,
        subtitle: `${row.Architecture} · ${row.Interface} · ${state.runtime}${backend.backend === "vulkan" ? " via Vulkan fallback" : ""}`,
        memory: gpuPool + hostPool,
        physicalMemory: Number(row["Memory (GB)"]),
        bandwidth: Number(row["Bandwidth (GB/s)"]),
        power: row["TDP / board power (W)"],
        speed: offloaded ? splitSpeed(state.model, state.bits, state.users, need, gpuPool, gpuRate, state.hostCpu, state.runtime) : tokPerSecond(row, state.model, false),
        fits: !blocker,
        blocker,
        gpuShare: offloaded ? gpuPool / need : null,
        evidence: row["Evidence status"],
        source: row["Source URL"],
        kind: "gpu",
      };
    });
}

function evidenceScore(text) {
  const value = String(text || "");
  const verified = Number((value.match(/(\d+) verified/) || [0, 0])[1]);
  const corrected = Number((value.match(/(\d+) corrected/) || [0, 0])[1]);
  const unsourced = Number((value.match(/(\d+) estimate-or-unsourced/) || [0, 0])[1]);
  return verified * 100 + corrected * 25 - unsourced * 10;
}

function compareNumber(a, b, direction = "desc") {
  const av = Number.isFinite(Number(a)) ? Number(a) : -Infinity;
  const bv = Number.isFinite(Number(b)) ? Number(b) : -Infinity;
  return direction === "asc" ? av - bv : bv - av;
}

function hardwareComparator(a, b) {
  const mode = state.hardwareSort;
  let primary = 0;
  if (mode === "memory-desc") primary = compareNumber(a.memory, b.memory, "desc");
  else if (mode === "memory-asc") primary = compareNumber(a.memory, b.memory, "asc");
  else if (mode === "bandwidth-desc") primary = compareNumber(a.bandwidth, b.bandwidth, "desc");
  else if (mode === "bandwidth-asc") primary = compareNumber(a.bandwidth, b.bandwidth, "asc");
  else if (mode === "power-asc") primary = compareNumber(a.power, b.power, "asc");
  else if (mode === "power-desc") primary = compareNumber(a.power, b.power, "desc");
  else if (mode === "name-asc") primary = String(a.name).localeCompare(String(b.name));
  else if (mode === "name-desc") primary = String(b.name).localeCompare(String(a.name));
  else if (mode === "vendor-asc") primary = String(a.vendor).localeCompare(String(b.vendor));
  else if (mode === "evidence-desc") primary = evidenceScore(b.evidence) - evidenceScore(a.evidence);
  else if (mode === "evidence-asc") primary = evidenceScore(a.evidence) - evidenceScore(b.evidence);
  else if (mode === "speed-asc") primary = compareNumber(a.speed, b.speed, "asc");
  else primary = compareNumber(a.speed, b.speed, "desc");
  if (primary) return primary;
  return String(a.name).localeCompare(String(b.name));
}

function filteredRows() {
  return hardwareRows()
    .filter((row) => state.filter === "all" || (state.filter === "fits" ? row.fits : row.fits && row.speed >= state.target))
    .sort(hardwareComparator);
}

function setModel(model) {
  state.model = model;
  refreshModelDropdown();
  render();
}

function modelWarnings() {
  const model = state.model;
  const messages = [];
  const limit = Number(model["Max context"]);
  if (limit && state.context > limit) messages.push({ text: `${state.context.toLocaleString()} tokens of context is above this model's published limit of ${limit.toLocaleString()}. Memory is sized for the context you selected.` });
  if (state.promptTokens + state.answerTokens > state.context) messages.push({ text: `Prompt (${state.promptTokens.toLocaleString()}) plus answer (${state.answerTokens.toLocaleString()}) tokens do not fit in a ${state.context.toLocaleString()}-token context.` });
  const rule = MF.taskRule(taskRules, state.task);
  if (rule.vlm && !MF.isVlm(model)) messages.push({ text: `${state.task} needs a vision-language model. ${model.Model} is listed as ${model.Type || "text only"}.`, blocker: true });
  const floor = requiredBits(model);
  if (state.bits < floor) messages.push({ text: `The selected quantization (${state.bits} bits) is below the ${floor}-bit floor for ${state.task}. ${rule.why}`.trim(), blocker: true });
  const support = archSupport(model);
  if (support === "N") messages.push({ text: `${state.runtime} cannot load the ${model["Runtime arch"]} architecture, so no hardware is shown as fitting. Choose another runtime.`, blocker: true });
  if (support === "P") messages.push({ text: `${state.runtime} supports the ${model["Runtime arch"]} architecture only partially. Check the runtime's release notes before committing.` });
  messages.push(...platformWarnings({ gpu: state.preferredGpu, cpu: state.hostCpu, ram: state.hostRam, quantity: state.gpuQuantity, offload: state.offload, runtime: state.runtime }));
  return messages;
}

function renderSummary() {
  const need = memoryNeed(state.model);
  const gpuFits = hardwareRowsForView("gpus").filter((row) => row.fits).length;
  const cpuFits = hardwareRowsForView("cpus").filter((row) => row.fits).length;
  byId("model-summary-kicker").textContent = ["Selected model", state.model.Type, state.model.Year].filter(Boolean).join(" · ");
  byId("model-summary-title").textContent = state.model.Model;
  byId("total-params").textContent = `${decimal.format(state.model["Total params (B)"])}B`;
  byId("active-params").textContent = `${decimal.format(state.model["Active params (B)"])}B`;
  byId("need-memory").textContent = `${decimal.format(need.total)} GB`;
  byId("fits-count").textContent = state.view === "cpus" ? cpuFits : gpuFits;
  byId("fits-label").textContent = state.view === "cpus" ? "CPU fits" : "GPU fits";
  byId("gpu-count").textContent = gpuFits;
  byId("cpu-count").textContent = cpuFits;
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
  renderWarnings("model-warnings", modelWarnings());
}

function hardwareRowsForView(view) {
  const current = state.view;
  state.view = view;
  const rows = hardwareRows();
  state.view = current;
  return rows;
}

// Workbook "1 Plan a Build" speed: the whole KV allocation is added to the bytes read per token.
function excelTokPerSecond(hardware, model, cpu = false) {
  return MF.excelDecodeTokPerSecond(model, {
    bits: state.bits, kvBits: state.kvBits, context: state.context, users: state.users,
    efficiency: efficiency(hardware, cpu),
    bandwidth: cpu ? hardware["Memory BW (GB/s)"] : hardware["Bandwidth (GB/s)"],
    quantity: cpu ? 1 : state.gpuQuantity,
  });
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

function updateColumnSortButtons() {
  for (const button of $$(".column-sort")) {
    const mode = button.dataset.sortGroup === "hardware" ? state.hardwareSort : state.modelSort;
    const [prefix, direction] = mode.split("-");
    const active = prefix === button.dataset.sortPrefix;
    button.classList.toggle("active", active);
    button.querySelector("b").textContent = active ? (direction === "asc" ? "↑" : "↓") : "↕";
    button.setAttribute("aria-sort", active ? (direction === "asc" ? "ascending" : "descending") : "none");
  }
}

function renderRows() {
  const host = byId("results");
  const rows = filteredRows();
  updateColumnSortButtons();
  byId("hardware-sort-output").textContent = `Sorted by ${byId("hardware-sort").selectedOptions[0].textContent} · ${rows.length} shown`;
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
    const speedLabel = row.fits ? (row.speed >= state.target ? "Meets target" : "Fits, slower") : row.blocker;
    const excelSpeed = excelTokPerSecond(row.raw, state.model, row.kind === "cpu");
    const speedDelta = excelSpeed ? (row.speed / excelSpeed - 1) * 100 : 0;
    const evidenceLevel = evidenceClass(row.evidence);
    item.innerHTML = `
      <div class="hardware-cell">
        <span class="vendor-chip">${vendorInitial(row.vendor, row.name)}</span>
        <div class="hardware-name"><strong>${escapeHtml(row.name)}</strong><small>${escapeHtml(row.subtitle)}</small></div>
      </div>
      <div class="fit-cell">
        <div class="fit-meta"><b>${decimal.format(row.memory)} GB usable</b><span>${row.memory >= need ? `${decimal.format(ratio * 100)}%` : `short ${decimal.format(need - row.memory)} GB`}</span></div>
        <div class="fit-bar ${row.memory >= need ? "" : "over"}"><span style="width:${bar}%"></span></div>
        <div class="fit-copy">Need <strong>${decimal.format(need)} GB</strong> from ${decimal.format(row.physicalMemory)} GB ${row.kind === "gpu" ? "device memory" : "system RAM"}${row.gpuShare === null ? "" : ` · ${Math.round(row.gpuShare * 100)}% on GPU, rest on host RAM`}</div>
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
  updatePowerReadouts();
  renderSummary();
  renderRows();
}

function modelSearchText(row) {
  return `${row.Model} ${row.Vendor} ${row.Type} ${row["Runtime arch"]} ${row.License}`.toLowerCase();
}

function refreshModelDropdown() {
  const select = byId("model-search");
  const filtered = state.data.models.filter((row) => modelMatchesFilters(row, "model"));
  let rows = state.modelQuery ? filtered.filter((row) => modelSearchText(row).includes(state.modelQuery)) : filtered;
  // Typing in the quick filter narrows the list but never swaps the model already selected.
  if (state.model && filtered.includes(state.model) && !rows.includes(state.model)) rows = [state.model, ...rows];
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

  if (kind === "model") {
    const tabs = document.createElement("div");
    tabs.className = "knowledge-tabs";
    tabs.innerHTML = `<button class="active" data-knowledge-tab="overview">Model 360</button><button data-knowledge-tab="latest">Latest models</button><button data-knowledge-tab="evidence">Evidence</button><button data-knowledge-tab="learning">Learning</button>`;
    content.append(tabs);
  }

  const intro = document.createElement("p");
  intro.className = "knowledge-view";
  intro.dataset.knowledgeView = "overview";
  intro.textContent = facts.length
    ? `${facts.length} source checks are recorded for this row. Status reflects the public evidence log, not a claim of benchmark validation.`
    : "No parameter-level evidence records were found for this public row.";
  content.append(intro);

  if (kind === "model") {
    const profile = document.createElement("section");
    profile.className = "model-360 knowledge-view";
    profile.dataset.knowledgeView = "overview";
    const need = memoryNeed(row);
    profile.innerHTML = `<p class="section-label">Model 360</p><div class="model-360-grid"><div><small>Parameters</small><strong>${decimal.format(row["Total params (B)"])}B</strong><span>${decimal.format(row["Active params (B)"])}B active</span></div><div><small>Architecture</small><strong>${escapeHtml(row["Runtime arch"] || row.Type || "—")}</strong><span>${row.Layers || "—"} layers · ${row["KV heads"] || "—"} KV heads</span></div><div><small>Current workload</small><strong>${decimal.format(need.total)} GB</strong><span>${state.bits}-bit · ${state.context.toLocaleString()} context</span></div><div><small>Published context</small><strong>${Number(row["Max context"]) ? Number(row["Max context"]).toLocaleString() : "—"}</strong><span>${escapeHtml(row.License || "License pending")}</span></div></div><p class="model-education">Parameter count determines checkpoint size; active parameters drive compute traffic; context, KV heads, and head dimension determine KV-cache growth. Architecture and runtime support decide which inference engines can execute the model.</p>`;
    content.append(profile);

    const reference = referenceForModel(row);
    if (reference) {
      const lineage = document.createElement("section");
      lineage.className = "reference-note knowledge-view";
      lineage.dataset.knowledgeView = "latest";
      lineage.innerHTML = `<p class="section-label">Reference catalog · public evidence</p><div><strong>Tier ${escapeHtml(reference.Tier)}${reference["Reference rank"] && reference["Reference rank"] !== "-" ? ` · ranked #${escapeHtml(reference["Reference rank"])}` : " · unranked"}</strong><p>${escapeHtml(reference.Family)} · ${escapeHtml(reference["Architecture"])} · ${escapeHtml(reference["Total params (B)"])}B total / ${escapeHtml(reference["Active params (B)"])}B active</p><p>Released ${escapeHtml(reference["Release date"] || "date pending")} · ${escapeHtml(reference.License)} · ${escapeHtml(reference["Context (max)"])} context</p></div><div class="drawer-links">${reference["Model card URL"] ? `<a href="${escapeHtml(reference["Model card URL"])}" target="_blank" rel="noreferrer">Publisher model card ↗</a>` : ""}${reference["GGUF repository URL"] ? `<a href="${escapeHtml(reference["GGUF repository URL"])}" target="_blank" rel="noreferrer">GGUF repository ↗</a>` : ""}${reference["License source URL"] ? `<a href="${escapeHtml(reference["License source URL"])}" target="_blank" rel="noreferrer">License source ↗</a>` : ""}</div><small>Measured quality, timing, memory, and platform results remain withheld because the reference export is redacted/internal.</small>`;
      content.append(lineage);
    }

    const privateRuns = privateRunsForModel(row);
    if (privateRuns.length) {
      const platformById = new Map(state.privateReference.platforms.map((platform) => [platform.platform_id, platform]));
      const servingByRun = new Map(state.privateReference.serving.map((run) => [run.run_id, run]));
      const benchmark = document.createElement("section");
      benchmark.className = "private-benchmark knowledge-view";
      benchmark.dataset.knowledgeView = "overview";
      const bestRuns = [...privateRuns].sort((a, b) => Number(b.decode_tps || 0) - Number(a.decode_tps || 0)).slice(0, 8);
      benchmark.innerHTML = `<p class="section-label">Measured performance · local private evidence</p><div class="private-benchmark-summary"><strong>${privateRuns.length} measured runs</strong><span>${new Set(privateRuns.map((run) => run.platform_id)).size} platforms · quality, speed, timing, and provenance retained</span></div><div class="private-run-list">${bestRuns.map((run) => {
        const platform = platformById.get(run.platform_id) || {};
        const serving = servingByRun.get(run.run_id) || {};
        const repo = serving.model_repo ? `https://huggingface.co/${escapeHtml(serving.model_repo)}${serving.model_revision ? `/tree/${escapeHtml(serving.model_revision)}` : ""}` : "";
        return `<article><div><strong>${escapeHtml(run.model_id)}</strong><small>${escapeHtml(platform.device || platform.name || run.platform_id)} · ${escapeHtml(serving.engine || platform.runtime || "runtime pending")}</small></div><div><b>${decimal.format(Number(run.decode_tps))}</b><small>decode tok/s</small></div><div><b>${decimal.format(Number(run.overall), 3)}</b><small>overall quality</small></div><p>Measured on ${escapeHtml(platform.name || run.platform_id)} · ${escapeHtml(run.quant)} · context ${escapeHtml(serving.ctx || "not recorded")} · judge ${escapeHtml(run.judge_name || "not recorded")}</p>${repo ? `<a href="${repo}" target="_blank" rel="noreferrer">Exact HF revision ↗</a>` : ""}</article>`;
      }).join("")}</div><small>This complete measured data is loaded from the ignored local reference database. It is not published because the source export is redacted/private.</small>`;
      content.append(benchmark);
    } else {
      const benchmarkFacts = facts.filter((fact) => /benchmark|tok\/s|throughput|latency|measured|ttft|decode/i.test(`${fact.Parameter} ${fact.Note} ${fact["Source type"]}`));
      const benchmark = document.createElement("section");
      benchmark.className = "benchmark-note knowledge-view";
      benchmark.dataset.knowledgeView = "overview";
      benchmark.innerHTML = `<p class="section-label">Measured performance</p><strong>${benchmarkFacts.length ? `${benchmarkFacts.length} measured-performance records` : "No public measured benchmark is available"}</strong><p>${benchmarkFacts.length ? "Open each record below for the measured machine, runtime, precision, workload, and source citation." : "The tok/s values in the selector are bandwidth-model estimates, not measurements. The reference site has private/redacted measurements; they are available locally only and omitted from the public build."}</p>`;
      content.append(benchmark);
    }
  }

  const sourceUrls = urlList(row["Source URL"]);
  if (sourceUrls.length) {
    const links = document.createElement("div");
    links.className = `drawer-links${kind === "model" ? " knowledge-view" : ""}`;
    if (kind === "model") links.dataset.knowledgeView = "evidence";
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
    list.className = "fact-list knowledge-view";
    list.dataset.knowledgeView = "evidence";
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
      if (fact["Source URL"]) {
        const source = document.createElement("a");
        source.className = "fact-source";
        source.href = fact["Source URL"];
        source.target = "_blank";
        source.rel = "noreferrer";
        source.textContent = `${fact["Source type"] || sourceLabel(fact["Source URL"])} · checked ${fact.Checked || "date not recorded"} ↗`;
        body.append(source);
      }
      block.append(heading, body);
      list.append(block);
    }
    content.append(list);
  }

  if (kind === "model") {
    const latest = document.createElement("section");
    latest.className = "latest-learning knowledge-view";
    latest.dataset.knowledgeView = "latest";
    latest.innerHTML = `<p class="section-label">Verified latest releases</p><div class="latest-model-grid"><article><span>QWEN</span><strong>Qwen3.8 Flash-Next 180B-A6B</strong><p>Official Qwen 4-architecture preview · 6B active · 262K context</p></article><article><span>GOOGLE</span><strong>Gemma 4 E2B / E4B</strong><p>Official multimodal edge models · 128K context · effective parameter sizing</p></article><article><span>HUGGING FACE BIO</span><strong>Carbon-X · 3B · 8B · 500M</strong><p>DNA + language models · Apache 2.0 · genomics-specific runtime notes</p></article></div><small>“Gemini 4 Hydrogen” and “Gemini 4 Carbon” are excluded: the public uploads are results cards or one-parameter toy repos, not verified Google model weights.</small>`;
    content.append(latest);

    const learning = document.createElement("section");
    learning.className = "learning-panel knowledge-view";
    learning.dataset.knowledgeView = "learning";
    learning.innerHTML = `<p class="section-label">How to learn a model</p><ol><li><b>Identify the artifact.</b><span>Publisher, model family, release date, exact Hugging Face repository and revision.</span></li><li><b>Separate total from active.</b><span>Total parameters drive memory; active parameters drive decode compute.</span></li><li><b>Calculate the full fit.</b><span>Weights + KV cache + runtime overhead, not file size alone.</span></li><li><b>Check the workload.</b><span>General chat, coding, vision, agents and DNA models need different minimums.</span></li><li><b>Verify measurement provenance.</b><span>Measured speed needs machine, runtime/build, quantization, context, workload, date and source.</span></li></ol>`;
    content.append(learning);

    content.querySelectorAll("[data-knowledge-tab]").forEach((button) => button.addEventListener("click", () => {
      content.querySelectorAll("[data-knowledge-tab]").forEach((item) => item.classList.toggle("active", item === button));
      const view = button.dataset.knowledgeTab;
      content.querySelectorAll("[data-knowledge-view]").forEach((item) => item.hidden = item.dataset.knowledgeView !== view);
    }));
    content.querySelectorAll("[data-knowledge-view]").forEach((item) => item.hidden = item.dataset.knowledgeView !== "overview");
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

function hardwareGpuPool(item) {
  const raw = state.hardwareVram > 0 ? state.hardwareVram : Number(item.raw["Memory (GB)"]);
  return MF.gpuUsableGb(raw, state.hardwareGpuQuantity, state.hardwareMemoryHeadroom);
}

function hardwareHostPool(item) {
  return item.kind === "gpu" ? offloadPoolGb(item.raw, state.hardwareCpu, state.hardwareRam, state.hardwareOffload, state.hardwareRuntime) : 0;
}

function hardwareMemory(item) {
  if (item.kind === "gpu") return hardwareGpuPool(item) + hardwareHostPool(item);
  const raw = state.hardwareRam > 0 ? state.hardwareRam : Number(item.raw["Max RAM (GB)"]);
  return MF.cpuUsableGb(raw);
}

function hardwareBandwidth(item) {
  const raw = Number(item.kind === "gpu" ? item.raw["Bandwidth (GB/s)"] : item.raw["Memory BW (GB/s)"]);
  const quantity = item.kind === "gpu" ? MF.quantityScale(state.hardwareGpuQuantity) : 1;
  return raw * quantity * state.hardwareBandwidthUtil / 100;
}

function hardwareNeed(model) {
  return MF.memoryNeed(model, { bits: state.hardwareBits, kvBits: state.hardwareKvBits, context: state.hardwareContext, users: state.hardwareUsers });
}

function hardwareSpeed(item, model, need) {
  const cpu = item.kind === "cpu";
  const rate = efficiency(item.raw, cpu, state.hardwareRuntime) * hardwareBandwidth(item);
  const perToken = Number(model["Active params (B)"]) * state.hardwareBits / 8;
  if (!perToken) return 0;
  if (!cpu) {
    const gpuPool = hardwareGpuPool(item);
    if (hardwareHostPool(item) > 0 && need > gpuPool) return splitSpeed(model, state.hardwareBits, state.hardwareUsers, need, gpuPool, rate, state.hardwareCpu, state.hardwareRuntime);
  }
  return rate / perToken / state.hardwareUsers;
}

function currentHardwareItem() {
  return { raw: state.hardware, kind: state.hardwareKind, name: state.hardwareKind === "gpu" ? gpuDisplayName(state.hardware.Accelerator) : state.hardware["CPU / platform"] };
}

function modelComparator(a, b) {
  const mode = state.modelSort;
  let primary = 0;
  if (mode === "total-desc") primary = compareNumber(a.model["Total params (B)"], b.model["Total params (B)"], "desc");
  else if (mode === "active-desc") primary = compareNumber(a.model["Active params (B)"], b.model["Active params (B)"], "desc");
  else if (mode === "need-asc") primary = compareNumber(a.need, b.need, "asc");
  else if (mode === "need-desc") primary = compareNumber(a.need, b.need, "desc");
  else if (mode === "name-asc") primary = String(a.model.Model).localeCompare(String(b.model.Model));
  else if (mode === "name-desc") primary = String(b.model.Model).localeCompare(String(a.model.Model));
  else if (mode === "vendor-asc") primary = String(a.model.Vendor).localeCompare(String(b.model.Vendor));
  else if (mode === "architecture-asc") primary = String(a.model["Runtime arch"]).localeCompare(String(b.model["Runtime arch"]));
  else if (mode === "architecture-desc") primary = String(b.model["Runtime arch"]).localeCompare(String(a.model["Runtime arch"]));
  else if (mode === "evidence-desc") primary = evidenceScore(b.model["Evidence status"]) - evidenceScore(a.model["Evidence status"]);
  else if (mode === "evidence-asc") primary = evidenceScore(a.model["Evidence status"]) - evidenceScore(b.model["Evidence status"]);
  else if (mode === "speed-asc") primary = compareNumber(a.speed, b.speed, "asc");
  else primary = compareNumber(a.speed, b.speed, "desc");
  if (primary) return primary;
  return String(a.model.Model).localeCompare(String(b.model.Model));
}

function compatibleModels(item, { ignoreShowFilter = false } = {}) {
  const capacity = hardwareMemory(item);
  const filter = ignoreShowFilter ? "all" : byId("hardware-model-filter").value;
  const supported = efficiency(item.raw, item.kind === "cpu", state.hardwareRuntime) > 0;
  const rule = MF.taskRule(taskRules, state.hardwareTask);
  return state.data.models
    .filter((model) => model["Active params (B)"] != null && model.Layers != null && model["KV heads"] != null && model["Head dim"] != null)
    .filter((model) => modelMatchesFilters(model, "hardware"))
    .filter((model) => !state.hardwareModelQuery || modelSearchText(model).includes(state.hardwareModelQuery))
    .map((model) => {
      const need = hardwareNeed(model).total;
      const blocker = fitBlocker(model, { bits: state.hardwareBits, task: state.hardwareTask, runtime: state.hardwareRuntime, supported, memoryOk: need <= capacity });
      return { model, need, speed: hardwareSpeed(item, model, need), safeQuant: MF.taskAllows(model, rule, state.hardwareBits), fits: !blocker, blocker };
    })
    .filter((row) => filter === "all" || (filter === "fits" ? row.fits : row.fits && row.speed >= state.hardwareTarget))
    .sort(modelComparator);
}

function hardwareWarnings(item) {
  const messages = [];
  const cpu = item.kind === "cpu";
  if (!cpu) {
    const backend = gpuEfficiency(item.raw, state.hardwareRuntime);
    if (!backend.efficiency) messages.push({ text: `${state.hardwareRuntime} does not run on ${item.name}, so no model is shown as fitting. Choose another runtime.`, blocker: true });
    else if (backend.backend === "vulkan") messages.push({ text: `${state.hardwareRuntime} has no native backend for ${item.name}. Speed uses the slower Vulkan fallback (${Math.round(backend.efficiency * 100)}% bandwidth efficiency).` });
    if (MF.isUnifiedMemory(item.raw) && state.hardwareGpuQuantity > 1) messages.push({ text: `${item.name} is a single unified-memory system. A quantity above 1 assumes separate machines and is not a supported cluster estimate.` });
  } else if (!efficiency(item.raw, true, state.hardwareRuntime)) {
    messages.push({ text: `${state.hardwareRuntime} does not run on ${item.name}, so no model is shown as fitting. Choose another runtime.`, blocker: true });
  }
  messages.push(...platformWarnings({ gpu: cpu ? null : item.raw, cpu: state.hardwareCpu, ram: state.hardwareRam, quantity: cpu ? 0 : state.hardwareGpuQuantity, offload: cpu ? "N" : state.hardwareOffload, runtime: state.hardwareRuntime }));
  return messages;
}

function renderHardwareView() {
  if (!state.hardware) return;
  updatePowerReadouts();
  const item = currentHardwareItem();
  const capacity = hardwareMemory(item);
  const rows = compatibleModels(item);
  updateColumnSortButtons();
  const fits = compatibleModels(item, { ignoreShowFilter: true }).filter((row) => row.fits).length;
  byId("model-sort-output").textContent = `Sorted by ${byId("model-sort").selectedOptions[0].textContent} · ${rows.length} shown`;
  const hostPool = hardwareHostPool(item);
  byId("hardware-profile-note").textContent = `${state.hardwareTask}: models below the task quantization floor are not counted as fitting. ${hostPool > 0 ? `CPU offload adds ${decimal.format(hostPool)} GB of host RAM; overflow layers run at CPU speed.` : "CPU offload is off."} Runtime ${state.hardwareRuntime}; KV cache ${state.hardwareKvBits}-bit; prompt ${byId("hardware-prompt-input").value}; answer ${byId("hardware-answer-input").value}.`;
  renderWarnings("hardware-warnings", hardwareWarnings(item));
  const summary = byId("hardware-summary");
  const planning = String(item.raw["Evidence status"] || "").includes("Pre-release planning");
  summary.innerHTML = `<div><p class="section-label">Selected ${item.kind === "gpu" ? "accelerator" : "CPU platform"}${planning ? " · PRE-RELEASE PLANNING" : ""}</p><h2>${escapeHtml(item.name)}</h2><button class="text-button" id="show-hardware-evidence"><i data-lucide="badge-check"></i><span>Fact check</span></button></div><div class="metric"><span>${decimal.format(capacity)} GB</span><small>Usable memory</small></div><div class="metric"><span>${decimal.format(hardwareBandwidth(item))} GB/s</span><small>Bandwidth</small></div><div class="metric"><span>${fits}</span><small>Models that fit</small></div>`;
  summary.querySelector("#show-hardware-evidence").addEventListener("click", () => openEvidence(item.raw, item.kind));
  const host = byId("hardware-model-results");
  host.replaceChildren();
  if (!rows.length) {
    host.innerHTML = `<div class="empty-state"><h3>No model matches these filters.</h3><p>Lower the quantization, shorten the context, or select a different model family.</p></div>`;
    return;
  }
  for (const row of rows) {
    const model = row.model;
    const ratio = row.need / capacity * 100;
    const fast = row.fits && row.speed >= state.hardwareTarget;
    const article = document.createElement("article");
    article.className = "result-row model-result-row";
    if (!row.safeQuant) article.classList.add("quality-warning");
    const excelNeed = excelMemoryNeed(model, state.hardwareBits, state.hardwareContext, state.hardwareUsers, state.hardwareKvBits).total;
    article.innerHTML = `<div class="hardware-cell"><span class="vendor-chip">${escapeHtml(String(model.Vendor || "LLM").slice(0, 2).toUpperCase())}</span><div class="hardware-name"><strong>${escapeHtml(model.Model)}</strong><small>${escapeHtml(model.Type)} · ${decimal.format(model["Total params (B)"])}B parameters</small></div></div><div class="fit-cell"><div class="fit-meta"><b>${decimal.format(row.need)} GB needed</b><span>${row.need <= capacity ? `${decimal.format(ratio)}%` : `short ${decimal.format(row.need - capacity)} GB`}</span></div><div class="fit-bar ${row.need <= capacity ? "" : "over"}"><span style="width:${Math.min(100, ratio)}%"></span></div><div class="fit-copy">${decimal.format(hardwareNeed(model).weights)} GB weights · ${decimal.format(hardwareNeed(model).kv)} GB KV · Excel ${decimal.format(excelNeed)} GB (${(row.need / excelNeed - 1) * 100 >= 0 ? "+" : ""}${decimal.format((row.need / excelNeed - 1) * 100)}%)</div></div><div class="speed-cell"><strong class="${fast ? "fast" : "slow"}">${decimal.format(row.speed)}</strong><small>tok/s est. · ${fast ? "Meets target" : row.fits ? "Fits, slower" : row.blocker}</small></div><div class="hardware-arch"><code>${escapeHtml(model["Runtime arch"] || "—")}</code><small>${model.Layers}L × ${model["KV heads"]} KV × ${model["Head dim"]}d</small></div><div class="evidence-cell"><span class="status-pill ${evidenceClass(model["Evidence status"])}">${evidenceShort(model["Evidence status"])}</span><button class="row-detail" aria-label="Know more about ${escapeHtml(model.Model)}"><i data-lucide="arrow-up-right"></i><span>Know more</span></button></div>`;
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
  if (page === "pipeline") window.ModelFitPipeline.render();
  if (page === "database") renderDatabase();
  if (page === "evidence") renderEvidence();
  if (page === "knowledge") renderKnowledge();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function populateFilterOptions() {
  const tasks = Object.keys(taskRules);
  for (const [ids, names, fallback] of [[["runtime-select", "hardware-runtime-select"], runtimeNames(), "llama.cpp"], [["task-select", "hardware-task-select"], tasks, tasks[0]]]) {
    for (const id of ids) {
      const select = byId(id);
      const previous = select.value;
      select.replaceChildren(...names.map((name) => new Option(name, name)));
      select.value = names.includes(previous) ? previous : fallback;
    }
  }
  for (const sheet of [...new Set(state.data.evidence.map((row) => row.Sheet).filter(Boolean))].sort()) byId("evidence-sheet-filter").add(new Option(sheet, sheet));
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
  const defaultGpu = state.data.gpus.find((row) => row.Accelerator === "Intel Arc Pro B60") || state.data.gpus[0];
  const defaultCpu = state.data.cpus.find((row) => row["CPU / platform"] === "Core Ultra 9 285K") || state.data.cpus[0];
  state.hardwareCpu = defaultCpu;
  state.hardwareRam = Number(defaultCpu["Max RAM (GB)"]);
  byId("hardware-cpu-select").value = defaultCpu["CPU / platform"];
  syncHardwareCpuMemory(defaultCpu);
  if (defaultGpu) selectGlobalGpu(defaultGpu, { renderHardware: false });
  else {
    state.hardware = defaultCpu;
    state.hardwareKind = "cpu";
  }
  selectGlobalGpu(null, { renderHardware: false });
  refreshModelDropdown();
}

function normalizeReferenceName(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9.]+/g, "");
}

function referenceForModel(row) {
  const target = normalizeReferenceName(row.Model);
  return state.data.referenceModels.find((reference) => normalizeReferenceName(reference.Model) === target || normalizeReferenceName(reference.Family) === target) || null;
}

function privateRunsForModel(row) {
  if (!state.privateReference) return [];
  const reference = referenceForModel(row);
  const targets = new Set([normalizeReferenceName(row.Model)]);
  if (reference) {
    targets.add(normalizeReferenceName(reference.Model));
    targets.add(normalizeReferenceName(reference.Family));
  }
  return state.privateReference.quality.filter((run) => targets.has(normalizeReferenceName(run.family)));
}

function preparePrivateMeasurements(payload) {
  const platforms = new Map(payload.platforms.map((row) => [row.platform_id, row]));
  const serving = new Map(payload.serving.map((row) => [row.run_id, row]));
  const measurements = payload.quality.map((run) => {
    const platform = platforms.get(run.platform_id) || {};
    const provenance = serving.get(run.run_id) || {};
    const repo = provenance.model_repo || "";
    const revision = provenance.model_revision || "";
    return {
      "Model": run.family,
      "Quantization": run.quant,
      "Measured on": platform.name || run.platform_id,
      "Device": platform.device || "",
      "Runtime": provenance.engine || platform.runtime || "",
      "Context": provenance.ctx || "",
      "Quality overall": run.overall,
      "Quality CI low": run.overall_ci_lo,
      "Quality CI high": run.overall_ci_hi,
      "GSM8K": run.gsm8k,
      "HumanEval": run.humaneval,
      "Alpaca": run.alpaca,
      "MeetingBank": run.meetingbank,
      "Prefill tok/s": run.prefill_tps,
      "Decode tok/s": run.decode_tps,
      "Run wall (s)": run.mean_wall_s,
      "Judge": run.judge_name,
      "Model file": provenance.model_file || run.model_id,
      "Model revision": revision,
      "Model SHA256": provenance.model_sha256 || "",
      "Source URL": repo && revision ? `https://huggingface.co/${repo}/tree/${revision}` : repo ? `https://huggingface.co/${repo}` : "",
      "Evidence status": "MEASURED-INTERNAL / redacted export / public citation pending",
    };
  });
  state.data.privateMeasurements = measurements;
  state.data.privateFindings = payload.findings.map((row) => ({
    "Finding": row.title, "Area": row.area, "Severity": row.severity, "Detail": row.detail,
    "Evidence": row.evidence, "Source URL": "https://reimagined-adventure-l6zp8gm.pages.github.io/#briefing",
    "Evidence status": "MEASURED-INTERNAL / private reference finding",
  }));
  state.data.privateIssues = payload.issues.map((row) => ({
    "Issue": row.issue, "Category": row.category, "Severity": row.severity, "Run/source": row.source,
    "Source URL": "https://reimagined-adventure-l6zp8gm.pages.github.io/#lineage",
    "Evidence status": "MEASURED-INTERNAL / audit issue",
  }));
  return measurements;
}

// Newest catalog rows: latest release year, most recently synced first, only rows with source-verified values.
function latestModels(limit = 20) {
  const year = Math.max(...state.data.models.map((row) => Number(row.Year) || 0));
  return state.data.models
    .map((row, index) => ({ row, index }))
    .filter(({ row }) => Number(row.Year) === year && evidenceScore(row["Evidence status"]) > 0)
    .sort((a, b) => String(b.row["Last synced"] || "").localeCompare(String(a.row["Last synced"] || "")) || b.index - a.index)
    .slice(0, limit)
    .map(({ row }) => row);
}

function renderKnowledge() {
  const host = byId("knowledge-latest-models");
  host.replaceChildren();
  const rows = latestModels();
  for (const row of rows) {
    const article = document.createElement("article");
    article.className = "knowledge-model";
    article.innerHTML = `<span class="knowledge-vendor">${escapeHtml(row.Vendor)}</span><h3>${escapeHtml(row.Model)}</h3><p>${escapeHtml(row.Type)} · ${decimal.format(row["Total params (B)"])}B total · ${decimal.format(row["Active params (B)"])}B active</p><div><span>${decimal.format(row["Max context"])} context</span><span>${escapeHtml(row.License)}</span></div><button class="text-button"><i data-lucide="arrow-up-right"></i><span>Know more</span></button>`;
    article.querySelector("button").addEventListener("click", () => openEvidence(row, "model"));
    host.append(article);
  }
  byId("knowledge-latest-count").textContent = `${rows.length} newest verified rows`;
  byId("knowledge-evidence-count").textContent = `${state.data.evidence.length} checks`;
  iconize();
}

function databaseRows() {
  const name = byId("database-select").value;
  const rows = state.data[name];
  if (!state.intelOnly || !["gpus", "cpus"].includes(name)) return rows;
  return rows.filter((row) => isIntelHardware(row, name === "gpus" ? "gpu" : "cpu"));
}

function comparableValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const numeric = String(value).replace(/,/g, "");
  if (/^-?\d+(?:\.\d+)?$/.test(numeric)) return Number(numeric);
  const date = Date.parse(value);
  if (!Number.isNaN(date) && /\d{4}-\d{2}-\d{2}/.test(String(value))) return date;
  return String(value).toLocaleLowerCase();
}

function compareValues(a, b, direction = "asc") {
  const av = comparableValue(a);
  const bv = comparableValue(b);
  if (av === null && bv === null) return 0;
  if (av === null) return 1;
  if (bv === null) return -1;
  const result = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv), undefined, { numeric: true });
  return direction === "asc" ? result : -result;
}

function sortedDatabaseRows() {
  const name = byId("database-select").value;
  const query = byId("database-search").value.toLowerCase().trim();
  const sort = state.databaseSort;
  const rows = databaseRows().filter((row) => !query || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query)));
  if (sort.table === name && sort.column) rows.sort((a, b) => compareValues(a[sort.column], b[sort.column], sort.direction));
  return rows;
}

function renderLoadMore(id, shown, total, step, onMore) {
  const wrap = byId(id);
  wrap.hidden = total <= shown;
  wrap.replaceChildren();
  if (wrap.hidden) return;
  const label = document.createElement("span");
  label.textContent = `Showing ${shown.toLocaleString()} of ${total.toLocaleString()}`;
  const button = document.createElement("button");
  button.className = "secondary-button";
  button.type = "button";
  button.textContent = `Show ${Math.min(step, total - shown).toLocaleString()} more`;
  button.addEventListener("click", onMore);
  wrap.append(label, button);
}

function renderDatabase() {
  const name = byId("database-select").value;
  const table = byId("database-table");
  const sort = state.databaseSort;
  table.replaceChildren();
  const sortedRows = sortedDatabaseRows();
  const visible = sortedRows.slice(0, state.databaseLimit);
  renderLoadMore("database-more-wrap", visible.length, sortedRows.length, DATABASE_PAGE, () => { state.databaseLimit += DATABASE_PAGE; renderDatabase(); });
  const allCols = databaseRows().length ? Object.keys(databaseRows()[0]).filter((key) => !key.startsWith("_")) : [];
  const sortSelect = byId("database-sort-column");
  const currentOptions = [...sortSelect.options].map((option) => option.value);
  if (currentOptions.join("|") !== allCols.join("|")) {
    sortSelect.replaceChildren(new Option("Original order", ""), ...allCols.map((column) => new Option(column, column)));
  }
  sortSelect.value = sort.table === name ? sort.column : "";
  byId("database-sort-direction").dataset.direction = sort.direction;
  byId("database-sort-direction").querySelector("span").textContent = sort.direction === "asc" ? "Ascending" : "Descending";
  byId("database-sort-direction").disabled = !sort.column;
  byId("database-sort-output").textContent = sort.table === name && sort.column ? `Sorted by ${sort.column} · ${sort.direction === "asc" ? "ascending" : "descending"} · ${sortedRows.length} matching rows` : `Original order · ${sortedRows.length} matching rows · choose a sort field`;
  if (!visible.length) {
    table.innerHTML = "<tbody><tr><td>No matching rows.</td></tr></tbody>";
    return;
  }
  const cols = Object.keys(visible[0]).filter((key) => !key.startsWith("_"));
  const head = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const col of cols) {
    const th = document.createElement("th");
    const button = document.createElement("button");
    button.className = "table-sort";
    const active = sort.table === name && sort.column === col;
    button.innerHTML = `<span>${escapeHtml(col)}</span><span class="sort-mark">${active ? (sort.direction === "asc" ? "↑" : "↓") : "↕"}</span>`;
    button.title = `Sort by ${col}`;
    th.setAttribute("aria-sort", active ? (sort.direction === "asc" ? "ascending" : "descending") : "none");
    button.addEventListener("click", () => {
      state.databaseSort = { table: name, column: col, direction: active && sort.direction === "asc" ? "desc" : "asc" };
      renderDatabase();
    });
    th.append(button);
    hr.append(th);
  }
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

function sortedEvidenceRows() {
  const query = byId("evidence-search").value.toLowerCase().trim();
  const intelGpuNames = new Set(state.data.gpus.filter((row) => row.Vendor === "Intel").map((row) => row.Accelerator));
  const intelCpuNames = new Set(state.data.cpus.filter((row) => isIntelHardware(row, "cpu")).map((row) => row["CPU / platform"]));
  let rows = state.data.evidence.filter((row) => {
    const matchesIntel = !state.intelOnly || !["GPUs", "CPUs"].includes(row.Sheet) || (row.Sheet === "GPUs" ? intelGpuNames.has(row.Item) : intelCpuNames.has(row.Item));
    const matchesQuery = (!query || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query))) && matchesIntel;
    const matchesStatus = state.evidenceStatus === "all" || (state.evidenceStatus === "OTHER" ? !["VERIFIED", "MISMATCH", "DERIVED/ESTIMATE"].includes(row.Status) : row.Status === state.evidenceStatus);
    const matchesSheet = state.evidenceSheet === "all" || row.Sheet === state.evidenceSheet;
    return matchesQuery && matchesStatus && matchesSheet;
  });
  rows.sort((a, b) => compareValues(a[state.evidenceSort.column], b[state.evidenceSort.column], state.evidenceSort.direction));
  return rows;
}

function renderEvidence() {
  const sortedRows = sortedEvidenceRows();
  byId("evidence-count").textContent = `${sortedRows.length} matching checks · ${state.data.evidence.length} total`;
  byId("evidence-sort-output").textContent = `Sorted by ${byId("evidence-sort-column").selectedOptions[0].textContent} · ${state.evidenceSort.direction === "asc" ? "ascending" : "descending"} · ${sortedRows.length} matching checks`;
  const rows = sortedRows.slice(0, state.evidenceLimit);
  renderLoadMore("evidence-more-wrap", rows.length, sortedRows.length, EVIDENCE_PAGE, () => { state.evidenceLimit += EVIDENCE_PAGE; renderEvidence(); });
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

function downloadCsv(name, rows) {
  const cols = rows.length ? Object.keys(rows[0]).filter((key) => !key.startsWith("_")) : [];
  const csvCell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [cols.map(csvCell).join(","), ...rows.map((row) => cols.map((col) => csvCell(row[col])).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a"); link.href = url; link.download = `${name}.csv`; link.click(); URL.revokeObjectURL(url);
}

function renderVerification() {
  const host = byId("verification-content");
  const report = state.verification;
  if (!report || !report.summary) { host.textContent = "No check report is published with this build."; return; }
  const total = report.summary.pass + report.summary.fail;
  host.innerHTML = `<div class="parity-metrics"><div><strong>${report.summary.pass} / ${total}</strong><small>Checks passed</small></div><div><strong>${escapeHtml(String(report.generated || "").slice(0, 10))}</strong><small>Last run · ${escapeHtml(report.mode || "offline")}</small></div></div><ul class="verification-list">${(report.groups || []).map((group) => `<li><span>${escapeHtml(group.name)}</span><b class="${group.fail ? "fail" : ""}">${group.pass} passed${group.fail ? ` · ${group.fail} failed` : ""}${group.warn ? ` · ${group.warn} to review` : ""}</b></li>`).join("")}</ul><p class="parity-caption">Repeat every check with <code>python scripts/reverify_all.py</code>. Open this page with <code>?selftest=1</code> to exercise each control in both themes.</p>`;
}

async function copySummary() {
  const need = memoryNeed(state.model);
  const gpuFits = hardwareRowsForView("gpus").filter((row) => row.fits).length;
  const cpuFits = hardwareRowsForView("cpus").filter((row) => row.fits).length;
  const text = [
    `Model Fit sizing: ${state.model.Model}`,
    `Parameters: ${decimal.format(state.model["Total params (B)"])}B total, ${decimal.format(state.model["Active params (B)"])}B active`,
    `Workload: ${byId("quant-select").value}, ${state.context.toLocaleString()} context, ${state.users} user${state.users > 1 ? "s" : ""}, KV ${state.kvBits}-bit, ${state.runtime}, ${state.task}`,
    `Memory needed: ${decimal.format(need.total)} GB (weights ${decimal.format(need.weights)} + KV cache ${decimal.format(need.kv)} + runtime ${decimal.format(need.runtime)})`,
    `Hardware that fits: ${gpuFits} GPUs, ${cpuFits} CPUs`,
    "Speeds are bandwidth estimates, not benchmarks.",
    location.href.split("?")[0],
  ].join("\n");
  const label = byId("copy-summary").querySelector("span");
  try {
    await navigator.clipboard.writeText(text);
    label.textContent = "Copied";
  } catch {
    label.textContent = "Copy blocked";
  }
  setTimeout(() => { label.textContent = "Copy"; }, 1600);
  return text;
}

function resetModelFilters() {
  state.modelQuery = "";
  state.minMemory = 0;
  state.maxPower = 0;
  byId("model-quick-filter").value = "";
  byId("model-vendor-filter").value = "all";
  byId("model-type-filter").value = "all";
  byId("min-memory-filter").value = "0";
  byId("max-power-filter").value = "0";
  refreshModelDropdown();
  render();
}

function resetHardwareFilters() {
  state.hardwareModelQuery = "";
  byId("hardware-model-quick-filter").value = "";
  byId("hardware-model-vendor-filter").value = "all";
  byId("hardware-model-type-filter").value = "all";
  byId("hardware-model-filter").value = "fits";
  renderHardwareView();
}

function filterGlossary(query) {
  const list = $(".knowledge-glossary dl");
  let shown = 0;
  for (const entry of list.querySelectorAll(":scope > div:not(.empty-state)")) {
    entry.hidden = Boolean(query) && !entry.textContent.toLowerCase().includes(query);
    if (!entry.hidden) shown += 1;
  }
  let empty = list.querySelector(".empty-state");
  if (!shown && !empty) {
    empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = "<p>No glossary term matches. Clear the filter to see every term.</p>";
    list.append(empty);
  }
  if (empty) empty.hidden = shown > 0;
}

function bindEvents() {
  byId("model-quick-filter").addEventListener("input", (event) => { state.modelQuery = event.target.value.toLowerCase().trim(); refreshModelDropdown(); });
  byId("hardware-model-quick-filter").addEventListener("input", (event) => { state.hardwareModelQuery = event.target.value.toLowerCase().trim(); renderHardwareView(); });
  byId("reset-model-filters").addEventListener("click", resetModelFilters);
  byId("reset-hardware-filters").addEventListener("click", resetHardwareFilters);
  byId("copy-summary").addEventListener("click", copySummary);
  byId("glossary-search").addEventListener("input", (event) => filterGlossary(event.target.value.toLowerCase().trim()));
  byId("evidence-sheet-filter").addEventListener("change", (event) => { state.evidenceSheet = event.target.value; state.evidenceLimit = EVIDENCE_PAGE; renderEvidence(); });
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
  byId("hardware-sort").addEventListener("change", (event) => { state.hardwareSort = event.target.value; renderRows(); });
  const columnDefaults = { name: "asc", memory: "desc", power: "asc", speed: "desc", evidence: "desc", need: "asc", architecture: "asc" };
  for (const button of $$(".column-sort")) button.addEventListener("click", () => {
    const group = button.dataset.sortGroup;
    const prefix = button.dataset.sortPrefix;
    const select = byId(group === "hardware" ? "hardware-sort" : "model-sort");
    const current = group === "hardware" ? state.hardwareSort : state.modelSort;
    const [currentPrefix, currentDirection] = current.split("-");
    const direction = currentPrefix === prefix ? (currentDirection === "asc" ? "desc" : "asc") : columnDefaults[prefix];
    select.value = `${prefix}-${direction}`;
    if (!select.value) return;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
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
  const intelToggles = [byId("intel-only-toggle"), byId("hardware-intel-only-toggle"), byId("pipe-intel-only-toggle")];
  for (const button of intelToggles) button.addEventListener("click", () => {
    state.intelOnly = !state.intelOnly;
    for (const toggle of intelToggles) toggle.setAttribute("aria-pressed", String(state.intelOnly));
    byId("vendor-select").value = state.intelOnly ? "Intel" : "all";
    state.vendor = state.intelOnly ? "Intel" : "all";
    refreshCatalogDropdowns();
      if (state.intelOnly) {
      const intelCpu = state.data.cpus.find((row) => isIntelHardware(row, "cpu"));
      const intelGpu = state.data.gpus.find((row) => row.Vendor === "Intel");
      selectGlobalCpu(intelCpu, { renderHardware: false });
      selectGlobalGpu(intelGpu, { renderHardware: false });
    } else {
      selectGlobalCpu(null, { renderHardware: false });
      selectGlobalGpu(null, { renderHardware: false });
    }
    refreshModelDropdown();
    render();
    if (state.page === "hardware") renderHardwareView();
    if (state.page === "pipeline") window.ModelFitPipeline.render();
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
    ["pcie-width-select", "pcieWidth", render], ["device-vram-input", "deviceVram", render]
  ];
  for (const [id, key, callback] of profileInputs) byId(id).addEventListener("input", (event) => { state[key] = typeof state[key] === "number" ? Number(event.target.value) : event.target.value; callback(); });
  byId("host-cpu-select").addEventListener("change", (event) => { selectGlobalCpu(state.data.cpus.find((row) => row["CPU / platform"] === event.target.value) || null, { renderHardware: state.page === "hardware" }); render(); });
  byId("host-ram-input").addEventListener("input", (event) => { state.hostRam = Number(event.target.value); render(); });
  byId("preferred-gpu-select").addEventListener("change", (event) => { selectGlobalGpu(state.data.gpus.find((row) => row.Accelerator === event.target.value) || null, { renderHardware: state.page === "hardware" }); render(); });
  byId("min-memory-filter").addEventListener("change", (event) => { state.minMemory = Number(event.target.value); render(); });
  byId("max-power-filter").addEventListener("change", (event) => { state.maxPower = Number(event.target.value); render(); });

  byId("hardware-quant-select").addEventListener("change", (event) => { state.hardwareBits = Number(event.target.selectedOptions[0].dataset.bits); renderHardwareView(); });
  byId("hardware-context-select").addEventListener("change", (event) => { state.hardwareContext = Number(event.target.value); renderHardwareView(); });
  byId("hardware-user-select").addEventListener("change", (event) => { state.hardwareUsers = Number(event.target.value); renderHardwareView(); });
  byId("hardware-target-input").addEventListener("input", (event) => { state.hardwareTarget = Number(event.target.value) || 20; renderHardwareView(); });
  byId("hardware-cpu-select").addEventListener("change", (event) => { selectGlobalCpu(state.data.cpus.find((row) => row["CPU / platform"] === event.target.value) || null, { renderHardware: false }); render(); renderHardwareView(); });
  byId("hardware-gpu-select").addEventListener("change", (event) => { selectGlobalGpu(state.data.gpus.find((row) => row.Accelerator === event.target.value) || null, { renderHardware: false }); render(); renderHardwareView(); });
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
  byId("model-sort").addEventListener("change", (event) => { state.modelSort = event.target.value; renderHardwareView(); });

  byId("database-select").addEventListener("change", () => {
    state.databaseSort = { table: byId("database-select").value, column: "", direction: "asc" };
    state.databaseLimit = DATABASE_PAGE;
    renderDatabase();
  });
  byId("database-search").addEventListener("input", () => { state.databaseLimit = DATABASE_PAGE; renderDatabase(); });
  byId("database-sort-column").addEventListener("change", (event) => {
    state.databaseSort = { table: byId("database-select").value, column: event.target.value, direction: "asc" };
    renderDatabase();
  });
  byId("database-sort-direction").addEventListener("click", () => {
    state.databaseSort.direction = state.databaseSort.direction === "asc" ? "desc" : "asc";
    renderDatabase();
  });
  byId("download-table").addEventListener("click", () => downloadCsv(byId("database-select").value, sortedDatabaseRows()));
  byId("download-evidence").addEventListener("click", () => downloadCsv("evidence", sortedEvidenceRows()));
  byId("evidence-search").addEventListener("input", () => { state.evidenceLimit = EVIDENCE_PAGE; renderEvidence(); });
  byId("evidence-sort-column").addEventListener("change", (event) => { state.evidenceSort.column = event.target.value; renderEvidence(); });
  byId("evidence-sort-direction").addEventListener("click", () => {
    state.evidenceSort.direction = state.evidenceSort.direction === "asc" ? "desc" : "asc";
    const direction = state.evidenceSort.direction;
    byId("evidence-sort-direction").dataset.direction = direction;
    byId("evidence-sort-direction").querySelector("span").textContent = direction === "asc" ? "Ascending" : "Descending";
    renderEvidence();
  });
  for (const button of $$(".knowledge-nav button")) button.addEventListener("click", () => {
    $$(".knowledge-nav button").forEach((item) => { item.classList.toggle("active", item === button); item.setAttribute("aria-selected", String(item === button)); });
    $$("[data-knowledge-page-content]").forEach((view) => view.hidden = view.dataset.knowledgePageContent !== button.dataset.knowledgePage);
  });
  for (const button of $$("[data-open-page]")) button.addEventListener("click", () => showPage(button.dataset.openPage));
  for (const button of $$(".status-filter")) button.addEventListener("click", () => {
    state.evidenceStatus = button.dataset.status;
    state.evidenceLimit = EVIDENCE_PAGE;
    $$(".status-filter").forEach((item) => item.classList.toggle("active", item === button));
    renderEvidence();
  });
}

async function boot() {
  try {
    const response = await fetch("data.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`data.json returned ${response.status}`);
    state.data = await response.json();
    efficiencyTable = MF.buildEfficiencyTable(state.data.libraries);
    if (!Object.keys(efficiencyTable).length) throw new Error("data.json has no runtime table. Rebuild it with scripts/build_web_data.py.");
    taskRules = MF.buildTaskRules(state.data.taskRules);
    try {
      const privateResponse = await fetch("private_reference.json", { cache: "no-store" });
      if (privateResponse.ok) {
        state.privateReference = await privateResponse.json();
        const measurements = preparePrivateMeasurements(state.privateReference);
        const select = byId("database-select");
        select.add(new Option(`Private Measurements (${measurements.length})`, "privateMeasurements"));
        select.add(new Option(`Private Findings (${state.data.privateFindings.length})`, "privateFindings"));
        select.add(new Option(`Private Issues (${state.data.privateIssues.length})`, "privateIssues"));
        const friendlyName = (name) => name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
        for (const [name, rows] of Object.entries(state.privateReference.tables || {})) {
          const key = `private_${name}`;
          state.data[key] = rows.map((row) => Object.fromEntries(Object.entries(row).map(([column, value]) => [column, value !== null && typeof value === "object" ? JSON.stringify(value) : value])));
          select.add(new Option(`Private ${friendlyName(name)} (${rows.length})`, key));
        }
      }
    } catch { state.privateReference = null; }
    try {
      const parityResponse = await fetch("parity.json", { cache: "no-store" });
      if (parityResponse.ok) state.parity = await parityResponse.json();
    } catch { state.parity = null; }
    try {
      const verificationResponse = await fetch("verification.json", { cache: "no-store" });
      if (verificationResponse.ok) state.verification = await verificationResponse.json();
    } catch { state.verification = null; }
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
    window.ModelFitPipeline.init({
      data: state.data, efficiencyTable: () => efficiencyTable, runtimes: runtimeNames, intelOnly: () => state.intelOnly,
      gpuLabel: gpuCatalogLabel, gpuName: gpuDisplayName, escapeHtml, iconize,
    });
    render();
    renderEvidence();
    renderParityAudit();
    renderVerification();
    iconize();
    window.ModelFitApp = { state, showPage, render, renderHardwareView, hardwareRows, hardwareRowsForView, compatibleModels, currentHardwareItem, memoryNeed, copySummary, latestModels };
    document.dispatchEvent(new CustomEvent("modelfit:ready"));
    if (new URLSearchParams(location.search).has("selftest")) {
      const script = document.createElement("script");
      script.src = "selftest.js";
      document.head.append(script);
    }
  } catch (error) {
    byId("dataset-count").textContent = "Data could not be loaded";
    byId("model-summary-title").textContent = "Serve this page from a local or GitHub Pages web server.";
    byId("results").innerHTML = `<div class="empty-state"><h3>${escapeHtml(error.message)}</h3><p>Opening index.html directly blocks data.json. Run <code>python -m http.server 8000 -d docs</code>.</p></div>`;
  }
}

boot();
