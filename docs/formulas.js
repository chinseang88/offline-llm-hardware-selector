/* Model Fit sizing formulas: the one source used by the page, the Node checks and scripts/formulas.py. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ModelFit = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const WEIGHT_MARGIN = 1.03;
  const RUNTIME_SHARE = 0.07;
  const DEFAULT_HEADROOM_PCT = 8;
  const EXTRA_CARD_SCALE = 0.8;
  const HOST_RESERVE_GB = 8;
  const VOICE_EXTRA_GB = 5;
  const VOICE_BW_GBPS = 5;
  const PREFILL_MFU = 0.35;
  const CPU_PREFILL_FACTOR = 0.6;
  const PREFILL_GFLOPS_PER_B = 2;
  const OFFLOAD_RUNTIMES = ["llama.cpp", "Ollama", "LM Studio", "IPEX-LLM"];
  const HW_CLASSES = ["NVIDIA GPU", "AMD GPU", "Intel GPU / iGPU", "NPU (Intel/AMD)", "Apple Silicon", "x86 CPU", "ARM CPU", "Jetson (ARM+GPU)", "Qualcomm / Rockchip NPU"];
  const VENDOR_CLASS = { NVIDIA: "NVIDIA GPU", AMD: "AMD GPU", Intel: "Intel GPU / iGPU", Apple: "Apple Silicon" };

  // Mirrors data/task_rules.csv; used only when the dataset has no taskRules table.
  const DEFAULT_TASK_RULES = [
    ["General chat / Q&A", 4.85, 6.6, 4.85, "N"],
    ["Summarisation / RAG", 4.85, 6.6, 4.85, "N"],
    ["Coding", 5.7, 8.5, 4.85, "N"],
    ["Maths / reasoning", 5.7, 8.5, 4.85, "N"],
    ["Agents / tool calling", 5.7, 8.5, 4.85, "N"],
    ["Vision (image + text)", 4.85, 6.6, 4.85, "Y"],
    ["Classification / extraction", 4.85, 4.85, 4.85, "N"],
    ["Translation / multilingual", 5.7, 8.5, 4.85, "N"],
  ];

  function num(value) {
    if (value === null || value === undefined || value === "") return NaN;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : NaN;
  }

  function positive(value) {
    const parsed = num(value);
    return parsed > 0 ? parsed : 0;
  }

  function hasSizingSpecs(model) {
    return ["Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim"].every((key) => positive(model && model[key]) > 0);
  }

  function kvPerTokenGb(model, kvBits) {
    const bits = positive(kvBits) || 8;
    return 2 * positive(model.Layers) * positive(model["KV heads"]) * positive(model["Head dim"]) * bits / 8 / 1e9;
  }

  // App and CLI memory: 3% weight margin, KV at the chosen precision, 7% of weights for execution.
  function memoryNeed(model, options) {
    const o = options || {};
    const users = positive(o.users) || 1;
    const weights = positive(model["Total params (B)"]) * positive(o.bits) / 8 * WEIGHT_MARGIN;
    const kv = kvPerTokenGb(model, o.kvBits) * positive(o.context) * users;
    const runtime = weights * RUNTIME_SHARE;
    return { weights, kv, runtime, total: weights + kv + runtime };
  }

  // Workbook "1 Plan a Build": raw weights, context capped at the model limit, max(1 GB, 10%) runtime.
  function excelMemoryNeed(model, options) {
    const o = options || {};
    const users = positive(o.users) || 1;
    const limit = positive(model["Max context"]);
    const context = limit ? Math.min(positive(o.context), limit) : positive(o.context);
    const weights = positive(model["Total params (B)"]) * positive(o.bits) / 8;
    const kv = kvPerTokenGb(model, o.kvBits) * context * users;
    const runtime = Math.max(1, 0.1 * (weights + kv));
    return { weights, kv, runtime, total: weights + kv + runtime };
  }

  function quantityScale(quantity) {
    const count = Math.max(1, positive(quantity) || 1);
    return 1 + EXTRA_CARD_SCALE * (count - 1);
  }

  function excelQuantityScale(quantity) {
    const count = Math.max(1, positive(quantity) || 1);
    return Math.min(count, 8) * (count > 1 ? EXTRA_CARD_SCALE : 1);
  }

  function decodeTokPerSecond(options) {
    const o = options || {};
    const perToken = positive(o.activeParamsB) * positive(o.bits) / 8;
    const efficiency = positive(o.efficiency);
    if (!perToken || !efficiency) return 0;
    const utilization = o.utilization === undefined ? 1 : positive(o.utilization);
    return efficiency * positive(o.bandwidth) * quantityScale(o.quantity) * utilization / perToken / (positive(o.users) || 1);
  }

  function excelDecodeTokPerSecond(model, options) {
    const o = options || {};
    const efficiency = positive(o.efficiency);
    const perToken = positive(model["Active params (B)"]) * positive(o.bits) / 8 + excelMemoryNeed(model, o).kv;
    if (!perToken || !efficiency) return 0;
    return efficiency * positive(o.bandwidth) * excelQuantityScale(o.quantity) / perToken / (positive(o.users) || 1);
  }

  // Workbook hardware sheet: when a model spills past GPU memory, the GPU share and the CPU share run in series.
  function splitDecodeTokPerSecond(options) {
    const o = options || {};
    const perToken = positive(o.perTokenGb);
    if (!perToken) return 0;
    const share = Math.min(1, Math.max(0, num(o.gpuShare) || 0));
    const gpuRate = positive(o.gpuRate);
    const cpuRate = positive(o.cpuRate);
    if ((share > 0 && !gpuRate) || (share < 1 && !cpuRate)) return 0;
    const seconds = perToken * ((share > 0 ? share / gpuRate : 0) + (share < 1 ? (1 - share) / cpuRate : 0));
    return seconds > 0 ? 1 / seconds / (positive(o.users) || 1) : 0;
  }

  // Workbook Power sheet: prefill is compute-bound. tok/s = MFU x TFLOPS x 1000 / (2 GFLOPs per billion active params).
  function prefillTokPerSecond(options) {
    const o = options || {};
    const gflops = PREFILL_GFLOPS_PER_B * positive(o.activeParamsB);
    const tflops = positive(o.tflops);
    if (!gflops || !tflops) return 0;
    const mfu = o.mfu === undefined || o.mfu === null ? PREFILL_MFU : positive(o.mfu);
    const scale = o.cpu ? CPU_PREFILL_FACTOR : quantityScale(o.quantity);
    return mfu * tflops * scale * 1000 / gflops / (positive(o.users) || 1);
  }

  // Overflow layers on the CPU run in series with the GPU share, as for decode.
  function splitPrefillTokPerSecond(options) {
    const o = options || {};
    return splitDecodeTokPerSecond({ perTokenGb: 1, gpuShare: o.gpuShare, gpuRate: o.gpuRate, cpuRate: o.cpuRate, users: 1 });
  }

  // Workbook Power sheet: TTFT = prompt tokens / prefill tok/s + 1 / decode tok/s. Null when either rate is unknown.
  function timeToFirstToken(options) {
    const o = options || {};
    const prefill = positive(o.prefillTokPerSecond);
    const decode = positive(o.decodeTokPerSecond);
    if (!prefill || !decode) return null;
    return positive(o.promptTokens) / prefill + 1 / decode;
  }

  function isUnifiedMemory(gpu) {
    return /unified/i.test(String((gpu && gpu.Interface) || ""));
  }

  function gpuUsableGb(memoryGb, quantity, headroomPct) {
    const headroom = headroomPct === undefined || headroomPct === null ? DEFAULT_HEADROOM_PCT : Math.max(0, num(headroomPct) || 0);
    return positive(memoryGb) * (1 - headroom / 100) * Math.max(1, positive(quantity) || 1);
  }

  function cpuUsableGb(ramGb) {
    const ram = positive(ramGb);
    return ram - Math.min(HOST_RESERVE_GB, ram * 0.25);
  }

  function offloadSupported(runtime) {
    return OFFLOAD_RUNTIMES.includes(String(runtime || ""));
  }

  function hostOffloadGb(ramGb, platformMaxGb) {
    const ram = positive(ramGb);
    const limit = positive(platformMaxGb);
    return Math.max(0, (limit ? Math.min(ram, limit) : ram) - HOST_RESERVE_GB);
  }

  function parseEfficiency(cell) {
    const match = String(cell === null || cell === undefined ? "" : cell).trim().match(/^\d+(?:\.\d+)?/);
    return match ? Number(match[0]) : 0;
  }

  function buildEfficiencyTable(libraries) {
    const table = {};
    for (const row of libraries || []) {
      const name = row.Library;
      if (!name) continue;
      const classes = {};
      const cells = {};
      for (const hwClass of HW_CLASSES) {
        classes[hwClass] = parseEfficiency(row[hwClass]);
        cells[hwClass] = String(row[hwClass] === null || row[hwClass] === undefined ? "" : row[hwClass]);
      }
      table[name] = {
        classes,
        cells,
        typical: parseEfficiency(row["Typical eff."]),
        vulkan: parseEfficiency(row["Vulkan fallback eff."]),
        minCuda: num(row["Min CUDA compute capability"]),
      };
    }
    return table;
  }

  function hardwareClass(row, kind) {
    const declared = String((row && row["Runtime HW class"]) || "").trim();
    if (HW_CLASSES.includes(declared)) return declared;
    if (kind === "cpu") return /Jetson/.test(String((row && row["CPU / platform"]) || "")) ? "Jetson (ARM+GPU)" : "x86 CPU";
    return VENDOR_CLASS[row && row.Vendor] || "";
  }

  // Workbook rule: native cell when the backend is usable, otherwise the runtime's Vulkan fallback (never on Apple).
  function gpuRuntimeEfficiency(table, runtime, gpu) {
    const entry = (table || {})[runtime];
    if (!entry) return { efficiency: 0, backend: "unsupported" };
    const hwClass = hardwareClass(gpu, "gpu");
    const native = entry.classes[hwClass] || 0;
    let nativeOk = true;
    if (gpu.Vendor === "NVIDIA") {
      const cc = num(gpu["CUDA compute capability"]);
      nativeOk = Number.isFinite(cc) && Number.isFinite(entry.minCuda) && cc >= entry.minCuda;
    } else if (gpu.Vendor === "AMD") {
      nativeOk = String(gpu["ROCm support"] || "") !== "N" || /Vulkan/i.test(entry.cells[hwClass] || "");
    }
    if (nativeOk && native > 0) return { efficiency: native, backend: "native" };
    if (gpu.Vendor === "Apple" || !entry.vulkan) return { efficiency: 0, backend: "unsupported" };
    return { efficiency: entry.vulkan, backend: "vulkan" };
  }

  function cpuRuntimeEfficiency(table, runtime, cpu) {
    const entry = (table || {})[runtime];
    if (!entry) return 0;
    return entry.classes[hardwareClass(cpu, "cpu")] || 0;
  }

  function buildTaskRules(rows) {
    const rules = {};
    const source = rows && rows.length ? rows.map((row) => [row.Task, row["Base min bits (4-30B)"], row["Small model <4B min bits"], row["Large model >=30B min bits"], row["VLM required?"], row.Why, row["Verdict label if too low"]]) : DEFAULT_TASK_RULES;
    for (const [task, base, small, large, vlm, why, verdict] of source) {
      if (!task) continue;
      rules[task] = { task, base: positive(base), small: positive(small), large: positive(large), vlm: String(vlm || "N").toUpperCase() === "Y", why: why || "", verdict: verdict || "" };
    }
    return rules;
  }

  function taskRule(rules, task) {
    return (rules || {})[task] || (rules || {})["General chat / Q&A"] || { task: "General chat / Q&A", base: 4.85, small: 6.6, large: 4.85, vlm: false, why: "", verdict: "" };
  }

  function requiredBits(model, rule) {
    const total = positive(model["Total params (B)"]);
    return total < 4 ? rule.small : total >= 30 ? rule.large : rule.base;
  }

  function isVlm(model) {
    return /VLM|Vision/i.test(String((model && model.Type) || ""));
  }

  function taskAllows(model, rule, bits) {
    return positive(bits) >= requiredBits(model, rule) && (!rule.vlm || isVlm(model));
  }

  function voiceWeightsGb(row) {
    const params = num(row && row["Total params (B)"]);
    const bits = num(row && row["Bits / weight"]);
    return params > 0 && bits > 0 ? params * bits / 8 : null;
  }

  // Workbook "Multi-Stage Pipeline Fit": voice stages reserve first, then LLM slots are checked cumulatively.
  // One set = Stage 1-4 with its own copy of every model (workbook "Concurrent Model Selector" convention).
  function pipelineFit(input) {
    const o = input || {};
    const quantity = Math.max(1, positive(o.quantity) || 1);
    const headroom = Math.max(0, num(o.headroomPct) || 0);
    const usable = positive(o.memoryGb) * quantity * (1 - headroom / 100);
    const effectiveBw = positive(o.bandwidthGbps) * quantityScale(quantity) * positive(o.utilization);
    const kvBits = positive(o.kvBits) || 16;
    const sequential = o.activity === "sequential";
    const promptTokens = positive(o.promptTokens);

    let voiceWeights = 0;
    let voiceExtra = 0;
    let voiceBw = 0;
    let pending = 0;
    const voice = (o.voice || []).map((stage) => {
      const onGpu = stage.placement !== "CPU";
      const weights = onGpu ? (Number.isFinite(num(stage.weightsGb)) ? num(stage.weightsGb) : null) : 0;
      const extra = onGpu ? Math.max(0, num(stage.extraGb) || 0) : 0;
      const bw = onGpu ? Math.max(0, num(stage.bwGbps) || 0) : 0;
      const status = !onGpu ? "CPU - not sized" : weights === null ? "WEIGHTS PENDING" : "INPUTS SET";
      if (status === "WEIGHTS PENDING") pending += 1;
      voiceWeights += weights || 0;
      voiceExtra += extra;
      voiceBw += bw;
      return { stage: stage.stage, name: stage.name, placement: onGpu ? "GPU" : "CPU", weightsGb: weights, extraGb: extra, bwGbps: bw, status };
    });
    const voiceMemory = voiceWeights + voiceExtra;

    let memory = voiceMemory;
    let bwSum = 0;
    let bwMax = 0;
    let fits = 0;
    let used = 0;
    const llms = (o.llms || []).map((slot, index) => {
      const model = slot.model;
      if (!model) return { slot: index + 1, model: null, status: "UNUSED", weights: 0, kvOverhead: 0, total: 0, bwNeeded: 0, cumulativeMemory: memory, cumulativeBw: voiceBw + (sequential ? bwMax : bwSum), fitsAlone: null, tokAlone: null, target: 0, prefillAlone: null, ttftAlone: null, ttftLoaded: null };
      used += 1;
      const bits = positive(slot.bits);
      const context = positive(slot.context);
      const target = positive(slot.target);
      const weights = positive(model["Total params (B)"]) * bits / 8 * WEIGHT_MARGIN;
      const kvOverhead = kvPerTokenGb(model, kvBits) * context + weights * RUNTIME_SHARE;
      const total = weights + kvOverhead;
      const perToken = positive(model["Active params (B)"]) * bits / 8 * WEIGHT_MARGIN;
      const bwNeeded = perToken * target;
      memory += total;
      bwSum += bwNeeded;
      bwMax = Math.max(bwMax, bwNeeded);
      const cumulativeBw = voiceBw + (sequential ? bwMax : bwSum);
      const limit = positive(model["Max context"]);
      let status = "FITS";
      if (!bits || !context || !target) status = "REVIEW: CHECK INPUTS";
      else if (!hasSizingSpecs(model)) status = "REVIEW: MODEL SPECS MISSING";
      else if (limit && context > limit) status = "REVIEW: CONTEXT > MODEL LIMIT";
      else if (memory > usable) status = "NO - GPU MEMORY";
      else if (cumulativeBw > effectiveBw) status = "NO - BANDWIDTH";
      const review = status.startsWith("REVIEW");
      if (status === "FITS") fits += 1;
      const tokAlone = review || !perToken ? null : Math.max(0, effectiveBw - voiceBw) / perToken;
      const prefillAlone = review ? null : prefillTokPerSecond({ activeParamsB: model["Active params (B)"], tflops: o.tflops, quantity, mfu: o.mfu }) || null;
      return {
        slot: index + 1, model: model.Model, status, weights, kvOverhead, total, bwNeeded,
        cumulativeMemory: memory, cumulativeBw,
        fitsAlone: review ? null : voiceMemory + total <= usable,
        tokAlone, target, prefillAlone,
        ttftAlone: timeToFirstToken({ promptTokens, prefillTokPerSecond: prefillAlone, decodeTokPerSecond: tokAlone }),
        ttftLoaded: null,
      };
    });
    const bwRequired = voiceBw + (sequential ? bwMax : bwSum);
    const reviews = llms.filter((row) => row.status.startsWith("REVIEW")).length;
    let verdict = "INPUTS SET";
    if (usable <= 0 || effectiveBw <= 0) verdict = "REVIEW: select accelerator, quantity, headroom and utilization";
    else if (reviews) verdict = "REVIEW: fix LLM inputs";

    // Concurrent sets: every set needs its own memory and bandwidth; the smaller limit wins.
    const countable = verdict === "INPUTS SET" && memory > 0;
    const setsByMemory = countable ? Math.floor(usable / memory + 1e-9) : null;
    const setsByBandwidth = countable && bwRequired > 0 ? Math.floor(effectiveBw / bwRequired + 1e-9) : null;
    const sets = !countable ? null : setsByBandwidth === null ? setsByMemory : Math.min(setsByMemory, setsByBandwidth);
    const setsLimit = sets === null ? null : setsByBandwidth === null || setsByMemory < setsByBandwidth ? "memory" : setsByBandwidth < setsByMemory ? "bandwidth" : "memory and bandwidth";

    // First token with every set busy: all LLMs that answer together share the prefill compute; each still decodes at its target.
    const together = sets ? sets * (sequential ? 1 : Math.max(1, used)) : 0;
    let ttftWorst = null;
    for (const row of llms) {
      if (!together || !row.prefillAlone) continue;
      row.ttftLoaded = timeToFirstToken({ promptTokens, prefillTokPerSecond: row.prefillAlone / together, decodeTokPerSecond: row.target });
      if (row.ttftLoaded !== null && (ttftWorst === null || row.ttftLoaded > ttftWorst)) ttftWorst = row.ttftLoaded;
    }
    return {
      usableGb: usable, effectiveBwGbps: effectiveBw, voice, voiceWeightsGb: voiceWeights, voiceExtraGb: voiceExtra,
      voiceMemoryGb: voiceMemory, voiceBwGbps: voiceBw, voicePending: pending, llms,
      memoryUsedGb: memory, memoryMarginGb: usable - memory, bwRequiredGbps: bwRequired,
      llmsUsed: used, llmsFit: fits, allFit: verdict === "INPUTS SET" && used > 0 && fits === used, verdict,
      setMemoryGb: memory, setBwGbps: bwRequired, setsByMemory, setsByBandwidth, sets, setsLimit, ttftWorstLoaded: ttftWorst,
    };
  }

  return {
    WEIGHT_MARGIN, RUNTIME_SHARE, DEFAULT_HEADROOM_PCT, EXTRA_CARD_SCALE, HOST_RESERVE_GB, VOICE_EXTRA_GB, VOICE_BW_GBPS,
    PREFILL_MFU, CPU_PREFILL_FACTOR, PREFILL_GFLOPS_PER_B,
    OFFLOAD_RUNTIMES, HW_CLASSES, DEFAULT_TASK_RULES,
    hasSizingSpecs, kvPerTokenGb, memoryNeed, excelMemoryNeed, quantityScale, excelQuantityScale,
    decodeTokPerSecond, excelDecodeTokPerSecond, splitDecodeTokPerSecond, isUnifiedMemory,
    prefillTokPerSecond, splitPrefillTokPerSecond, timeToFirstToken,
    gpuUsableGb, cpuUsableGb, offloadSupported, hostOffloadGb,
    parseEfficiency, buildEfficiencyTable, hardwareClass, gpuRuntimeEfficiency, cpuRuntimeEfficiency,
    buildTaskRules, taskRule, requiredBits, isVlm, taskAllows, voiceWeightsGb, pipelineFit,
  };
});
