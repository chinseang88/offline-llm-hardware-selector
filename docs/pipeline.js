/* Multi-LLM page: configure Stage 1 VAD, Stage 2 diarization, Stage 3 ASR and Stage 4 LLMs, then count the sets that run at once. */
(function () {
  "use strict";
  const MF = window.ModelFit;
  const CONTEXTS = [[2048, "2K"], [4096, "4K"], [8192, "8K"], [16384, "16K"], [32768, "32K"], [65536, "64K"], [131072, "128K"], [262144, "256K"]];
  const DEFAULT_GPU = "Intel CRI XT 32Xe 160GB 300W (CCG)";
  const DEFAULT_QUANT = "Q4_K_M / INT4-AWQ";
  const DEFAULT_LLMS = ["Qwen3 4B", "Qwen3.5 9B", "Gemma 4 12B", "Qwen3 32B"];
  const DEFAULT_VOICE = [["VAD", "Silero VAD"], ["Diarization", "pyannote community-1"], ["ASR", "Qwen3-ASR-1.7B"]];
  const STAGE_TITLES = { VAD: "Voice activity detection (VAD)", Diarization: "Speaker diarization", ASR: "Speech recognition (ASR)" };
  const number = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
  const fine = new Intl.NumberFormat("en", { maximumFractionDigits: 3 });

  let ctx = null;
  let state = null;
  const byId = (id) => document.getElementById(id);

  function defaults() {
    const models = new Set(ctx.data.models.map((row) => row.Model));
    const voiceNames = new Set((ctx.data.voiceModels || []).map((row) => `${row.Stage}|${row.Model}`));
    const gpus = ctx.data.gpus;
    const gpu = gpus.find((row) => row.Accelerator === DEFAULT_GPU) || [...gpus].sort((a, b) => Number(b["Memory (GB)"]) - Number(a["Memory (GB)"]))[0];
    return {
      gpu: gpu ? gpu.Accelerator : "",
      quantity: 1,
      runtime: "llama.cpp",
      kvBits: 16,
      headroom: 10,
      activity: "simultaneous",
      util: "auto",
      promptTokens: 2048,
      voice: DEFAULT_VOICE.map(([stage, model]) => ({ stage, model: voiceNames.has(`${stage}|${model}`) ? model : "", placement: "GPU", manual: "", extra: MF.VOICE_EXTRA_GB, bw: MF.VOICE_BW_GBPS })),
      llms: DEFAULT_LLMS.map((model) => ({ model: models.has(model) ? model : "", quant: DEFAULT_QUANT, context: 8192, target: 20 })),
    };
  }

  function quantBits(name) {
    const row = ctx.data.quants.find((quant) => quant.Quantization === name) || ctx.data.quants.find((quant) => quant.Quantization === DEFAULT_QUANT);
    return row ? Number(row["Bits/weight (eff.)"]) : 4.85;
  }

  function selectedGpu() {
    return ctx.data.gpus.find((row) => row.Accelerator === state.gpu) || null;
  }

  function runtimeEfficiency() {
    const gpu = selectedGpu();
    return gpu ? MF.gpuRuntimeEfficiency(ctx.efficiencyTable(), state.runtime, gpu) : { efficiency: 0, backend: "unsupported" };
  }

  function utilization() {
    return state.util === "auto" ? runtimeEfficiency().efficiency : Number(state.util);
  }

  function voiceCatalog(stage, name) {
    return (ctx.data.voiceModels || []).find((row) => row.Stage === stage && row.Model === name) || null;
  }

  function voiceInputs() {
    return state.voice.map((stage) => {
      const manual = stage.manual === "" ? NaN : Number(stage.manual);
      const catalog = MF.voiceWeightsGb(voiceCatalog(stage.stage, stage.model));
      const selected = Boolean(stage.model);
      return {
        stage: stage.stage,
        name: stage.model,
        placement: selected ? stage.placement : "CPU",
        weightsGb: Number.isFinite(manual) && manual >= 0 ? manual : catalog,
        extraGb: selected ? stage.extra : 0,
        bwGbps: selected ? stage.bw : 0,
        selected,
      };
    });
  }

  function compute() {
    const gpu = selectedGpu();
    return MF.pipelineFit({
      memoryGb: gpu ? gpu["Memory (GB)"] : 0,
      bandwidthGbps: gpu ? gpu["Bandwidth (GB/s)"] : 0,
      quantity: state.quantity,
      headroomPct: state.headroom,
      utilization: utilization(),
      kvBits: state.kvBits,
      activity: state.activity,
      tflops: gpu ? ctx.tflops("GPUs", gpu.Accelerator) : 0,
      promptTokens: state.promptTokens,
      voice: voiceInputs(),
      llms: state.llms.map((slot) => ({
        model: ctx.data.models.find((row) => row.Model === slot.model) || null,
        bits: quantBits(slot.quant), context: slot.context, target: slot.target,
      })),
    });
  }

  function pillClass(status) {
    if (/^NO|MISSING/.test(status)) return "red";
    if (/^(FITS|INPUTS SET)$/.test(status)) return "";
    return "amber";
  }

  function option(value, label, selected) {
    return `<option value="${ctx.escapeHtml(value)}"${selected ? " selected" : ""}>${ctx.escapeHtml(label)}</option>`;
  }

  function rebuildGpuSelect() {
    const select = byId("pipe-gpu-select");
    const rows = ctx.data.gpus.filter((row) => Number(row["Memory (GB)"]) > 0 && Number(row["Bandwidth (GB/s)"]) > 0).filter((row) => !ctx.intelOnly() || row.Vendor === "Intel");
    select.replaceChildren(...rows.map((row) => new Option(ctx.gpuLabel(row), row.Accelerator)));
    if (!rows.some((row) => row.Accelerator === state.gpu)) state.gpu = rows[0] ? rows[0].Accelerator : "";
    select.value = state.gpu;
  }

  function syncControls() {
    rebuildGpuSelect();
    const runtimeSelect = byId("pipe-runtime-select");
    runtimeSelect.replaceChildren(...ctx.runtimes().map((name) => new Option(name, name)));
    runtimeSelect.value = state.runtime;
    byId("pipe-qty-select").value = String(state.quantity);
    byId("pipe-kv-select").value = String(state.kvBits);
    byId("pipe-headroom-select").value = String(state.headroom);
    byId("pipe-activity-select").value = state.activity;
    byId("pipe-bw-util-select").value = String(state.util);
    byId("pipe-prompt-select").value = String(state.promptTokens);
    byId("pipe-intel-only-toggle").setAttribute("aria-pressed", String(ctx.intelOnly()));
  }

  // Stages 1 to 3: one card each, in pipeline order.
  function renderStages(result) {
    byId("pipe-stage-cards").innerHTML = state.voice.map((stage, index) => {
      const out = result.voice[index];
      const catalog = MF.voiceWeightsGb(voiceCatalog(stage.stage, stage.model));
      const names = (ctx.data.voiceModels || []).filter((row) => row.Stage === stage.stage).map((row) => row.Model);
      const status = !stage.model ? "NOT USED" : out.status;
      const memory = (out.weightsGb || 0) + out.extraGb;
      return `<section class="pipeline-card stage-card" aria-label="Stage ${index + 1}: ${ctx.escapeHtml(stage.stage)}">
        <div class="pipeline-card-head"><div><p class="section-label">Stage ${index + 1}</p><h2>${ctx.escapeHtml(STAGE_TITLES[stage.stage] || stage.stage)}</h2></div><span class="status-pill ${status === "NOT USED" ? "amber" : pillClass(status)}">${ctx.escapeHtml(status)}</span></div>
        <div class="stage-fields">
          <label>Model<select data-voice="${index}" data-field="model">${option("", "(Not used)", !stage.model)}${names.map((name) => option(name, name, name === stage.model)).join("")}</select></label>
          <label>Runs on<select data-voice="${index}" data-field="placement">${option("GPU", "GPU", stage.placement === "GPU")}${option("CPU", "CPU (not sized)", stage.placement === "CPU")}</select></label>
          <label>Resident weights (GB)<input data-voice="${index}" data-field="manual" type="number" min="0" step="0.001" inputmode="decimal" value="${ctx.escapeHtml(stage.manual)}" placeholder="${catalog === null ? "measure" : fine.format(catalog)}"></label>
          <label>Activation reserve (GB)<input data-voice="${index}" data-field="extra" type="number" min="0" step="0.5" inputmode="decimal" value="${stage.extra}"></label>
          <label>Bandwidth (GB/s)<input data-voice="${index}" data-field="bw" type="number" min="0" step="0.5" inputmode="decimal" value="${stage.bw}"></label>
        </div>
        <p class="stage-total">Each set reserves <b>${fine.format(memory)} GB</b> and <b>${number.format(out.bwGbps)} GB/s</b> for this stage.</p>
      </section>`;
    }).join("");
  }

  // Stage 4: up to four LLMs. Status is the running check inside one set, in slot order.
  function renderLlms(result) {
    const models = ctx.data.models.filter((row) => MF.hasSizingSpecs(row));
    const rows = state.llms.map((slot, index) => {
      const out = result.llms[index];
      const used = Boolean(out.model);
      const cell = (value, digits) => used ? (digits === 3 ? fine : number).format(value) : "—";
      return `<tr class="${used ? "" : "unused"}">
        <th scope="row">LLM ${index + 1}</th>
        <td><select data-llm="${index}" data-field="model" aria-label="LLM ${index + 1} model">${option("", "(Unused)", !slot.model)}${models.map((row) => option(row.Model, `${row.Model} · ${number.format(row["Total params (B)"])}B`, row.Model === slot.model)).join("")}</select></td>
        <td><select data-llm="${index}" data-field="quant" aria-label="LLM ${index + 1} quantization">${ctx.data.quants.map((quant) => option(quant.Quantization, quant.Quantization, quant.Quantization === slot.quant)).join("")}</select></td>
        <td><select data-llm="${index}" data-field="context" aria-label="LLM ${index + 1} context">${CONTEXTS.map(([value, label]) => option(value, label, value === slot.context)).join("")}</select></td>
        <td><input data-llm="${index}" data-field="target" type="number" min="1" step="1" inputmode="numeric" value="${slot.target}" aria-label="LLM ${index + 1} target tokens per second"></td>
        <td>${cell(out.weights, 3)}</td><td>${cell(out.kvOverhead, 3)}</td><td>${cell(out.total, 3)}</td><td>${cell(out.bwNeeded)}</td>
        <td>${used ? ctx.seconds(out.ttftAlone) : "—"}</td><td>${used ? ctx.seconds(out.ttftLoaded) : "—"}</td>
        <td><span class="status-pill ${used ? pillClass(out.status) : "amber"}">${ctx.escapeHtml(used ? out.status : "UNUSED")}</span></td>
      </tr>`;
    }).join("");
    byId("pipe-llm-table").innerHTML = `<thead><tr><th scope="col">Slot</th><th scope="col">Model</th><th scope="col">Quantization</th><th scope="col">Context</th><th scope="col">Target tok/s</th><th scope="col">Weights (GB)</th><th scope="col">KV + overhead (GB)</th><th scope="col">Total (GB)</th><th scope="col">Bandwidth (GB/s)</th><th scope="col">First token, alone</th><th scope="col">First token, every set busy</th><th scope="col">Status in one set</th></tr></thead><tbody>${rows}</tbody><tfoot><tr><th scope="row" colspan="7">One set, Stage 1 to 4 (${fine.format(result.voiceMemoryGb)} GB and ${number.format(result.voiceBwGbps)} GB/s from Stages 1 to 3)</th><td>${fine.format(result.setMemoryGb)}</td><td>${number.format(result.setBwGbps)}</td><td colspan="3">of ${number.format(result.usableGb)} GB and ${number.format(result.effectiveBwGbps)} GB/s available</td></tr></tfoot>`;
    const badge = byId("pipe-llm-badge");
    badge.className = `status-pill ${result.llmsUsed && result.llmsFit === result.llmsUsed ? "" : result.llmsUsed ? "red" : "amber"}`;
    badge.textContent = result.llmsUsed ? `${result.llmsFit} of ${result.llmsUsed} fit in one set` : "No LLM selected";
  }

  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

  // The answer: how many complete sets of Stage 1 to 4 run at the same time, and the first-token time under that load.
  function renderAnswer(result) {
    const gpu = selectedGpu();
    const eff = runtimeEfficiency();
    const planning = gpu && ctx.isPlanning(gpu);
    const hardware = gpu ? `${ctx.gpuName(gpu.Accelerator)}${state.quantity > 1 ? ` × ${state.quantity}` : ""}` : "no accelerator";
    const sets = result.sets;
    let headline;
    let reason;
    if (result.verdict !== "INPUTS SET") {
      headline = "No answer yet";
      reason = result.verdict;
    } else if (sets === null) {
      headline = "No answer yet";
      reason = "Select a model in at least one stage.";
    } else {
      headline = sets === 0 ? "No complete set of Stage 1 to 4 fits" : `${plural(sets, "set")} of Stage 1 to 4 ${sets === 1 ? "runs" : "run"} concurrently`;
      const byMemory = `Memory allows ${plural(result.setsByMemory, "set")}`;
      reason = result.setsByBandwidth === null ? `${byMemory}. No GPU stage uses bandwidth, so memory is the only limit.` : `${byMemory} and bandwidth allows ${result.setsByBandwidth}. The limit is ${result.setsLimit}.`;
    }
    const notes = [`On ${hardware}`, "each set loads its own copy of every model"];
    if (state.util === "auto") notes.push(eff.efficiency ? `bandwidth use ${Math.round(eff.efficiency * 100)}% from ${state.runtime}${eff.backend === "vulkan" ? " (Vulkan fallback)" : ""}` : `${state.runtime} does not run on this accelerator: pick another runtime or set bandwidth use by hand`);
    else notes.push(`bandwidth use fixed at ${Math.round(Number(state.util) * 100)}%`);
    notes.push(`KV cache ${state.kvBits}-bit`, state.activity === "sequential" ? "LLMs in a set answer one at a time" : "all LLMs in a set answer at once");
    const unused = state.voice.filter((stage) => !stage.model).map((stage) => stage.stage);
    if (unused.length) notes.push(`${unused.join(" and ")} not used, so not counted`);
    if (!result.llmsUsed) notes.push("Stage 4 has no LLM selected");
    if (result.voicePending) notes.push(`${plural(result.voicePending, "voice stage weight")} counted as 0 GB until measured`);

    const alone = result.llms.map((row) => row.ttftAlone).filter((value) => value !== null);
    const tflops = gpu ? ctx.tflops("GPUs", gpu.Accelerator) : 0;
    const prompt = `${state.promptTokens.toLocaleString()}-token prompt`;
    let ttft = "—";
    let ttftNote;
    if (!tflops) ttftNote = planning ? "Time to first token: no public compute figure for this pre-release row" : "Time to first token: no compute estimate for this accelerator";
    else if (result.ttftWorstLoaded !== null) { ttft = ctx.seconds(result.ttftWorstLoaded); ttftNote = `Slowest time to first token with every set busy · ${prompt}`; }
    else if (alone.length) { ttft = ctx.seconds(Math.max(...alone)); ttftNote = `Slowest time to first token, one LLM running alone · ${prompt}`; }
    else ttftNote = "Time to first token: select an LLM in Stage 4";
    const count = (value) => value === null ? "—" : String(value);
    byId("pipeline-summary").innerHTML = `<div><p class="section-label">Answer${planning ? " · pre-release planning row" : ""}</p><h2>${ctx.escapeHtml(headline)}</h2><p class="pipeline-verdict">${ctx.escapeHtml(reason)}</p><p class="pipeline-notes">${ctx.escapeHtml(notes.join(" · "))}</p></div><div class="metric"><span>${count(result.setsByMemory)}</span><small>Sets by memory · ${number.format(result.setMemoryGb)} GB per set of ${number.format(result.usableGb)} GB usable</small></div><div class="metric"><span>${count(result.setsByBandwidth)}</span><small>Sets by bandwidth · ${number.format(result.setBwGbps)} GB/s per set of ${number.format(result.effectiveBwGbps)} GB/s</small></div><div class="metric"><span>${ttft}</span><small>${ctx.escapeHtml(ttftNote)}</small></div>`;
  }

  function render() {
    if (!ctx || !state) return null;
    syncControls();
    const result = compute();
    renderStages(result);
    renderLlms(result);
    renderAnswer(result);
    ctx.iconize();
    return result;
  }

  function onStageChange(event) {
    const target = event.target;
    const field = target.dataset.field;
    if (!field) return;
    if (target.dataset.voice !== undefined) {
      const stage = state.voice[Number(target.dataset.voice)];
      if (field === "model") { stage.model = target.value; stage.manual = ""; }
      else if (field === "placement") stage.placement = target.value;
      else if (field === "manual") stage.manual = target.value;
      else stage[field] = Math.max(0, Number(target.value) || 0);
    } else if (target.dataset.llm !== undefined) {
      const slot = state.llms[Number(target.dataset.llm)];
      if (field === "context") slot.context = Number(target.value);
      else if (field === "target") slot.target = Math.max(1, Number(target.value) || 1);
      else slot[field] = target.value;
    }
    render();
  }

  function init(context) {
    ctx = context;
    state = defaults();
    byId("pipe-gpu-select").addEventListener("change", (event) => { state.gpu = event.target.value; render(); });
    byId("pipe-qty-select").addEventListener("change", (event) => { state.quantity = Number(event.target.value); render(); });
    byId("pipe-runtime-select").addEventListener("change", (event) => { state.runtime = event.target.value; render(); });
    byId("pipe-kv-select").addEventListener("change", (event) => { state.kvBits = Number(event.target.value); render(); });
    byId("pipe-headroom-select").addEventListener("change", (event) => { state.headroom = Number(event.target.value); render(); });
    byId("pipe-activity-select").addEventListener("change", (event) => { state.activity = event.target.value; render(); });
    byId("pipe-bw-util-select").addEventListener("change", (event) => { state.util = event.target.value; render(); });
    byId("pipe-prompt-select").addEventListener("change", (event) => { state.promptTokens = Number(event.target.value); render(); });
    byId("pipe-reset").addEventListener("click", () => { state = defaults(); render(); });
    byId("pipe-stage-cards").addEventListener("change", onStageChange);
    byId("pipe-llm-table").addEventListener("change", onStageChange);
  }

  window.ModelFitPipeline = { init, render, compute, reset() { state = defaults(); return render(); }, get state() { return state; } };
})();
