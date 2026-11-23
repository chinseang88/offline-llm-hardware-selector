/* Multi-LLM page: configure Stage 1 VAD, Stage 2 diarization, Stage 3 ASR and Stage 4 LLM, each with its own runtime and KV cache precision; Stage 5 counts the sets that run at once. */
(function () {
  "use strict";
  const MF = window.ModelFit;
  const CONTEXTS = [[2048, "2K"], [4096, "4K"], [8192, "8K"], [16384, "16K"], [32768, "32K"], [65536, "64K"], [131072, "128K"], [262144, "256K"]];
  const DEFAULT_GPU = "Intel CRI XT 32Xe 160GB 300W (CCG)";
  const DEFAULT_QUANT = "Q4_K_M / INT4-AWQ";
  const DEFAULT_LLM = "Qwen3 32B";
  const DEFAULT_RUNTIME = "llama.cpp";
  const KV_BITS = [[4, "4-bit"], [8, "8-bit"], [16, "16-bit FP16"]];
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
    const runtime = ctx.runtimes().includes(DEFAULT_RUNTIME) ? DEFAULT_RUNTIME : ctx.runtimes()[0] || "";
    return {
      gpu: gpu ? gpu.Accelerator : "",
      quantity: 1,
      headroom: 10,
      util: "auto",
      promptTokens: 2048,
      voice: DEFAULT_VOICE.map(([stage, model]) => ({ stage, model: voiceNames.has(`${stage}|${model}`) ? model : "", placement: "GPU", runtime, kvBits: MF.VOICE_RESERVE_BITS, manual: "", extra: MF.VOICE_EXTRA_GB, bw: MF.VOICE_BW_GBPS })),
      llm: { model: models.has(DEFAULT_LLM) ? DEFAULT_LLM : "", quant: DEFAULT_QUANT, runtime, kvBits: 16, context: 8192, target: 20 },
    };
  }

  function quantBits(name) {
    const row = ctx.data.quants.find((quant) => quant.Quantization === name) || ctx.data.quants.find((quant) => quant.Quantization === DEFAULT_QUANT);
    return row ? Number(row["Bits/weight (eff.)"]) : 4.85;
  }

  function selectedGpu() {
    return ctx.data.gpus.find((row) => row.Accelerator === state.gpu) || null;
  }

  // Share of the accelerator's bandwidth one stage's runtime can use; the BW utilization control can fix it for every stage.
  function stageUse(runtime) {
    if (state.util !== "auto") return { efficiency: Number(state.util), backend: "fixed" };
    const gpu = selectedGpu();
    return gpu ? MF.gpuRuntimeEfficiency(ctx.efficiencyTable(), runtime, gpu) : { efficiency: 0, backend: "unsupported" };
  }

  function useLabel(runtime) {
    const use = stageUse(runtime);
    const percent = `${Math.round(use.efficiency * 100)}%`;
    return use.backend === "fixed" ? `${percent}, fixed by hand` : `${percent}, ${runtime}${use.backend === "vulkan" ? " via Vulkan fallback" : ""}`;
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
        utilization: stageUse(stage.runtime).efficiency,
        kvBits: stage.kvBits,
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
      tflops: gpu ? ctx.tflops("GPUs", gpu.Accelerator) : 0,
      promptTokens: state.promptTokens,
      voice: voiceInputs(),
      llms: [{
        model: ctx.data.models.find((row) => row.Model === state.llm.model) || null,
        bits: quantBits(state.llm.quant), context: state.llm.context, target: state.llm.target,
        utilization: stageUse(state.llm.runtime).efficiency, kvBits: state.llm.kvBits,
      }],
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
    byId("pipe-qty-select").value = String(state.quantity);
    byId("pipe-headroom-select").value = String(state.headroom);
    byId("pipe-bw-util-select").value = String(state.util);
    byId("pipe-prompt-select").value = String(state.promptTokens);
    byId("pipe-intel-only-toggle").setAttribute("aria-pressed", String(ctx.intelOnly()));
  }

  function runtimeField(attrs, value, disabled) {
    return `<label>Runtime<select ${attrs} data-field="runtime"${disabled ? " disabled" : ""}>${ctx.runtimes().map((name) => option(name, name, name === value)).join("")}</select></label>`;
  }

  function kvField(attrs, value, disabled) {
    return `<label>KV cache<select ${attrs} data-field="kvBits"${disabled ? " disabled" : ""}>${KV_BITS.map(([bits, label]) => option(bits, label, bits === value)).join("")}</select></label>`;
  }

  // What one stage takes from the accelerator's bandwidth once its runtime's efficiency is counted.
  function bandwidthShare(runtime, rawGbps) {
    if (!stageUse(runtime).efficiency) return ` <b>${ctx.escapeHtml(runtime)}</b> does not run on this accelerator: choose another runtime or set BW utilization by hand.`;
    return rawGbps > 0 ? ` That takes <b>${number.format(rawGbps)} GB/s</b> of the accelerator (${ctx.escapeHtml(useLabel(runtime))}).` : "";
  }

  // Stage 4: the one LLM of each set.
  function llmCard(result) {
    const out = result.llms[0];
    const used = Boolean(out.model);
    const slot = state.llm;
    const models = ctx.data.models.filter((row) => MF.hasSizingSpecs(row));
    const status = used ? out.status : "NOT USED";
    return `<section class="pipeline-card stage-card" aria-label="Stage 4: LLM">
        <div class="pipeline-card-head"><div><p class="section-label">Stage 4</p><h2>Language model (LLM)</h2></div><span class="status-pill ${used ? pillClass(status) : "amber"}">${ctx.escapeHtml(status)}</span></div>
        <div class="stage-fields">
          <label>Model<select data-llm="0" data-field="model">${option("", "(Not used)", !slot.model)}${models.map((row) => option(row.Model, `${row.Model} · ${number.format(row["Total params (B)"])}B`, row.Model === slot.model)).join("")}</select></label>
          <label>Quantization<select data-llm="0" data-field="quant">${ctx.data.quants.map((quant) => option(quant.Quantization, quant.Quantization, quant.Quantization === slot.quant)).join("")}</select></label>
          ${runtimeField('data-llm="0"', slot.runtime, false)}
          ${kvField('data-llm="0"', slot.kvBits, false)}
          <label>Context<select data-llm="0" data-field="context">${CONTEXTS.map(([value, label]) => option(value, label, value === slot.context)).join("")}</select></label>
          <label>Target tok/s<input data-llm="0" data-field="target" type="number" min="1" step="1" inputmode="numeric" value="${slot.target}"></label>
          <label>First token, running alone<output>${used ? ctx.seconds(out.ttftAlone) : "—"}</output></label>
        </div>
        <p class="stage-total">Each set needs <b>${fine.format(out.total)} GB</b> (weights ${fine.format(out.weights)} + KV cache and overhead ${fine.format(out.kvOverhead)}) and <b>${number.format(out.bwNeeded)} GB/s</b> for this stage.${used ? bandwidthShare(slot.runtime, out.rawBwNeeded) : ""}</p>
      </section>`;
  }

  // Stages 1 to 4: one card each, in pipeline order.
  function renderStages(result) {
    byId("pipe-stage-cards").innerHTML = state.voice.map((stage, index) => {
      const out = result.voice[index];
      const catalog = MF.voiceWeightsGb(voiceCatalog(stage.stage, stage.model));
      const names = (ctx.data.voiceModels || []).filter((row) => row.Stage === stage.stage).map((row) => row.Model);
      const status = !stage.model ? "NOT USED" : out.status;
      const memory = (out.weightsGb || 0) + out.extraGb;
      const onGpu = Boolean(stage.model) && stage.placement === "GPU";
      return `<section class="pipeline-card stage-card" aria-label="Stage ${index + 1}: ${ctx.escapeHtml(stage.stage)}">
        <div class="pipeline-card-head"><div><p class="section-label">Stage ${index + 1}</p><h2>${ctx.escapeHtml(STAGE_TITLES[stage.stage] || stage.stage)}</h2></div><span class="status-pill ${status === "NOT USED" ? "amber" : pillClass(status)}">${ctx.escapeHtml(status)}</span></div>
        <div class="stage-fields">
          <label>Model<select data-voice="${index}" data-field="model">${option("", "(Not used)", !stage.model)}${names.map((name) => option(name, name, name === stage.model)).join("")}</select></label>
          <label>Runs on<select data-voice="${index}" data-field="placement">${option("GPU", "GPU", stage.placement === "GPU")}${option("CPU", "CPU (not sized)", stage.placement === "CPU")}</select></label>
          ${runtimeField(`data-voice="${index}"`, stage.runtime, !onGpu)}
          ${kvField(`data-voice="${index}"`, stage.kvBits, !onGpu)}
          <label>Resident weights (GB)<input data-voice="${index}" data-field="manual" type="number" min="0" step="0.001" inputmode="decimal" value="${ctx.escapeHtml(stage.manual)}" placeholder="${catalog === null ? "measure" : fine.format(catalog)}"></label>
          <label>Activation reserve (GB at FP16)<input data-voice="${index}" data-field="extra" type="number" min="0" step="0.5" inputmode="decimal" value="${stage.extra}"></label>
          <label>Bandwidth (GB/s)<input data-voice="${index}" data-field="bw" type="number" min="0" step="0.5" inputmode="decimal" value="${stage.bw}"></label>
        </div>
        <p class="stage-total">Each set reserves <b>${fine.format(memory)} GB</b> (weights ${fine.format(out.weightsGb || 0)} + reserve ${fine.format(out.extraGb)}${onGpu && stage.kvBits !== MF.VOICE_RESERVE_BITS ? ` at ${stage.kvBits}-bit` : ""}) and <b>${number.format(out.bwGbps)} GB/s</b> for this stage.${onGpu ? bandwidthShare(stage.runtime, out.rawBwGbps) : ""}</p>
      </section>`;
    }).join("") + llmCard(result);
  }

  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

  // Stage 5: how many complete sets of Stage 1 to 4 run at the same time, and the first-token time under that load.
  function renderAnswer(result) {
    const gpu = selectedGpu();
    const planning = gpu && ctx.isPlanning(gpu);
    const hardware = gpu ? `${ctx.gpuName(gpu.Accelerator)}${state.quantity > 1 ? ` × ${state.quantity}` : ""}` : "no accelerator";
    // Stages that take accelerator bandwidth, each with the runtime chosen in its card.
    const active = [
      ...state.voice.filter((stage) => stage.model && stage.placement === "GPU").map((stage) => ({ name: stage.stage, runtime: stage.runtime, kvBits: stage.kvBits })),
      ...(state.llm.model ? [{ name: "LLM", runtime: state.llm.runtime, kvBits: state.llm.kvBits }] : []),
    ].map((stage) => ({ ...stage, use: stageUse(stage.runtime) }));
    const blocked = active.filter((stage) => !stage.use.efficiency);
    const sets = result.sets;
    let headline;
    let reason;
    if (result.verdict !== "INPUTS SET") {
      headline = "No answer yet";
      reason = blocked.length ? `${blocked.map((stage) => `${stage.name}: ${stage.runtime} does not run on this accelerator`).join(". ")}. Choose another runtime in that stage or set BW utilization by hand.` : result.verdict;
    } else if (sets === null) {
      headline = "No answer yet";
      reason = "Select a model in at least one stage.";
    } else {
      headline = sets === 0 ? "No complete set of Stage 1 to 4 fits" : `${plural(sets, "set")} of Stage 1 to 4 ${sets === 1 ? "runs" : "run"} concurrently`;
      const byMemory = `Memory allows ${plural(result.setsByMemory, "set")}`;
      reason = result.setsByBandwidth === null ? `${byMemory}. No GPU stage uses bandwidth, so memory is the only limit.` : `${byMemory} and bandwidth allows ${result.setsByBandwidth}. The limit is ${result.setsLimit}.`;
    }
    const notes = [`On ${hardware}`, "each set loads its own copy of every model"];
    const percent = (use) => `${Math.round(use.efficiency * 100)}%`;
    if (state.util !== "auto") notes.push(`bandwidth use fixed at ${Math.round(Number(state.util) * 100)}% for every stage`);
    else if (active.length && new Set(active.map((stage) => stage.runtime)).size === 1) notes.push(`every stage on ${active[0].runtime}${active[0].use.efficiency ? ` (${percent(active[0].use)} bandwidth use${active[0].use.backend === "vulkan" ? ", Vulkan fallback" : ""})` : ""}`);
    else if (active.length) notes.push(`bandwidth use by stage runtime: ${active.map((stage) => `${stage.name} ${stage.runtime}${stage.use.efficiency ? ` ${percent(stage.use)}` : " not supported"}`).join(", ")}`);
    if (active.length && new Set(active.map((stage) => stage.kvBits)).size === 1) notes.push(`KV cache ${active[0].kvBits}-bit in every stage`);
    else if (active.length) notes.push(`KV cache by stage: ${active.map((stage) => `${stage.name} ${stage.kvBits}-bit`).join(", ")}`);
    const unused = state.voice.filter((stage) => !stage.model).map((stage) => stage.stage);
    if (unused.length) notes.push(`${unused.join(" and ")} not used, so not counted`);
    if (!result.llmsUsed) notes.push("Stage 4 has no LLM selected");
    if (result.voicePending) notes.push(`${plural(result.voicePending, "voice stage weight")} counted as 0 GB until measured`);

    const alone = result.llms[0].ttftAlone;
    const tflops = gpu ? ctx.tflops("GPUs", gpu.Accelerator) : 0;
    const prompt = `${state.promptTokens.toLocaleString()}-token prompt${gpu && ctx.computeDerived("GPUs", gpu.Accelerator) ? " · compute derived from bandwidth" : ""}`;
    let ttft = "—";
    let ttftNote;
    if (!tflops) ttftNote = "Time to first token: no compute estimate for this accelerator";
    else if (result.ttftWorstLoaded !== null) { ttft = ctx.seconds(result.ttftWorstLoaded); ttftNote = `Time to first token with every set busy · ${prompt}`; }
    else if (alone !== null) { ttft = ctx.seconds(alone); ttftNote = `Time to first token, one set running alone · ${prompt}`; }
    else ttftNote = "Time to first token: select an LLM in Stage 4";
    const count = (value) => value === null ? "—" : String(value);
    byId("pipeline-summary").innerHTML = `<div><p class="section-label">Stage 5${planning ? " · pre-release planning row" : ""}</p><h2>${ctx.escapeHtml(headline)}</h2><p class="pipeline-verdict">${ctx.escapeHtml(reason)}</p><p class="pipeline-notes">${ctx.escapeHtml(notes.join(" · "))}</p></div><div class="metric"><span>${count(result.setsByMemory)}</span><small>Sets by memory · ${number.format(result.setMemoryGb)} GB per set of ${number.format(result.usableGb)} GB usable</small></div><div class="metric"><span>${count(result.setsByBandwidth)}</span><small>Sets by bandwidth · ${number.format(result.setRawBwGbps)} GB/s per set of ${number.format(result.rawBwGbps)} GB/s at the accelerator</small></div><div class="metric"><span>${ttft}</span><small>${ctx.escapeHtml(ttftNote)}</small></div>`;
  }

  function render() {
    if (!ctx || !state) return null;
    syncControls();
    const result = compute();
    renderStages(result);
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
      else if (field === "runtime") stage.runtime = target.value;
      else if (field === "kvBits") stage.kvBits = Number(target.value);
      else if (field === "manual") stage.manual = target.value;
      else stage[field] = Math.max(0, Number(target.value) || 0);
    } else if (target.dataset.llm !== undefined) {
      if (field === "context" || field === "kvBits") state.llm[field] = Number(target.value);
      else if (field === "target") state.llm.target = Math.max(1, Number(target.value) || 1);
      else state.llm[field] = target.value;
    }
    render();
  }

  function init(context) {
    ctx = context;
    state = defaults();
    byId("pipe-gpu-select").addEventListener("change", (event) => { state.gpu = event.target.value; render(); });
    byId("pipe-qty-select").addEventListener("change", (event) => { state.quantity = Number(event.target.value); render(); });
    byId("pipe-headroom-select").addEventListener("change", (event) => { state.headroom = Number(event.target.value); render(); });
    byId("pipe-bw-util-select").addEventListener("change", (event) => { state.util = event.target.value; render(); });
    byId("pipe-prompt-select").addEventListener("change", (event) => { state.promptTokens = Number(event.target.value); render(); });
    byId("pipe-reset").addEventListener("click", () => { state = defaults(); render(); });
    byId("pipe-stage-cards").addEventListener("change", onStageChange);
  }

  window.ModelFitPipeline = { init, render, compute, reset() { state = defaults(); return render(); }, get state() { return state; } };
})();
