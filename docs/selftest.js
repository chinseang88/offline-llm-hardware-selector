/* In-page self test. Open the site with ?selftest=1 to exercise every page and control in both themes.
   Results: a panel at the bottom of the page and window.__selftest = { done, pass, fail, notes, results }. */
(async function () {
  "use strict";
  const app = window.ModelFitApp;
  const MF = window.ModelFit;
  const pipeline = window.ModelFitPipeline;
  const theme = window.ModelFitTheme;
  const state = app.state;
  const data = state.data;
  const results = [];
  const byId = (id) => document.getElementById(id);
  const all = (selector, root = document) => [...root.querySelectorAll(selector)];
  // Wait one frame; fall back to a timer where frames are not produced (hidden tabs, headless browsers).
  const tick = () => new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    requestAnimationFrame(() => setTimeout(finish, 0));
    setTimeout(finish, 80);
  });
  const near = (a, b, tolerance = 0.06) => Math.abs(a - b) <= tolerance;
  const number = (text) => parseFloat(String(text).replace(/,/g, ""));
  const check = (name, ok, detail = "") => results.push({ name, status: ok ? "pass" : "fail", detail: ok ? "" : String(detail) });
  const note = (name, detail) => results.push({ name, status: "note", detail: String(detail) });
  const set = (id, value) => {
    const element = byId(id);
    element.value = String(value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const gpuRows = () => app.hardwareRowsForView("gpus");
  const fitsOf = (rows) => rows.filter((row) => row.fits).length;
  const shown = (id) => all(".result-row", byId(id)).length;
  const need = () => app.memoryNeed(state.model);
  // Database, Evidence and Know more must show the fastest first-token time of the current Model → Hardware selection.
  const firstTokenStrip = (pageId) => {
    const text = byId(pageId).querySelector("[data-ttft-strip]").textContent;
    const best = gpuRows().filter((row) => row.fits && row.ttft !== null).sort((a, b) => a.ttft - b.ttft)[0];
    const shownSeconds = text.match(/Time to first token\s*([\d.,]+) s/);
    return Boolean(best && shownSeconds) && text.includes(best.name) && /tok\/s prefill \+ 1 ÷/.test(text) && near(number(shownSeconds[1]), best.ttft, Math.max(0.006, best.ttft * 0.006));
  };

  async function section(name, body) {
    try {
      await body();
    } catch (error) {
      check(`${name}: ran without an error`, false, error && error.stack ? error.stack.split("\n").slice(0, 2).join(" ") : error);
    }
  }

  // ---------------------------------------------------------------- start-up
  await section("Start-up", async () => {
    const ids = all("[id]").map((element) => element.id);
    check("Start-up: element ids are unique", new Set(ids).size === ids.length, ids.filter((id, index) => ids.indexOf(id) !== index).slice(0, 5));
    check("Start-up: dataset loaded", data.models.length > 0 && data.gpus.length > 0 && data.cpus.length > 0 && data.evidence.length > 0);
    check("Start-up: dataset stamp shows the row counts", byId("dataset-count").textContent.includes(`${data.models.length} models`));
    check("Start-up: runtime list comes from the runtime table", all("option", byId("runtime-select")).length === data.libraries.length);
    check("Start-up: task list comes from the task rules", all("option", byId("task-select")).length === data.taskRules.length);
    if (window.lucide) check("Start-up: icons rendered", all("svg.lucide").length > 10);
    else note("Start-up: icon library", "The icon CDN did not load, so icons are not drawn. Layout and behaviour are unaffected.");
    check("Start-up: verification panel rendered", byId("verification-content").textContent.trim().length > 0);
    check("Start-up: formula audit rendered", byId("parity-audit-content").textContent.includes("%"));
    check("Start-up: the pipeline tab is named Multi-LLM", document.querySelector('.nav-page[data-page="pipeline"]').textContent.trim() === "Multi-LLM");
    check("Start-up: compute estimates loaded for first-token time", data.compute.length > 0 && app.tflopsFor("GPUs", "Intel Arc Pro B60") > 0);
    const noCompute = [...data.gpus.filter((row) => !app.tflopsFor("GPUs", row.Accelerator)).map((row) => row.Accelerator), ...data.cpus.filter((row) => !app.tflopsFor("CPUs", row["CPU / platform"])).map((row) => row["CPU / platform"])];
    check("Start-up: every GPU and CPU has a compute figure for first-token time", noCompute.length === 0, noCompute.slice(0, 5).join(", "));
    check("Start-up: workbook estimates are never marked as derived", !app.computeDerived("GPUs", "Intel Arc Pro B60") && !app.computeDerived("GPUs", "RTX 5090"));
    check("Start-up: method section explains time to first token and concurrent sets", /Time to first token/.test(byId("method").textContent) && /Concurrent sets/.test(byId("method").textContent));
  });

  // ------------------------------------------------------ model -> hardware
  await section("Model page", async () => {
    app.showPage("model");
    set("model-search", "Qwen3 32B");
    check("Model page: selecting a model updates the summary", byId("model-summary-title").textContent === "Qwen3 32B");
    const base = need();
    check("Model page: memory shown equals the formula", near(number(byId("need-memory").textContent), base.total), `${byId("need-memory").textContent} vs ${base.total}`);
    check("Model page: weights + KV + runtime add up", near(base.weights + base.kv + base.runtime, base.total, 1e-9));
    check("Model page: GPU counter equals the rows that fit", Number(byId("gpu-count").textContent) === fitsOf(gpuRows()) && fitsOf(gpuRows()) > 0);
    check("Model page: hardware rows are listed", shown("results") > 0);

    set("quant-select", "Q8_0 / INT8");
    check("Model page: a higher quantization needs more memory", need().total > base.total);
    set("quant-select", "Q4_K_M / INT4-AWQ");
    set("context-select", 32768);
    check("Model page: KV cache grows 4x from 8K to 32K context", near(need().kv, base.kv * 4, 1e-6));
    set("context-select", 8192);
    set("user-select", 4);
    check("Model page: KV cache grows with users", near(need().kv, base.kv * 4, 1e-6));
    set("user-select", 1);
    set("kv-bits-select", 16);
    check("Model page: 16-bit KV doubles the 8-bit KV cache", near(need().kv, base.kv * 2, 1e-6));
    set("kv-bits-select", 8);

    set("runtime-select", "MLX / mlx-lm");
    check("Model page: MLX only fits Apple hardware", gpuRows().filter((row) => row.fits).every((row) => row.vendor === "Apple"));
    set("runtime-select", "llama.cpp");

    set("task-select", "Vision (image + text)");
    check("Model page: a text-only model is blocked for the vision task", fitsOf(gpuRows()) === 0 && !byId("model-warnings").hidden);
    set("task-select", "General chat / Q&A");
    check("Model page: warnings clear when the task is valid again", byId("model-warnings").hidden);

    const small = data.models.find((row) => Number(row["Total params (B)"]) < 4 && Number(row["Total params (B)"]) > 0.5 && !/VLM|Vision/i.test(row.Type || ""));
    if (small) {
      set("model-search", small.Model);
      set("task-select", "Coding");
      check("Model page: a small model at Q4 is below the coding floor", fitsOf(gpuRows()) === 0 && /floor/.test(byId("model-warnings").textContent));
      set("quant-select", "Q8_0 / INT8");
      check("Model page: the same model at Q8 clears the floor", fitsOf(gpuRows()) > 0);
      set("quant-select", "Q4_K_M / INT4-AWQ");
      set("task-select", "General chat / Q&A");
    }
    set("model-search", "Qwen3 32B");
    set("context-select", 262144);
    check("Model page: context above the model limit raises a warning", /published limit/.test(byId("model-warnings").textContent));
    set("context-select", 8192);

    const before = Math.max(...gpuRows().map((row) => row.memory));
    const smallCard = gpuRows().find((row) => row.physicalMemory <= 12 && row.vendor !== "Apple");
    set("host-cpu-select", "Core Ultra 9 285K");
    set("offload-select", "Y");
    const offloaded = gpuRows().find((row) => smallCard && row.name === smallCard.name);
    check("Model page: CPU offload adds host RAM to the GPU pool", offloaded && offloaded.memory > smallCard.memory + 1, offloaded && offloaded.memory);
    check("Model page: an offloaded model is slower than GPU-only speed", offloaded && offloaded.gpuShare !== null && offloaded.speed < smallCard.speed);
    set("runtime-select", "vLLM");
    check("Model page: a runtime that cannot split warns and ignores offload", /cannot split/.test(byId("model-warnings").textContent) && near(Math.max(...gpuRows().map((row) => row.memory)), before));
    set("runtime-select", "llama.cpp");
    set("offload-select", "N");
    set("host-cpu-select", "");

    const single = gpuRows()[0];
    set("gpu-quantity-input", 2);
    const pair = gpuRows()[0];
    check("Model page: two cards double usable memory", near(pair.memory, single.memory * 2, 1e-6));
    check("Model page: the second card adds 80% speed", near(pair.speed, single.speed * 1.8, 1e-6));
    set("gpu-quantity-input", 1);

    set("fit-filter", "all");
    const everything = shown("results");
    set("fit-filter", "fits");
    const fitting = shown("results");
    set("target-input", 100);
    set("fit-filter", "fast");
    const fast = shown("results");
    check("Model page: Show filter narrows all > fits > meets target", everything >= fitting && fitting >= fast && everything > fast, `${everything}/${fitting}/${fast}`);
    set("target-input", 20);
    set("fit-filter", "fits");

    set("hardware-sort", "name-asc");
    const names = all("#results .hardware-name strong").map((element) => element.textContent);
    check("Model page: sorting by name orders the rows", names.length > 1 && names.every((name, index) => index === 0 || names[index - 1].localeCompare(name) <= 0));
    byId("table-head").querySelector('[data-sort-prefix="memory"]').click();
    check("Model page: column headers change the sort", state.hardwareSort.startsWith("memory"));
    set("hardware-sort", "speed-desc");

    set("vendor-select", "NVIDIA");
    check("Model page: vendor filter keeps one vendor", all("#results .vendor-chip").every((chip) => chip.textContent === "NV") && shown("results") > 0);
    set("vendor-select", "all");

    byId("intel-only-toggle").click();
    check("Model page: Intel-only shows Intel hardware only", gpuRows().every((row) => row.vendor === "Intel") && gpuRows().length > 0);
    check("Model page: Intel-only is mirrored on every page", ["hardware-intel-only-toggle", "pipe-intel-only-toggle"].every((id) => byId(id).getAttribute("aria-pressed") === "true"));
    byId("intel-only-toggle").click();
    check("Model page: Intel-only switches off again", !state.intelOnly && gpuRows().some((row) => row.vendor !== "Intel"));

    const withoutLegacy = gpuRows().length;
    byId("legacy-toggle").click();
    check("Model page: Legacy adds older GPUs", gpuRows().length > withoutLegacy && byId("legacy-toggle").getAttribute("aria-pressed") === "true");
    byId("legacy-toggle").click();

    all(".view-switcher button").find((button) => button.dataset.view === "cpus").click();
    check("Model page: CPU view lists CPUs", byId("fits-label").textContent === "CPU fits" && Number(byId("fits-count").textContent) === Number(byId("cpu-count").textContent));
    all(".view-switcher button").find((button) => button.dataset.view === "gpus").click();
    check("Model page: GPU view returns", byId("fits-label").textContent === "GPU fits");

    set("model-quick-filter", "gemma");
    const options = all("option", byId("model-search")).map((option) => option.textContent.toLowerCase());
    check("Model page: quick filter narrows the model list", options.length < data.models.length && options.slice(1).every((text) => text.includes("gemma")));
    check("Model page: quick filter keeps the selected model", state.model.Model === "Qwen3 32B");
    set("min-memory-filter", 48);
    byId("reset-model-filters").click();
    check("Model page: Reset clears the filters", byId("model-quick-filter").value === "" && state.minMemory === 0 && all("option", byId("model-search")).length === data.models.length);

    const summary = await app.copySummary();
    check("Model page: copy summary names the model and memory", summary.includes("Qwen3 32B") && summary.includes("GB"));
    check("Model page: copy summary includes the fastest first token", /Fastest first token/.test(summary));

    const timedRow = () => gpuRows().find((row) => row.value === "RTX 5090");
    const base5090 = timedRow();
    const expected = state.promptTokens / MF.prefillTokPerSecond({ activeParamsB: state.model["Active params (B)"], tflops: app.tflopsFor("GPUs", "RTX 5090") }) + 1 / base5090.speed;
    check("Model page: first-token time = prompt / prefill + 1 / decode", base5090.ttft !== null && near(base5090.ttft, expected, 1e-9), `${base5090.ttft} vs ${expected}`);
    check("Model page: each hardware row shows a first-token line", all("#results .result-row").every((row) => /First token/.test(row.querySelector(".speed-cell").textContent)));
    check("Model page: summary reports the fastest and median first token", /Fastest/.test(byId("ttft-summary").textContent) && /Median/.test(byId("ttft-summary").textContent));
    check("Model page: every GPU that fits has a first-token time", gpuRows().filter((row) => row.fits).every((row) => row.ttft !== null) && all("#results .result-row").every((row) => /First token \d/.test(row.querySelector(".speed-cell").textContent)));
    const derivedRow = gpuRows().find((row) => row.fits && app.computeDerived("GPUs", row.value));
    if (derivedRow) {
      const derivedPrefill = MF.prefillTokPerSecond({ activeParamsB: state.model["Active params (B)"], tflops: app.tflopsFor("GPUs", derivedRow.value) });
      check("Model page: a row with derived compute uses the same first-token formula", near(derivedRow.ttft, state.promptTokens / derivedPrefill + 1 / derivedRow.speed, 1e-9));
      check("Model page: a row with derived compute is labelled", all("#results .result-row").some((row) => row.querySelector(".hardware-name strong").textContent === derivedRow.name && /derived compute/.test(row.querySelector(".speed-cell").textContent)) && /derived from bandwidth/.test(byId("ttft-summary").textContent));
    }
    all(".view-switcher button").find((button) => button.dataset.view === "cpus").click();
    check("Model page: every CPU that fits has a first-token time", app.hardwareRowsForView("cpus").filter((row) => row.fits).every((row) => row.ttft !== null) && all("#results .result-row").every((row) => /First token \d/.test(row.querySelector(".speed-cell").textContent)));
    all(".view-switcher button").find((button) => button.dataset.view === "gpus").click();
    set("prompt-length-input", 8192);
    check("Model page: a longer prompt raises first-token time", timedRow().ttft > base5090.ttft * 3);
    set("prompt-length-input", 2048);
    set("user-select", 4);
    check("Model page: concurrent users raise first-token time", timedRow().ttft > base5090.ttft * 3.9);
    set("user-select", 1);
    set("gpu-quantity-input", 2);
    check("Model page: a second card lowers first-token time", timedRow().ttft < base5090.ttft);
    set("gpu-quantity-input", 1);
    set("hardware-sort", "ttft-asc");
    const quickest = gpuRows().filter((row) => row.fits && row.ttft !== null).sort((a, b) => a.ttft - b.ttft)[0];
    check("Model page: sorting by first token puts the fastest row first", byId("results").querySelector(".hardware-name strong").textContent === quickest.name);
    set("hardware-sort", "speed-desc");

    byId("show-model-evidence").click();
    check("Model page: Know more opens the detail drawer", byId("evidence-drawer").classList.contains("open") && byId("drawer-title").textContent === "Qwen3 32B");
    const tabs = all("[data-knowledge-tab]", byId("drawer-content"));
    tabs.find((tab) => tab.dataset.knowledgeTab === "evidence").click();
    check("Model page: drawer tabs switch", all(".fact-row", byId("drawer-content")).some((row) => !row.closest("[hidden]")));
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    check("Model page: Escape closes the drawer", !byId("evidence-drawer").classList.contains("open"));
    byId("results").querySelector(".row-detail").click();
    check("Model page: hardware fact check opens", byId("evidence-drawer").classList.contains("open"));
    byId("close-drawer").click();
  });

  // ------------------------------------------------------ hardware -> models
  await section("Hardware page", async () => {
    app.showPage("hardware");
    check("Hardware page: page is visible", !byId("hardware-page").hidden && byId("model-page").hidden);
    set("hardware-gpu-select", "Intel Arc Pro B60");
    const metric = (index) => number(all("#hardware-summary .metric span")[index].textContent);
    const card = data.gpus.find((row) => row.Accelerator === "Intel Arc Pro B60");
    check("Hardware page: usable memory is VRAM less 8% headroom", near(metric(0), MF.gpuUsableGb(card["Memory (GB)"], 1, 8)), metric(0));
    check("Hardware page: compatible models are listed", shown("hardware-model-results") > 0 && metric(2) > 0);
    const modelRows = () => app.compatibleModels(app.currentHardwareItem());
    const tflops = app.tflopsFor("GPUs", "Intel Arc Pro B60");
    check("Hardware page: summary shows the compute estimate behind first-token time", near(metric(3), tflops) && /TFLOPS/.test(all("#hardware-summary .metric span")[3].textContent));
    const firstModel = modelRows()[0];
    const expectedFirst = 2048 / MF.prefillTokPerSecond({ activeParamsB: firstModel.model["Active params (B)"], tflops }) + 1 / firstModel.speed;
    check("Hardware page: first-token time = prompt / prefill + 1 / decode", near(firstModel.ttft, expectedFirst, 1e-9), `${firstModel.ttft} vs ${expectedFirst}`);
    check("Hardware page: every model that fits has a first-token time", modelRows().every((row) => row.ttft !== null) && all("#hardware-model-results .result-row").every((row) => /First token \d/.test(row.querySelector(".speed-cell").textContent)));
    set("hardware-prompt-input", 8192);
    check("Hardware page: a longer prompt raises first-token time", modelRows().find((row) => row.model === firstModel.model).ttft > firstModel.ttft * 3);
    set("hardware-prompt-input", 2048);
    set("hardware-user-select", 4);
    check("Hardware page: concurrent users raise first-token time", modelRows().find((row) => row.model === firstModel.model).ttft > firstModel.ttft * 3.9);
    set("hardware-user-select", 1);
    set("model-sort", "ttft-asc");
    const quickestModel = [...modelRows()].sort((a, b) => a.ttft - b.ttft)[0];
    check("Hardware page: sorting by first token puts the fastest model first", near(modelRows()[0].ttft, quickestModel.ttft, 1e-12));
    set("model-sort", "speed-desc");
    set("hardware-gpu-quantity-input", 2);
    check("Hardware page: two cards double usable memory", near(metric(0), MF.gpuUsableGb(card["Memory (GB)"], 2, 8)));
    set("hardware-gpu-quantity-input", 1);
    set("memory-headroom-input", 20);
    check("Hardware page: headroom changes usable memory", near(metric(0), card["Memory (GB)"] * 0.8));
    set("memory-headroom-input", 8);
    const bandwidth = metric(1);
    set("bandwidth-util-input", 50);
    check("Hardware page: bandwidth utilization scales bandwidth", near(metric(1), bandwidth / 2, 0.11));
    set("bandwidth-util-input", 100);

    set("hardware-runtime-select", "TensorRT-LLM");
    check("Hardware page: an unsupported runtime blocks every model and says why", metric(2) === 0 && !byId("hardware-warnings").hidden && /does not run/.test(byId("hardware-warnings").textContent));
    set("hardware-runtime-select", "llama.cpp");
    check("Hardware page: warnings clear with a supported runtime", byId("hardware-warnings").hidden);

    const fitsBefore = metric(2);
    const memoryBefore = metric(0);
    check("Hardware page: a CPU stays selected after the model page cleared its host", byId("hardware-cpu-select").value !== "" && state.hardwareCpu !== null);
    set("hardware-cpu-select", "Core Ultra 9 285K");
    set("hardware-offload-select", "Y");
    check("Hardware page: CPU offload adds host RAM", metric(0) > memoryBefore + 8 && metric(2) >= fitsBefore, `${metric(0)} GB, ${metric(2)} fit`);
    set("hardware-offload-select", "N");

    set("hardware-task-select", "Classification / extraction");
    const fitsAtQ4 = metric(2);
    set("hardware-quant-select", "Q8_0 / INT8");
    check("Hardware page: a higher quantization fits fewer models", metric(2) < fitsAtQ4, `${metric(2)} vs ${fitsAtQ4}`);
    set("hardware-quant-select", "Q4_K_M / INT4-AWQ");
    set("hardware-task-select", "General chat / Q&A");
    set("hardware-quant-select", "Q8_0 / INT8");
    check("Hardware page: small models need more bits for chat, so Q8 admits them", app.compatibleModels(app.currentHardwareItem()).some((row) => row.fits && Number(row.model["Total params (B)"]) < 4));
    set("hardware-quant-select", "Q4_K_M / INT4-AWQ");
    check("Hardware page: at Q4 small models are held back by the chat floor", app.compatibleModels(app.currentHardwareItem(), { ignoreShowFilter: true }).filter((row) => Number(row.model["Total params (B)"]) < 4).every((row) => row.blocker === "Below task quant floor" || row.blocker === "Runtime cannot load model"));
    set("hardware-task-select", "Vision (image + text)");
    check("Hardware page: the vision task keeps vision models only", app.compatibleModels(app.currentHardwareItem()).every((row) => !row.fits || /VLM|Vision/i.test(row.model.Type)));
    set("hardware-task-select", "General chat / Q&A");

    set("hardware-model-filter", "all");
    const everything = shown("hardware-model-results");
    set("hardware-model-filter", "fits");
    check("Hardware page: Show filter narrows the list", everything > shown("hardware-model-results"));
    set("hardware-model-quick-filter", "qwen");
    const narrowed = shown("hardware-model-results");
    check("Hardware page: quick filter narrows the models", narrowed > 0 && narrowed < fitsBefore);
    byId("reset-hardware-filters").click();
    check("Hardware page: Reset clears the filters", byId("hardware-model-quick-filter").value === "" && shown("hardware-model-results") === fitsBefore);
    set("model-sort", "name-asc");
    const names = all("#hardware-model-results .hardware-name strong").map((element) => element.textContent);
    check("Hardware page: sorting by name orders the models", names.every((name, index) => index === 0 || names[index - 1].localeCompare(name) <= 0));
    set("model-sort", "speed-desc");
    document.querySelector("#hardware-summary .text-button").click();
    check("Hardware page: fact check opens the drawer", byId("evidence-drawer").classList.contains("open"));
    byId("close-drawer").click();
  });

  // ------------------------------------------------------------ multi-llm
  await section("Multi-LLM page", async () => {
    app.showPage("pipeline");
    let result = pipeline.reset();
    const gpu = data.gpus.find((row) => row.Accelerator === pipeline.state.gpu);
    const stageCards = () => all("#pipe-stage-cards .stage-card");
    const answer = () => byId("pipeline-summary").textContent;
    const change = (host, selector, value) => { const element = byId(host).querySelector(selector); element.value = value; element.dispatchEvent(new Event("change", { bubbles: true })); };
    const expectedSets = (r) => Math.min(Math.floor(r.usableGb / r.setMemoryGb + 1e-9), Math.floor(r.rawBwGbps / r.setRawBwGbps + 1e-9));
    const useOf = (runtime) => MF.gpuRuntimeEfficiency(MF.buildEfficiencyTable(data.libraries), runtime, gpu).efficiency;
    const runtimeSelects = () => all('#pipe-stage-cards select[data-field="runtime"]');
    const stageLabels = () => stageCards().map((card) => card.querySelector(".section-label").textContent).join(",");
    check("Multi-LLM page: Stages 1 to 4 each have one card", !byId("pipeline-page").hidden && stageCards().length === 4 && stageLabels() === "Stage 1,Stage 2,Stage 3,Stage 4");
    check("Multi-LLM page: Stage 4 holds exactly one LLM", all('#pipe-stage-cards [data-llm][data-field="model"]').length === 1 && result.llms.length === 1 && result.llmsUsed === 1);
    check("Multi-LLM page: Stage 5 follows the four stages and gives the concurrent sets", Boolean(byId("pipe-stage-cards").compareDocumentPosition(byId("pipeline-summary")) & Node.DOCUMENT_POSITION_FOLLOWING) && /^Stage 5/.test(answer()) && /concurrently/.test(answer()));
    check("Multi-LLM page: every stage has its own runtime, listed from the runtime table", runtimeSelects().length === 4 && stageCards().every((card) => card.querySelectorAll('select[data-field="runtime"]').length === 1) && runtimeSelects().every((select) => select.options.length === data.libraries.length));
    check("Multi-LLM page: KV cache is set in the LLM stage, not in the top strip", stageCards()[3].querySelectorAll('select[data-field="kvBits"]').length === 1 && !document.querySelector("#pipe-kv-select, #pipe-runtime-select") && all(".pipeline-hw-strip select").length === 4);
    check("Multi-LLM page: voice stages say how their cache is counted", stageCards().slice(0, 3).every((card) => /KV cache/.test(card.textContent) && card.querySelectorAll('[data-field="kvBits"]').length === 0) && /In the reserve/.test(stageCards()[2].textContent));
    check("Multi-LLM page: usable memory is accelerator memory less headroom", near(result.usableGb, gpu["Memory (GB)"] * 0.9, 1e-6));
    check("Multi-LLM page: one set is Stages 1 to 3 plus the Stage 4 LLM", near(result.setMemoryGb, result.voiceMemoryGb + result.llms[0].total, 1e-9) && near(result.setBwGbps, result.voiceBwGbps + result.llms[0].bwNeeded, 1e-9));
    check("Multi-LLM page: each stage's bandwidth is divided by its runtime's efficiency", near(result.setRawBwGbps, result.voice.reduce((sum, row) => sum + row.bwGbps / useOf("llama.cpp"), 0) + result.llms[0].bwNeeded / useOf("llama.cpp"), 1e-9) && near(result.rawBwGbps, gpu["Bandwidth (GB/s)"], 1e-9));
    check("Multi-LLM page: sets = the smaller of the memory and bandwidth limits", result.sets === expectedSets(result) && result.sets === Math.min(result.setsByMemory, result.setsByBandwidth));
    check("Multi-LLM page: the default stages give at least one concurrent set", result.sets >= 1 && answer().includes(`${result.sets} set`));
    const baseSets = result.sets;
    const baseSetMemory = result.setMemoryGb;
    change("pipe-stage-cards", '[data-llm="0"][data-field="kvBits"]', "8");
    check("Multi-LLM page: 8-bit KV in the LLM stage makes each set smaller than FP16", pipeline.compute().setMemoryGb < baseSetMemory && pipeline.compute().llms[0].kvBits === 8 && /LLM KV cache 8-bit/.test(answer()));
    change("pipe-stage-cards", '[data-llm="0"][data-field="kvBits"]', "16");
    set("pipe-qty-select", 2);
    check("Multi-LLM page: two accelerators at least double the sets by memory", near(pipeline.compute().usableGb, gpu["Memory (GB)"] * 2 * 0.9, 1e-6) && pipeline.compute().setsByMemory >= result.setsByMemory * 2);
    set("pipe-qty-select", 1);
    set("pipe-headroom-select", 0);
    check("Multi-LLM page: zero headroom uses all memory", near(pipeline.compute().usableGb, gpu["Memory (GB)"], 1e-6));
    set("pipe-headroom-select", 10);
    set("pipe-bw-util-select", "0.8");
    check("Multi-LLM page: bandwidth use can be fixed by hand for every stage", pipeline.compute().voice.every((row) => row.utilization === 0.8) && pipeline.compute().llms[0].utilization === 0.8 && near(pipeline.compute().setRawBwGbps, result.setBwGbps / 0.8, 1e-9) && /fixed at 80% for every stage/.test(answer()));
    set("pipe-bw-util-select", "auto");

    const other = data.libraries.map((row) => row.Library).find((name) => useOf(name) > 0 && useOf(name) !== useOf("llama.cpp"));
    change("pipe-stage-cards", '[data-llm="0"][data-field="runtime"]', other);
    let mixed = pipeline.compute();
    check("Multi-LLM page: the LLM stage can use a different runtime from the voice stages", pipeline.state.llm.runtime === other && pipeline.state.voice.every((stage) => stage.runtime === "llama.cpp") && near(mixed.llms[0].rawBwNeeded, mixed.llms[0].bwNeeded / useOf(other), 1e-9) && near(mixed.voiceRawBwGbps, result.voiceRawBwGbps, 1e-9) && mixed.sets === expectedSets(mixed) && answer().includes(`LLM ${other}`));
    change("pipe-stage-cards", '[data-voice="2"][data-field="runtime"]', other);
    mixed = pipeline.compute();
    check("Multi-LLM page: a voice stage's runtime changes only that stage's bandwidth share", near(mixed.voice[2].rawBwGbps, mixed.voice[2].bwGbps / useOf(other), 1e-9) && near(mixed.voice[0].rawBwGbps, result.voice[0].rawBwGbps, 1e-9) && stageCards()[2].textContent.includes(other));
    byId("pipe-reset").click();
    change("pipe-stage-cards", '[data-llm="0"][data-field="runtime"]', "MLX / mlx-lm");
    check("Multi-LLM page: an LLM runtime that does not run gives no answer and names the stage", pipeline.compute().sets === null && /No answer yet/.test(answer()) && /LLM: MLX \/ mlx-lm does not run on this accelerator/.test(answer()) && /REVIEW/.test(stageCards()[3].textContent));
    change("pipe-stage-cards", '[data-llm="0"][data-field="runtime"]', "llama.cpp");
    change("pipe-stage-cards", '[data-voice="0"][data-field="runtime"]', "MLX / mlx-lm");
    check("Multi-LLM page: a voice runtime that does not run is flagged in its own stage", pipeline.compute().sets === null && /VAD: MLX \/ mlx-lm does not run on this accelerator/.test(answer()) && /REVIEW/.test(stageCards()[0].textContent) && !/REVIEW/.test(stageCards()[3].textContent));
    set("pipe-bw-util-select", "0.5");
    check("Multi-LLM page: fixing bandwidth use by hand overrides an unsupported runtime", pipeline.compute().sets === baseSets);
    set("pipe-bw-util-select", "auto");
    change("pipe-stage-cards", '[data-voice="0"][data-field="placement"]', "CPU");
    check("Multi-LLM page: a stage moved to the CPU needs no accelerator runtime", pipeline.compute().sets !== null && stageCards()[0].querySelector('select[data-field="runtime"]').disabled);
    byId("pipe-reset").click();

    change("pipe-stage-cards", '[data-llm="0"][data-field="target"]', "400");
    check("Multi-LLM page: a speed target the bandwidth cannot carry gives zero sets", pipeline.compute().sets === 0 && pipeline.compute().setsLimit === "bandwidth" && /No complete set/.test(answer()));
    change("pipe-stage-cards", '[data-llm="0"][data-field="target"]', "10");
    check("Multi-LLM page: a lower speed target lets bandwidth carry more sets", pipeline.compute().setsByBandwidth > result.setsByBandwidth);
    change("pipe-stage-cards", '[data-llm="0"][data-field="target"]', "20");
    change("pipe-stage-cards", '[data-llm="0"][data-field="quant"]', "Q8_0 / INT8");
    check("Multi-LLM page: a higher quantization makes each set larger", pipeline.compute().setMemoryGb > baseSetMemory);
    change("pipe-stage-cards", '[data-llm="0"][data-field="quant"]', "Q4_K_M / INT4-AWQ");
    const small = data.models.find((row) => row.Model === "Qwen3 4B") || data.models.find((row) => MF.hasSizingSpecs(row) && Number(row["Total params (B)"]) < 8);
    change("pipe-stage-cards", '[data-llm="0"][data-field="model"]', small.Model);
    check("Multi-LLM page: a smaller Stage 4 LLM allows more sets", pipeline.compute().sets > baseSets && answer().includes(`${pipeline.compute().sets} sets`));
    change("pipe-stage-cards", '[data-llm="0"][data-field="model"]', "");
    check("Multi-LLM page: Stage 4 without an LLM is named in Stage 5", pipeline.compute().llmsUsed === 0 && /Stage 4 has no LLM selected/.test(answer()) && /NOT USED/.test(stageCards()[3].textContent));
    byId("pipe-reset").click();
    const voiceBefore = pipeline.compute().voiceMemoryGb;
    change("pipe-stage-cards", '[data-voice="2"][data-field="placement"]', "CPU");
    check("Multi-LLM page: a voice stage moved to CPU frees GPU memory in every set", pipeline.compute().voiceMemoryGb < voiceBefore);
    change("pipe-stage-cards", '[data-voice="0"][data-field="manual"]', "0.002");
    check("Multi-LLM page: a measured weight replaces a pending one", pipeline.compute().voice[0].status === "INPUTS SET");
    change("pipe-stage-cards", '[data-voice="1"][data-field="model"]', "");
    check("Multi-LLM page: a stage that is not used is named in Stage 5", /Diarization not used/.test(answer()) && /NOT USED/.test(stageCards()[1].textContent));
    const smallest = [...data.gpus].filter((row) => Number(row["Memory (GB)"]) > 0).sort((a, b) => a["Memory (GB)"] - b["Memory (GB)"])[0];
    set("pipe-gpu-select", smallest.Accelerator);
    check("Multi-LLM page: a small accelerator fits no set and names memory as the limit", pipeline.compute().sets === 0 && pipeline.compute().setsByMemory === 0 && pipeline.compute().llms[0].status === "NO - GPU MEMORY");
    byId("pipe-reset").click();
    check("Multi-LLM page: Reset restores the defaults", pipeline.state.gpu === gpu.Accelerator && pipeline.compute().llmsUsed === 1 && pipeline.compute().sets === baseSets && pipeline.state.llm.runtime === "llama.cpp" && pipeline.state.llm.kvBits === 16 && pipeline.state.voice.every((stage) => stage.runtime === "llama.cpp"));

    check("Multi-LLM page: the default accelerator reports time to first token", pipeline.compute().ttftWorstLoaded !== null && /first token with every set busy/.test(answer()) && /\d s/.test(all("#pipeline-summary .metric span")[2].textContent));
    check("Multi-LLM page: derived compute is named in Stage 5", app.computeDerived("GPUs", gpu.Accelerator) === /compute derived from bandwidth/.test(answer()));
    const untimed = all("option", byId("pipe-gpu-select")).map((option) => option.value).filter((name) => { set("pipe-gpu-select", name); const row = pipeline.compute().llms[0]; return row.tokAlone > 0 && row.ttftAlone === null; });
    check("Multi-LLM page: every accelerator that runs the Stage 4 LLM has a first-token time", untimed.length === 0, untimed.slice(0, 5).join(", "));
    set("pipe-gpu-select", "H200 NVL 141GB");
    result = pipeline.compute();
    const tflops = app.tflopsFor("GPUs", "H200 NVL 141GB");
    const llm = data.models.find((row) => row.Model === pipeline.state.llm.model);
    const prefill = MF.prefillTokPerSecond({ activeParamsB: llm["Active params (B)"], tflops });
    check("Multi-LLM page: first token alone = prompt / prefill + 1 / own speed", near(result.llms[0].ttftAlone, 2048 / prefill + 1 / result.llms[0].tokAlone, 1e-9) && /\d s/.test(stageCards()[3].querySelector("output").textContent));
    check("Multi-LLM page: with every set busy the LLMs share the prefill compute", result.sets >= 1 && near(result.ttftWorstLoaded, 2048 / (prefill / result.sets) + 1 / 20, 1e-9) && /first token with every set busy/.test(answer()));
    set("pipe-prompt-select", 8192);
    check("Multi-LLM page: a longer prompt raises first-token time", pipeline.compute().llms[0].ttftAlone > result.llms[0].ttftAlone * 3);
    byId("pipe-reset").click();
  });

  // -------------------------------------------------------------- database
  await section("Database page", async () => {
    app.showPage("database");
    const select = byId("database-select");
    for (const option of all("option", select)) {
      set("database-select", option.value);
      const headers = all("#database-table thead th").length;
      const body = all("#database-table tbody tr").length;
      check(`Database page: ${option.textContent.trim()} table renders`, headers > 0 && body > 0 && body === Math.min(500, data[option.value].length), `${headers} columns, ${body} rows`);
    }
    set("database-select", "models");
    set("database-search", "qwen");
    const filtered = all("#database-table tbody tr").length;
    check("Database page: search filters the rows", filtered > 0 && filtered < data.models.length);
    set("database-search", "");
    all("#database-table thead th button")[0].click();
    check("Database page: clicking a header sorts the table", state.databaseSort.column !== "" && all("#database-table thead th")[0].getAttribute("aria-sort") === "ascending");
    byId("database-sort-direction").click();
    check("Database page: sort direction toggles", state.databaseSort.direction === "desc");
    set("database-sort-column", "");
    check("Database page: original order can be restored", state.databaseSort.column === "");
    byId("intel-only-toggle").click();
    set("database-select", "gpus");
    check("Database page: Intel-only filters the GPU table", all("#database-table tbody tr").length === data.gpus.filter((row) => row.Vendor === "Intel").length);
    byId("intel-only-toggle").click();
    set("database-select", "compute");
    check("Database page: the compute table covers every GPU and CPU", data.compute.length === data.gpus.length + data.cpus.length && all("#database-table tbody tr").length === data.compute.length);
    set("database-select", "models");
    check("Database page: shows the time to first token of the current selection", firstTokenStrip("database-page"));
  });

  // -------------------------------------------------------------- evidence
  await section("Evidence page", async () => {
    app.showPage("evidence");
    const entries = () => all("#evidence-results .evidence-entry");
    check("Evidence page: first page of checks is listed", entries().length === Math.min(300, data.evidence.length));
    if (data.evidence.length > 300) {
      check("Evidence page: Show more is offered", !byId("evidence-more-wrap").hidden);
      byId("evidence-more-wrap").querySelector("button").click();
      check("Evidence page: Show more loads the next page", entries().length === Math.min(600, data.evidence.length));
    }
    all(".status-filter").find((button) => button.dataset.status === "MISMATCH").click();
    check("Evidence page: status filter keeps one status", entries().length > 0 && entries().every((entry) => entry.querySelector(".status-pill").textContent === "MISMATCH"));
    all(".status-filter").find((button) => button.dataset.status === "all").click();
    const sheets = all("option", byId("evidence-sheet-filter")).map((option) => option.value);
    check("Evidence page: table filter lists the evidence tables", sheets.includes("Models") && sheets.includes("GPUs") && sheets.includes("CPUs"));
    set("evidence-sheet-filter", "Compute");
    check("Evidence page: compute estimates behind first-token time are logged as estimates", entries().length > 0 && entries().every((entry) => entry.querySelector(".status-pill").textContent === "DERIVED/ESTIMATE"));
    set("evidence-sheet-filter", "GPUs");
    check("Evidence page: table filter keeps one table", entries().length > 0 && entries().every((entry) => entry.querySelector("code").textContent.endsWith("GPUs")));
    set("evidence-sheet-filter", "all");
    set("evidence-search", "bandwidth");
    check("Evidence page: search filters the checks", entries().length > 0 && /matching checks/.test(byId("evidence-count").textContent));
    set("evidence-search", "");
    set("evidence-sort-column", "Item");
    byId("evidence-sort-direction").click();
    check("Evidence page: sorting changes the order", state.evidenceSort.column === "Item" && state.evidenceSort.direction === "asc");
    byId("evidence-sort-direction").click();
    set("evidence-sort-column", "Checked");
    check("Evidence page: shows the time to first token of the current selection", firstTokenStrip("evidence-page"));
  });

  // ------------------------------------------------------------- knowledge
  await section("Know more page", async () => {
    app.showPage("knowledge");
    check("Know more page: shows the time to first token of the current selection", firstTokenStrip("knowledge-page"));
    const stripBefore = byId("knowledge-page").querySelector("[data-ttft-strip]").textContent;
    set("prompt-length-input", 8192);
    check("Know more page: the first-token time follows the Model → Hardware settings", byId("knowledge-page").querySelector("[data-ttft-strip]").textContent !== stripBefore && /8,192 prompt tokens/.test(byId("knowledge-page").querySelector("[data-ttft-strip]").textContent));
    set("prompt-length-input", 2048);
    for (const tab of all(".knowledge-nav button")) {
      tab.click();
      const panel = all("[data-knowledge-page-content]").find((view) => view.dataset.knowledgePageContent === tab.dataset.knowledgePage);
      check(`Know more page: ${tab.textContent.trim()} tab shows its panel`, panel && !panel.hidden && all("[data-knowledge-page-content]").filter((view) => !view.hidden).length === 1);
    }
    const guide = document.querySelector('[data-knowledge-page-content="ttft"]');
    const worked = byId("ttft-worked");
    const workedTotal = () => number(worked.querySelector(".ttft-worked-total b").textContent);
    const fastest = () => gpuRows().filter((row) => row.fits && row.ttft !== null).sort((a, b) => a.ttft - b.ttft)[0];
    all(".knowledge-nav button").find((tab) => tab.dataset.knowledgePage === "ttft").click();
    check("Know more page: Time to first token states the formula and its constants", /prompt tokens ÷ prefill tok\/s \+ 1 ÷ decode tok\/s/.test(guide.textContent) && guide.textContent.includes(`${MF.PREFILL_GFLOPS_PER_B} × active parameters`) && guide.textContent.includes(`${MF.PREFILL_MFU} × TFLOPS × 1000`) && guide.textContent.includes(`× ${MF.CPU_PREFILL_FACTOR}`) && guide.textContent.includes(`${MF.EXTRA_CARD_SCALE} × (cards`));
    let best = fastest();
    check("Know more page: the worked example is the fastest accelerator that fits", Boolean(best) && worked.textContent.includes(best.name) && worked.textContent.includes(state.model.Model) && near(workedTotal(), best.ttft, Math.max(0.006, best.ttft * 0.006)));
    const byHand = state.promptTokens / (MF.PREFILL_MFU * app.tflopsFor("GPUs", best.value) * 1000 / (MF.PREFILL_GFLOPS_PER_B * state.model["Active params (B)"]) * MF.quantityScale(state.gpuQuantity) / state.users) + 1 / best.speed;
    check("Know more page: the worked example equals the formula done by hand", best.gpuShare !== null || near(byHand, best.ttft, 1e-9), `${byHand} vs ${best.ttft}`);
    const totalBefore = workedTotal();
    set("prompt-length-input", 8192);
    check("Know more page: the worked example follows the prompt length", /8,192 prompt tokens/.test(worked.textContent) && workedTotal() > totalBefore * 3);
    set("prompt-length-input", 2048);
    app.showPage("database");
    byId("database-page").querySelector("[data-open-ttft-guide]").click();
    check("Know more page: How this is calculated opens the first-token guide", state.page === "knowledge" && !guide.hidden && all("[data-knowledge-page-content]").filter((view) => !view.hidden).length === 1);
    all(".knowledge-nav button").find((tab) => tab.dataset.knowledgePage === "latest").click();
    const cards = all("#knowledge-latest-models .knowledge-model");
    check("Know more page: newest models are listed from the catalog", cards.length > 0 && cards.length === app.latestModels().length);
    cards[0].querySelector("button").click();
    check("Know more page: a model card opens its details", byId("evidence-drawer").classList.contains("open"));
    byId("drawer-scrim").click();
    check("Know more page: clicking outside closes the drawer", !byId("evidence-drawer").classList.contains("open"));
    all(".knowledge-nav button").find((tab) => tab.dataset.knowledgePage === "learning").click();
    const terms = () => all(".knowledge-glossary dl > div:not(.empty-state)").filter((entry) => !entry.hidden).length;
    const everyTerm = terms();
    set("glossary-search", "kv");
    check("Know more page: glossary filter narrows the terms", terms() > 0 && terms() < everyTerm);
    set("glossary-search", "zzzz-no-term");
    check("Know more page: an empty glossary result explains itself", terms() === 0 && !document.querySelector(".knowledge-glossary .empty-state").hidden);
    set("glossary-search", "");
    check("Know more page: clearing the filter restores every term", terms() === everyTerm);
    check("Know more page: time to first token is explained in the guide and the glossary", all(".knowledge-glossary dt").some((term) => /Time to first token/.test(term.textContent)) && all(".learning-formula").some((formula) => /time to first token/.test(formula.textContent)));
    all(".knowledge-nav button").find((tab) => tab.dataset.knowledgePage === "evidence").click();
    document.querySelector('[data-open-page="evidence"]').click();
    check("Know more page: Open evidence ledger goes to the Evidence page", state.page === "evidence" && !byId("evidence-page").hidden);
    byId("open-about").click();
    check("About dialog: opens", byId("about-dialog").open);
    byId("close-about").click();
    check("About dialog: closes", !byId("about-dialog").open);
  });

  // ---------------------------------------------------------------- themes
  function parseColor(text) {
    let match = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/.exec(text);
    if (match) return { r: +match[1], g: +match[2], b: +match[3], a: match[4] === undefined ? 1 : match[4].endsWith("%") ? parseFloat(match[4]) / 100 : +match[4] };
    match = /^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+%?))?\)$/.exec(text);
    if (match) return { r: match[1] * 255, g: match[2] * 255, b: match[3] * 255, a: match[4] === undefined ? 1 : match[4].endsWith("%") ? parseFloat(match[4]) / 100 : +match[4] };
    return null;
  }
  const over = (top, bottom) => ({ r: top.r * top.a + bottom.r * (1 - top.a), g: top.g * top.a + bottom.g * (1 - top.a), b: top.b * top.a + bottom.b * (1 - top.a), a: 1 });
  function luminance(color) {
    const channel = (value) => (value / 255 <= 0.03928 ? value / 255 / 12.92 : ((value / 255 + 0.055) / 1.055) ** 2.4);
    return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
  }
  function contrastOf(element, page) {
    const layers = [];
    for (let node = element; node; node = node.parentElement) {
      const color = parseColor(getComputedStyle(node).backgroundColor);
      if (color && color.a > 0) {
        layers.push(color);
        if (color.a >= 0.999) break;
      }
    }
    let background = page;
    for (const layer of layers.reverse()) background = over(layer, background);
    const ink = parseColor(getComputedStyle(element).color);
    if (!ink) return null;
    const text = over(ink, background);
    const [high, low] = [luminance(text), luminance(background)].sort((a, b) => b - a);
    return (high + 0.05) / (low + 0.05);
  }
  function hasOwnText(element) {
    return [...element.childNodes].some((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim().length > 1);
  }
  function describe(element) {
    return `${element.tagName.toLowerCase()}${element.id ? `#${element.id}` : ""}${element.className && typeof element.className === "string" ? `.${element.className.trim().split(/\s+/).join(".")}` : ""} "${element.textContent.trim().slice(0, 28)}"`;
  }

  const pages = ["model", "hardware", "pipeline", "database", "evidence", "knowledge"];
  const originalChoice = theme.choice;
  const colors = {};
  for (const name of ["light", "dark"]) {
    await section(`${name} theme`, async () => {
      theme.set(name);
      await tick();
      const root = document.documentElement;
      const pageColor = parseColor(getComputedStyle(document.body).backgroundColor);
      colors[name] = getComputedStyle(document.body).backgroundColor;
      check(`${name} theme: applied to the page`, root.dataset.theme === name && getComputedStyle(root).colorScheme.includes(name));
      check(`${name} theme: page background is ${name}`, name === "dark" ? luminance(pageColor) < 0.05 : luminance(pageColor) > 0.7, colors[name]);
      check(`${name} theme: browser toolbar colour follows the page`, document.querySelector('meta[name="theme-color"]').content.length > 3);
      const views = [...pages.map((page) => ({ page, label: `${page} page` })), { page: "knowledge", topic: "ttft", label: "first-token guide" }];
      for (const { page, topic, label } of views) {
        app.showPage(page);
        if (page === "knowledge") all(".knowledge-nav button").find((tab) => tab.dataset.knowledgePage === (topic || "model360")).click();
        await tick();
        const scope = [document.querySelector(".site-header"), byId(`${page}-page`), ...(page === "model" ? [byId("method")] : [])];
        const candidates = scope.flatMap((host) => all("*", host)).filter((element) => hasOwnText(element) && element.getClientRects().length && !element.closest("option, [hidden]") && !element.disabled).slice(0, 900);
        const low = new Map();
        let borderline = 0;
        for (const element of candidates) {
          const ratio = contrastOf(element, pageColor);
          if (ratio === null) continue;
          const kind = `${element.tagName.toLowerCase()}.${typeof element.className === "string" ? element.className.trim().split(/\s+/).join(".") : ""}`;
          if (ratio < 3 && !low.has(kind)) low.set(kind, `${describe(element)} ${ratio.toFixed(2)}:1`);
          else if (ratio >= 3 && ratio < 4.5) borderline += 1;
        }
        check(`${name} theme: ${label} text is readable (${candidates.length} elements)`, low.size === 0, [...low.values()].slice(0, 8).join(" | "));
        if (borderline) note(`${name} theme: ${label}`, `${borderline} of ${candidates.length} text elements sit between 3:1 and 4.5:1 contrast`);
        check(`${name} theme: ${label} does not scroll sideways`, document.documentElement.scrollWidth <= window.innerWidth + 1, `${document.documentElement.scrollWidth} > ${window.innerWidth}`);
        // Narrower than any phone (for example a small editor preview pane): layout is not judged.
        if (window.innerWidth < 300) continue;
        const cut = scope.flatMap((host) => all("*", host)).filter((element) => element.getClientRects().length && element.getBoundingClientRect().right > window.innerWidth + 1 && !element.closest(".data-table-wrap, .header-nav, .knowledge-nav, [hidden]"));
        check(`${name} theme: ${label} has nothing cut off at ${window.innerWidth}px wide`, cut.length === 0, cut.slice(0, 4).map(describe).join(" | "));
      }
      all(".knowledge-nav button")[0].click();
      app.showPage("model");
      byId("show-model-evidence").click();
      await tick();
      const drawerText = all("*", byId("evidence-drawer")).filter((element) => hasOwnText(element) && element.getClientRects().length && !element.closest("[hidden]"));
      const lowDrawer = drawerText.map((element) => [element, contrastOf(element, pageColor)]).filter(([, ratio]) => ratio !== null && ratio < 3).map(([element, ratio]) => `${describe(element)} ${ratio.toFixed(2)}:1`);
      check(`${name} theme: detail drawer text is readable (${drawerText.length} elements)`, lowDrawer.length === 0, lowDrawer.slice(0, 4).join(" | "));
      byId("close-drawer").click();
      byId("open-about").click();
      const dialogText = all("*", byId("about-dialog")).filter((element) => hasOwnText(element));
      const lowDialog = dialogText.map((element) => [element, contrastOf(element, pageColor)]).filter(([, ratio]) => ratio !== null && ratio < 3).map(([element, ratio]) => `${describe(element)} ${ratio.toFixed(2)}:1`);
      check(`${name} theme: about dialog text is readable`, lowDialog.length === 0, lowDialog.join(" | "));
      byId("close-about").click();
    });
  }
  await section("Theme switch", async () => {
    check("Theme switch: light and dark use different page colours", colors.light !== colors.dark, `${colors.light} / ${colors.dark}`);
    theme.set("system");
    const seen = [];
    for (let step = 0; step < 3; step += 1) {
      byId("theme-toggle").click();
      seen.push(theme.choice);
    }
    check("Theme switch: the header button cycles light, dark, match system", seen.join(",") === "light,dark,system", seen.join(","));
    theme.set("dark");
    let saved = null;
    try { saved = localStorage.getItem("modelfit-theme"); } catch { saved = "blocked"; }
    if (saved === "blocked") note("Theme switch", "Storage is blocked in this browser, so the choice lasts for one visit only.");
    else check("Theme switch: the choice is remembered", saved === "dark");
    check("Theme switch: the button names the current theme", /Dark/.test(byId("theme-toggle").getAttribute("aria-label")));
    theme.set(originalChoice);
  });

  // ---------------------------------------------------------------- report
  // Optional, for screenshots: ?selftest=1&show=knowledge&topic=ttft&theme=dark leaves that page, Know more topic and theme on screen.
  const params = new URLSearchParams(location.search);
  app.showPage(pages.includes(params.get("show")) ? params.get("show") : "model");
  const topicTab = all(".knowledge-nav button").find((tab) => tab.dataset.knowledgePage === params.get("topic"));
  if (topicTab) topicTab.click();
  if (["light", "dark"].includes(params.get("theme"))) theme.set(params.get("theme"));
  const pass = results.filter((row) => row.status === "pass").length;
  const fail = results.filter((row) => row.status === "fail").length;
  const notes = results.filter((row) => row.status === "note").length;
  window.__selftest = { done: true, pass, fail, notes, results };
  document.title = `Self test: ${pass} passed, ${fail} failed`;
  const panel = document.createElement("section");
  panel.className = "selftest-panel";
  panel.id = "selftest-panel";
  panel.setAttribute("aria-label", "Self test results");
  const escape = (text) => String(text).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
  const list = (status) => results.filter((row) => row.status === status).map((row) => `<li><b>${escape(row.name)}</b>${row.detail ? `<span>${escape(row.detail)}</span>` : ""}</li>`).join("");
  panel.innerHTML = `<h2>Self test: ${pass} passed, ${fail} failed${notes ? `, ${notes} notes` : ""}</h2><p>Every page and control was exercised in the light and the dark theme. Reload without <code>?selftest=1</code> to return to the app.</p>${fail ? `<h3>Failed</h3><ul class="selftest-fail">${list("fail")}</ul>` : ""}${notes ? `<h3>Notes</h3><ul>${list("note")}</ul>` : ""}<details><summary>All ${pass} passed checks</summary><ul>${list("pass")}</ul></details>`;
  document.querySelector("main").append(panel);
})();
