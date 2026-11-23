/* Runs docs/formulas.js over a case file written by scripts/reverify_all.py and prints the results as JSON.
   Usage: node scripts/formula_harness.js <cases.json> */
const fs = require("fs");
const path = require("path");
const MF = require(path.join(__dirname, "..", "docs", "formulas.js"));

const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const table = MF.buildEfficiencyTable(input.libraries);
const rules = MF.buildTaskRules(input.taskRules);
const flag = (value) => (value === null || value === undefined ? null : value ? 1 : 0);

function pipeline(result) {
  return [
    result.usableGb, result.effectiveBwGbps, result.voiceWeightsGb, result.voiceExtraGb, result.voiceMemoryGb, result.voiceBwGbps, result.voicePending,
    result.memoryUsedGb, result.memoryMarginGb, result.bwRequiredGbps, result.llmsUsed, result.llmsFit, flag(result.allFit), result.verdict,
    result.setMemoryGb, result.setBwGbps, result.setsByMemory, result.setsByBandwidth, result.sets, result.setsLimit, result.ttftWorstLoaded,
    result.rawBwGbps, result.voiceRawBwGbps, result.setRawBwGbps,
    ...result.voice.flatMap((row) => [row.placement, row.weightsGb, row.extraGb, row.bwGbps, row.status, row.utilization, row.rawBwGbps, row.kvBits]),
    ...result.llms.flatMap((row) => [row.status, row.weights, row.kvOverhead, row.total, row.bwNeeded, row.cumulativeMemory, row.cumulativeBw, flag(row.fitsAlone), row.tokAlone, row.target, row.prefillAlone, row.ttftAlone, row.ttftLoaded, row.utilization, row.kvBits, row.rawBwNeeded, row.cumulativeRawBw]),
  ];
}

function run(testCase) {
  const a = testCase.args;
  switch (testCase.fn) {
    case "kvPerTokenGb": return [MF.kvPerTokenGb(a.model, a.kvBits)];
    case "memoryNeed": { const r = MF.memoryNeed(a.model, a); return [r.weights, r.kv, r.runtime, r.total]; }
    case "excelMemoryNeed": { const r = MF.excelMemoryNeed(a.model, a); return [r.weights, r.kv, r.runtime, r.total]; }
    case "quantityScale": return [MF.quantityScale(a.quantity), MF.excelQuantityScale(a.quantity)];
    case "decodeTokPerSecond": return [MF.decodeTokPerSecond(a)];
    case "excelDecodeTokPerSecond": return [MF.excelDecodeTokPerSecond(a.model, a)];
    case "splitDecodeTokPerSecond": return [MF.splitDecodeTokPerSecond(a)];
    case "prefillTokPerSecond": return [MF.prefillTokPerSecond(a)];
    case "splitPrefillTokPerSecond": return [MF.splitPrefillTokPerSecond(a)];
    case "timeToFirstToken": return [MF.timeToFirstToken(a)];
    case "gpuUsableGb": return [MF.gpuUsableGb(a.memoryGb, a.quantity, a.headroomPct)];
    case "cpuUsableGb": return [MF.cpuUsableGb(a.ramGb)];
    case "hostOffloadGb": return [MF.hostOffloadGb(a.ramGb, a.platformMaxGb)];
    case "offloadSupported": return [flag(MF.offloadSupported(a.runtime))];
    case "isUnifiedMemory": return [flag(MF.isUnifiedMemory(a.gpu))];
    case "parseEfficiency": return [MF.parseEfficiency(a.cell)];
    case "hardwareClass": return [MF.hardwareClass(a.row, a.kind)];
    case "gpuRuntimeEfficiency": { const r = MF.gpuRuntimeEfficiency(table, a.runtime, a.gpu); return [r.efficiency, r.backend]; }
    case "cpuRuntimeEfficiency": return [MF.cpuRuntimeEfficiency(table, a.runtime, a.cpu)];
    case "requiredBits": { const rule = MF.taskRule(rules, a.task); return [MF.requiredBits(a.model, rule), flag(MF.taskAllows(a.model, rule, a.bits)), flag(MF.isVlm(a.model))]; }
    case "hasSizingSpecs": return [flag(MF.hasSizingSpecs(a.model))];
    case "voiceWeightsGb": return [MF.voiceWeightsGb(a.row)];
    case "pipelineFit": return pipeline(MF.pipelineFit(a));
    default: throw new Error(`Unknown case ${testCase.fn}`);
  }
}

process.stdout.write(JSON.stringify({
  constants: {
    WEIGHT_MARGIN: MF.WEIGHT_MARGIN, RUNTIME_SHARE: MF.RUNTIME_SHARE, DEFAULT_HEADROOM_PCT: MF.DEFAULT_HEADROOM_PCT,
    EXTRA_CARD_SCALE: MF.EXTRA_CARD_SCALE, HOST_RESERVE_GB: MF.HOST_RESERVE_GB, VOICE_EXTRA_GB: MF.VOICE_EXTRA_GB, VOICE_BW_GBPS: MF.VOICE_BW_GBPS, VOICE_RESERVE_BITS: MF.VOICE_RESERVE_BITS,
    PREFILL_MFU: MF.PREFILL_MFU, CPU_PREFILL_FACTOR: MF.CPU_PREFILL_FACTOR, PREFILL_GFLOPS_PER_B: MF.PREFILL_GFLOPS_PER_B,
    OFFLOAD_RUNTIMES: MF.OFFLOAD_RUNTIMES, HW_CLASSES: MF.HW_CLASSES, DEFAULT_TASK_RULES: MF.DEFAULT_TASK_RULES,
  },
  results: input.cases.map(run),
}));
