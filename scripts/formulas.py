"""Model Fit sizing formulas for the CLI and the checks.

Python mirror of docs/formulas.js. scripts/reverify_all.py runs both against the same
inputs and fails if they disagree, so change the two files together.

Three profiles are kept apart on purpose:
  app      memory_need / decode_tok_per_second / split_decode_tok_per_second (web app and CLI)
  workbook excel_memory_need / excel_decode_tok_per_second ("1 Plan a Build" comparison)
  pipeline pipeline_fit ("Multi-Stage Pipeline Fit" sheet)
"""
import math
import re

WEIGHT_MARGIN = 1.03
RUNTIME_SHARE = 0.07
DEFAULT_HEADROOM_PCT = 8
EXTRA_CARD_SCALE = 0.8
HOST_RESERVE_GB = 8
VOICE_EXTRA_GB = 5
VOICE_BW_GBPS = 5
OFFLOAD_RUNTIMES = ("llama.cpp", "Ollama", "LM Studio", "IPEX-LLM")
HW_CLASSES = ("NVIDIA GPU", "AMD GPU", "Intel GPU / iGPU", "NPU (Intel/AMD)", "Apple Silicon", "x86 CPU", "ARM CPU",
              "Jetson (ARM+GPU)", "Qualcomm / Rockchip NPU")
VENDOR_CLASS = {"NVIDIA": "NVIDIA GPU", "AMD": "AMD GPU", "Intel": "Intel GPU / iGPU", "Apple": "Apple Silicon"}
SIZING_KEYS = ("Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim")
DEFAULT_TASK = "General chat / Q&A"


def num(value):
    if value is None or value == "":
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def positive(value):
    parsed = num(value)
    return parsed if parsed > 0 else 0.0


def has_sizing_specs(model):
    return all(positive((model or {}).get(key)) > 0 for key in SIZING_KEYS)


def kv_per_token_gb(model, kv_bits=8):
    bits = positive(kv_bits) or 8
    return 2 * positive(model.get("Layers")) * positive(model.get("KV heads")) * positive(model.get("Head dim")) * bits / 8 / 1e9


def memory_need(model, bits, context, users=1, kv_bits=8):
    users = positive(users) or 1
    weights = positive(model.get("Total params (B)")) * positive(bits) / 8 * WEIGHT_MARGIN
    kv = kv_per_token_gb(model, kv_bits) * positive(context) * users
    runtime = weights * RUNTIME_SHARE
    return {"weights": weights, "kv": kv, "runtime": runtime, "total": weights + kv + runtime}


def excel_memory_need(model, bits, context, users=1, kv_bits=8):
    users = positive(users) or 1
    limit = positive(model.get("Max context"))
    context = min(positive(context), limit) if limit else positive(context)
    weights = positive(model.get("Total params (B)")) * positive(bits) / 8
    kv = kv_per_token_gb(model, kv_bits) * context * users
    runtime = max(1, 0.1 * (weights + kv))
    return {"weights": weights, "kv": kv, "runtime": runtime, "total": weights + kv + runtime}


def quantity_scale(quantity=1):
    count = max(1, positive(quantity) or 1)
    return 1 + EXTRA_CARD_SCALE * (count - 1)


def excel_quantity_scale(quantity=1):
    count = max(1, positive(quantity) or 1)
    return min(count, 8) * (EXTRA_CARD_SCALE if count > 1 else 1)


def decode_tok_per_second(active_params_b, bits, efficiency, bandwidth, users=1, quantity=1, utilization=1):
    per_token = positive(active_params_b) * positive(bits) / 8
    efficiency = positive(efficiency)
    if not per_token or not efficiency:
        return 0.0
    return efficiency * positive(bandwidth) * quantity_scale(quantity) * positive(utilization) / per_token / (positive(users) or 1)


def excel_decode_tok_per_second(model, bits, context, efficiency, bandwidth, users=1, quantity=1, kv_bits=8):
    efficiency = positive(efficiency)
    per_token = positive(model.get("Active params (B)")) * positive(bits) / 8 + excel_memory_need(model, bits, context, users, kv_bits)["kv"]
    if not per_token or not efficiency:
        return 0.0
    return efficiency * positive(bandwidth) * excel_quantity_scale(quantity) / per_token / (positive(users) or 1)


def split_decode_tok_per_second(per_token_gb, gpu_share, gpu_rate, cpu_rate, users=1):
    per_token = positive(per_token_gb)
    if not per_token:
        return 0.0
    share = num(gpu_share)
    share = min(1.0, max(0.0, 0.0 if math.isnan(share) else share))
    gpu_rate, cpu_rate = positive(gpu_rate), positive(cpu_rate)
    if (share > 0 and not gpu_rate) or (share < 1 and not cpu_rate):
        return 0.0
    seconds = per_token * ((share / gpu_rate if share > 0 else 0) + ((1 - share) / cpu_rate if share < 1 else 0))
    return 1 / seconds / (positive(users) or 1) if seconds > 0 else 0.0


def is_unified_memory(gpu):
    return "unified" in str((gpu or {}).get("Interface") or "").lower()


def gpu_usable_gb(memory_gb, quantity=1, headroom_pct=None):
    headroom = DEFAULT_HEADROOM_PCT if headroom_pct is None else max(0.0, 0.0 if math.isnan(num(headroom_pct)) else num(headroom_pct))
    return positive(memory_gb) * (1 - headroom / 100) * max(1, positive(quantity) or 1)


def cpu_usable_gb(ram_gb):
    ram = positive(ram_gb)
    return ram - min(HOST_RESERVE_GB, ram * 0.25)


def offload_supported(runtime):
    return str(runtime or "") in OFFLOAD_RUNTIMES


def host_offload_gb(ram_gb, platform_max_gb=None):
    ram, limit = positive(ram_gb), positive(platform_max_gb)
    return max(0.0, (min(ram, limit) if limit else ram) - HOST_RESERVE_GB)


def parse_efficiency(cell):
    match = re.match(r"\d+(?:\.\d+)?", str("" if cell is None else cell).strip())
    return float(match.group(0)) if match else 0.0


def build_efficiency_table(libraries):
    table = {}
    for row in libraries or []:
        name = row.get("Library")
        if not name:
            continue
        table[name] = {
            "classes": {hw: parse_efficiency(row.get(hw)) for hw in HW_CLASSES},
            "cells": {hw: str("" if row.get(hw) is None else row.get(hw)) for hw in HW_CLASSES},
            "typical": parse_efficiency(row.get("Typical eff.")),
            "vulkan": parse_efficiency(row.get("Vulkan fallback eff.")),
            "min_cuda": num(row.get("Min CUDA compute capability")),
        }
    return table


def hardware_class(row, kind="gpu"):
    declared = str((row or {}).get("Runtime HW class") or "").strip()
    if declared in HW_CLASSES:
        return declared
    if kind == "cpu":
        return "Jetson (ARM+GPU)" if "Jetson" in str((row or {}).get("CPU / platform") or "") else "x86 CPU"
    return VENDOR_CLASS.get((row or {}).get("Vendor"), "")


def gpu_runtime_efficiency(table, runtime, gpu):
    """Native cell when the backend is usable, otherwise the runtime's Vulkan fallback (never on Apple)."""
    entry = (table or {}).get(runtime)
    if not entry:
        return {"efficiency": 0.0, "backend": "unsupported"}
    hw_class = hardware_class(gpu, "gpu")
    native = entry["classes"].get(hw_class, 0.0)
    native_ok = True
    vendor = gpu.get("Vendor")
    if vendor == "NVIDIA":
        cc = num(gpu.get("CUDA compute capability"))
        native_ok = not math.isnan(cc) and not math.isnan(entry["min_cuda"]) and cc >= entry["min_cuda"]
    elif vendor == "AMD":
        native_ok = str(gpu.get("ROCm support") or "") != "N" or "vulkan" in entry["cells"].get(hw_class, "").lower()
    if native_ok and native > 0:
        return {"efficiency": native, "backend": "native"}
    if vendor == "Apple" or not entry["vulkan"]:
        return {"efficiency": 0.0, "backend": "unsupported"}
    return {"efficiency": entry["vulkan"], "backend": "vulkan"}


def cpu_runtime_efficiency(table, runtime, cpu):
    entry = (table or {}).get(runtime)
    return entry["classes"].get(hardware_class(cpu, "cpu"), 0.0) if entry else 0.0


def build_task_rules(rows):
    rules = {}
    for row in rows or []:
        task = row.get("Task")
        if not task:
            continue
        rules[task] = {
            "task": task,
            "base": positive(row.get("Base min bits (4-30B)")),
            "small": positive(row.get("Small model <4B min bits")),
            "large": positive(row.get("Large model >=30B min bits")),
            "vlm": str(row.get("VLM required?") or "N").upper() == "Y",
            "why": row.get("Why") or "",
            "verdict": row.get("Verdict label if too low") or "",
        }
    return rules


def task_rule(rules, task):
    rules = rules or {}
    return rules.get(task) or rules.get(DEFAULT_TASK) or {"task": DEFAULT_TASK, "base": 4.85, "small": 6.6, "large": 4.85, "vlm": False, "why": "", "verdict": ""}


def required_bits(model, rule):
    total = positive(model.get("Total params (B)"))
    return rule["small"] if total < 4 else rule["large"] if total >= 30 else rule["base"]


def is_vlm(model):
    return bool(re.search(r"VLM|Vision", str((model or {}).get("Type") or ""), re.I))


def task_allows(model, rule, bits):
    return positive(bits) >= required_bits(model, rule) and (not rule["vlm"] or is_vlm(model))


def voice_weights_gb(row):
    params, bits = num((row or {}).get("Total params (B)")), num((row or {}).get("Bits / weight"))
    return params * bits / 8 if params > 0 and bits > 0 else None


def pipeline_fit(memory_gb, bandwidth_gbps, utilization, voice=(), llms=(), quantity=1, headroom_pct=10, kv_bits=16, activity="simultaneous"):
    """Voice stages reserve first; LLM slots are then checked cumulatively in order."""
    quantity = max(1, positive(quantity) or 1)
    headroom = max(0.0, 0.0 if math.isnan(num(headroom_pct)) else num(headroom_pct))
    usable = positive(memory_gb) * quantity * (1 - headroom / 100)
    effective_bw = positive(bandwidth_gbps) * quantity_scale(quantity) * positive(utilization)
    kv_bits = positive(kv_bits) or 16

    voice_weights = voice_extra = voice_bw = 0.0
    pending = 0
    voice_rows = []
    for stage in voice:
        on_gpu = stage.get("placement") != "CPU"
        raw = num(stage.get("weightsGb"))
        weights = (None if math.isnan(raw) else raw) if on_gpu else 0.0
        extra = max(0.0, positive(stage.get("extraGb"))) if on_gpu else 0.0
        bw = max(0.0, positive(stage.get("bwGbps"))) if on_gpu else 0.0
        status = "CPU - not sized" if not on_gpu else "WEIGHTS PENDING" if weights is None else "INPUTS SET"
        pending += status == "WEIGHTS PENDING"
        voice_weights += weights or 0.0
        voice_extra += extra
        voice_bw += bw
        voice_rows.append({"stage": stage.get("stage"), "name": stage.get("name"), "placement": "GPU" if on_gpu else "CPU",
                           "weightsGb": weights, "extraGb": extra, "bwGbps": bw, "status": status})
    voice_memory = voice_weights + voice_extra

    memory = voice_memory
    bw_sum = bw_max = 0.0
    fits = used = 0
    rows = []
    sequential = activity == "sequential"
    for index, slot in enumerate(llms, start=1):
        model = slot.get("model")
        if not model:
            rows.append({"slot": index, "model": None, "status": "UNUSED", "weights": 0.0, "kvOverhead": 0.0, "total": 0.0, "bwNeeded": 0.0,
                         "cumulativeMemory": memory, "cumulativeBw": voice_bw + (bw_max if sequential else bw_sum), "fitsAlone": None, "tokAlone": None})
            continue
        used += 1
        bits, context, target = positive(slot.get("bits")), positive(slot.get("context")), positive(slot.get("target"))
        weights = positive(model.get("Total params (B)")) * bits / 8 * WEIGHT_MARGIN
        kv_overhead = kv_per_token_gb(model, kv_bits) * context + weights * RUNTIME_SHARE
        total = weights + kv_overhead
        per_token = positive(model.get("Active params (B)")) * bits / 8 * WEIGHT_MARGIN
        bw_needed = per_token * target
        memory += total
        bw_sum += bw_needed
        bw_max = max(bw_max, bw_needed)
        cumulative_bw = voice_bw + (bw_max if sequential else bw_sum)
        limit = positive(model.get("Max context"))
        status = "FITS"
        if not bits or not context or not target:
            status = "REVIEW: CHECK INPUTS"
        elif not has_sizing_specs(model):
            status = "REVIEW: MODEL SPECS MISSING"
        elif limit and context > limit:
            status = "REVIEW: CONTEXT > MODEL LIMIT"
        elif memory > usable:
            status = "NO - GPU MEMORY"
        elif cumulative_bw > effective_bw:
            status = "NO - BANDWIDTH"
        review = status.startswith("REVIEW")
        fits += status == "FITS"
        rows.append({"slot": index, "model": model.get("Model"), "status": status, "weights": weights, "kvOverhead": kv_overhead, "total": total,
                     "bwNeeded": bw_needed, "cumulativeMemory": memory, "cumulativeBw": cumulative_bw,
                     "fitsAlone": None if review else voice_memory + total <= usable,
                     "tokAlone": None if review or not per_token else max(0.0, effective_bw - voice_bw) / per_token})
    reviews = sum(row["status"].startswith("REVIEW") for row in rows)
    verdict = "INPUTS SET"
    if usable <= 0 or effective_bw <= 0:
        verdict = "REVIEW: select accelerator, quantity, headroom and utilization"
    elif reviews:
        verdict = "REVIEW: fix LLM inputs"
    return {"usableGb": usable, "effectiveBwGbps": effective_bw, "voice": voice_rows, "voiceWeightsGb": voice_weights, "voiceExtraGb": voice_extra,
            "voiceMemoryGb": voice_memory, "voiceBwGbps": voice_bw, "voicePending": pending, "llms": rows, "memoryUsedGb": memory,
            "memoryMarginGb": usable - memory, "bwRequiredGbps": voice_bw + (bw_max if sequential else bw_sum), "llmsUsed": used, "llmsFit": fits,
            "allFit": verdict == "INPUTS SET" and used > 0 and fits == used, "verdict": verdict}
