"""Quantify the web/CLI calculator against the workbook's "1 Plan a Build" formulas.

Both sides come from scripts/formulas.py, so this reports only the intended differences:
  memory  app = 3% weight margin + 7% of weights for execution;
          workbook = raw weights + max(1 GB, 10% of weights+KV), context capped at the model limit
  decode  app reads the active quantized weights per token;
          workbook adds the whole KV allocation to that denominator
Both use the same runtime efficiency (llama.cpp row of data/libraries.csv) and 8-bit KV.
Hardware the runtime cannot use is skipped.
"""
import json
import statistics
from common import load_rows
import formulas as F

RUNTIME = "llama.cpp"
KV_BITS = 8


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def stats(values):
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p10": round(percentile(values, 0.1), 2),
        "p90": round(percentile(values, 0.9), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def memory(model, bits, context, users, excel=False):
    need = F.excel_memory_need(model, bits, context, users, KV_BITS) if excel else F.memory_need(model, bits, context, users, KV_BITS)
    return need["total"]


def speed(model, hardware, efficiency, bits, context, users, cpu=False, excel=False):
    bandwidth = hardware["Memory BW (GB/s)" if cpu else "Bandwidth (GB/s)"]
    if excel:
        return F.excel_decode_tok_per_second(model, bits, context, efficiency, bandwidth, users, 1, KV_BITS)
    return F.decode_tok_per_second(model["Active params (B)"], bits, efficiency, bandwidth, users)


def calculate():
    models = [m for m in load_rows("models") if all(m.get(k) is not None for k in ("Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim"))]
    table = F.build_efficiency_table(load_rows("libraries"))
    gpus = [(g, F.gpu_runtime_efficiency(table, RUNTIME, g)["efficiency"]) for g in load_rows("gpus") if g.get("Bandwidth (GB/s)") is not None]
    cpus = [(c, F.cpu_runtime_efficiency(table, RUNTIME, c)) for c in load_rows("cpus") if c.get("Memory BW (GB/s)") is not None]
    gpus = [(g, eff) for g, eff in gpus if eff > 0]
    cpus = [(c, eff) for c, eff in cpus if eff > 0]
    quants = {q["Quantization"]: q["Bits/weight (eff.)"] for q in load_rows("quant")}
    bits = next(value for name, value in quants.items() if "Q4_K_M" in name)
    result = {"basis": f"Q4_K_M, {KV_BITS}-bit KV, 1 user, {RUNTIME} efficiency; positive delta means the app estimate is higher than the workbook Plan-a-Build formula"}
    for context in (8192, 32768):
        memory_deltas = []
        gpu_deltas = []
        cpu_deltas = []
        for model in models:
            web_memory = memory(model, bits, context, 1)
            excel_memory = memory(model, bits, context, 1, excel=True)
            memory_deltas.append((web_memory / excel_memory - 1) * 100)
            for gpu, efficiency in gpus:
                web = speed(model, gpu, efficiency, bits, context, 1)
                old = speed(model, gpu, efficiency, bits, context, 1, excel=True)
                gpu_deltas.append((web / old - 1) * 100)
            for cpu, efficiency in cpus:
                web = speed(model, cpu, efficiency, bits, context, 1, cpu=True)
                old = speed(model, cpu, efficiency, bits, context, 1, cpu=True, excel=True)
                cpu_deltas.append((web / old - 1) * 100)
        result[f"{context // 1024}k"] = {
            "memory_delta_pct": stats(memory_deltas),
            "gpu_decode_delta_pct": stats(gpu_deltas),
            "cpu_decode_delta_pct": stats(cpu_deltas),
        }
    result["examples"] = []
    for name in ("Qwen3 32B", "Llama 3.3 70B", "Qwen3 30B-A3B"):
        model = next(m for m in models if m["Model"] == name)
        web = memory(model, bits, 8192, 1)
        excel = memory(model, bits, 8192, 1, excel=True)
        result["examples"].append({"model": name, "web_memory_gb": round(web, 2), "excel_memory_gb": round(excel, 2), "delta_pct": round((web / excel - 1) * 100, 2)})
    return result


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2))
