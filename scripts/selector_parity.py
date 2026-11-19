"""Quantify the public web/CLI calculator against the Excel workbook formulas.

Reports memory and decode deltas for representative public workloads. App memory:
3% weight margin + 7% weight execution overhead, with 8-bit KV by default.
Excel Plan-a-Build memory: no weight overhead + max(1, 10% of weights+KV) runtime.
The app decode path uses active quantized weights; Excel adds the full allocated
KV cache to the same traffic denominator. Both use runtime-specific efficiency.
"""
import json
import statistics
from common import load_rows


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


def kv_token(model, kv_bits=8):
    return 2 * model["Layers"] * model["KV heads"] * model["Head dim"] * kv_bits / 8 / 1e9


def memory(model, bits, context, users, excel=False):
    weights = model["Total params (B)"] * bits / 8
    kv = kv_token(model, 8) * context * users
    if excel:
        runtime = max(1, 0.1 * (weights + kv))
        return weights + kv + runtime
    return weights * 1.03 + kv + weights * 0.07


def web_efficiency(vendor, cpu=False):
    if cpu or vendor == "Intel":
        return 0.55
    return 0.65 if vendor == "Apple" else 0.60


def excel_efficiency(hardware, cpu=False):
    if cpu:
        name = hardware["CPU / platform"]
        if name.startswith("Mac ") or name.startswith("Apple M4"):
            return 0.65
        if "Jetson" in name:
            return 0.60
        return 0.55
    if hardware["Vendor"] == "Intel":
        return 0.50
    if hardware["Vendor"] == "Apple":
        return 0.65
    if hardware["Vendor"] == "NVIDIA":
        return 0.45 if hardware["Accelerator"] == "GT 710 1GB" else 0.65
    return 0.60


def speed(model, hardware, bits, context, users, cpu=False, excel=False):
    bandwidth = hardware["Memory BW (GB/s)" if cpu else "Bandwidth (GB/s)"]
    active = model["Active params (B)"] * bits / 8
    token_kv = kv_token(model, 8)
    denominator = active + (token_kv * context * users if excel else 0)
    return excel_efficiency(hardware, cpu) * bandwidth / denominator / users if excel else web_efficiency(hardware["Vendor"] if not cpu else "", cpu) * bandwidth / denominator / users


def calculate():
    models = [m for m in load_rows("models") if all(m.get(k) is not None for k in ("Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim"))]
    gpus = [g for g in load_rows("gpus") if g.get("Bandwidth (GB/s)") is not None]
    cpus = [c for c in load_rows("cpus") if c.get("Memory BW (GB/s)") is not None]
    quants = {q["Quantization"]: q["Bits/weight (eff.)"] for q in load_rows("quant")}
    bits = next(value for name, value in quants.items() if "Q4_K_M" in name)
    result = {"basis": "Q4_K_M, 8-bit KV, 1 user; positive delta means web/app estimate is higher than Excel"}
    for context in (8192, 32768):
        memory_deltas = []
        gpu_deltas = []
        cpu_deltas = []
        for model in models:
            web_memory = memory(model, bits, context, 1)
            excel_memory = memory(model, bits, context, 1, excel=True)
            memory_deltas.append((web_memory / excel_memory - 1) * 100)
            for gpu in gpus:
                web = speed(model, gpu, bits, context, 1)
                old = speed(model, gpu, bits, context, 1, excel=True)
                gpu_deltas.append((web / old - 1) * 100)
            for cpu in cpus:
                web = speed(model, cpu, bits, context, 1, cpu=True)
                old = speed(model, cpu, bits, context, 1, cpu=True, excel=True)
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
