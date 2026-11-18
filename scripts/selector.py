"""Offline LLM hardware selector – answer "what can run model X / what does it need" from the evidence-backed CSV tables.

Examples
  python scripts/selector.py "Qwen3 32B"                       # Q4_K_M, 8K context, 1 user, 20 tok/s target
  python scripts/selector.py "Llama 3.3 70B" --quant Q8_0 --context 32768 --users 4 --vendor NVIDIA
  python scripts/selector.py --hw "RTX 5090"                    # what fits on this accelerator
  python scripts/selector.py --list models | --list gpus | --list cpus
  python scripts/selector.py "Qwen3 32B" --json

Sizing model (same as the workbook):
  weights_GB      = total_params_B * bits/8 * 1.03
  kv_per_token_GB = 2 * layers * kv_heads * head_dim * 2 bytes / 1e9        (FP16 KV cache)
  need_GB         = weights + kv_per_token * context * users + 1.5 (runtime)
  usable_GB       = GPU memory * 0.92 ; CPU RAM: max_ram - min(8, 0.25*max_ram)
  tok/s per user  = eff * bandwidth_GBps / (active_params_B * bits/8 + kv_per_token_GB) / users
                    eff = 0.55 Intel GPU / CPU, 0.65 Apple, 0.60 other GPUs
These are planning estimates, not benchmarks.
"""
import argparse, json, re, sys
from common import load_rows

RUNTIME_GB = 1.5

def pick(rows, key, name):
    name_l = name.lower()
    exact = [r for r in rows if str(r[key]).lower() == name_l]
    if exact:
        return exact[0]
    part = [r for r in rows if name_l in str(r[key]).lower()]
    if len(part) == 1:
        return part[0]
    if not part:
        raise SystemExit(f"No {key} matching '{name}'. Try --list.")
    raise SystemExit(f"Ambiguous '{name}': " + "; ".join(str(r[key]) for r in part[:12]))

def quant_bits(quant_rows, q):
    ql = q.lower().replace(" ", "")
    for r in quant_rows:
        if ql in str(r["Quantization"]).lower().replace(" ", ""):
            return float(r["Bits/weight (eff.)"]), r["Quantization"]
    raise SystemExit("Unknown quant. Options: " + ", ".join(str(r["Quantization"]) for r in quant_rows))

def kv_per_token_gb(m):
    return 2 * float(m["Layers"]) * float(m["KV heads"]) * float(m["Head dim"]) * 2 / 1e9

def need_gb(m, bits, context, users):
    weights = float(m["Total params (B)"]) * bits / 8 * 1.03
    kv = kv_per_token_gb(m) * context * users
    return weights, kv, weights + kv + RUNTIME_GB

def eff_for(vendor, is_cpu=False):
    if is_cpu or vendor == "Intel":
        return 0.55
    return 0.65 if vendor == "Apple" else 0.60

def tps(bw, eff, m, bits, users, qty=1):
    per_tok = float(m["Active params (B)"]) * bits / 8 + kv_per_token_gb(m)
    scale = min(qty, 8) * (0.8 if qty > 1 else 1)
    return eff * bw * scale / per_tok / users

def evidence_tag(r):
    s = str(r.get("Evidence status") or "")
    return s if s else "unverified"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="model name (substring ok)")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--target-tps", type=float, default=20.0)
    ap.add_argument("--vendor", help="filter accelerators by vendor (Intel/NVIDIA/AMD/Apple)")
    ap.add_argument("--max-cards", type=int, default=1, help="allow up to N identical cards (tensor-parallel)")
    ap.add_argument("--include-legacy", action="store_true", help="include used/legacy GPUs")
    ap.add_argument("--hw", help="instead of a model: show what this accelerator or CPU can run at the chosen quant")
    ap.add_argument("--list", choices=["models", "gpus", "cpus", "quants"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    models, gpus, cpus, quants = load_rows("models"), load_rows("gpus"), load_rows("cpus"), load_rows("quant")
    if a.list:
        rows = {"models": models, "gpus": gpus, "cpus": cpus, "quants": quants}[a.list]
        key = {"models": "Model", "gpus": "Accelerator", "cpus": "CPU / platform", "quants": "Quantization"}[a.list]
        for r in rows:
            print(r[key])
        return
    bits, qname = quant_bits(quants, a.quant)

    if a.hw:
        gpu = next((g for g in gpus if a.hw.lower() in str(g["Accelerator"]).lower()), None)
        cpu = None if gpu else next((c for c in cpus if a.hw.lower() in str(c["CPU / platform"]).lower()), None)
        if not gpu and not cpu:
            raise SystemExit("No accelerator/CPU matching " + a.hw)
        if gpu:
            usable, bw, eff, label = float(gpu["Memory (GB)"]) * 0.92, float(gpu["Bandwidth (GB/s)"]), eff_for(gpu["Vendor"]), gpu["Accelerator"]
            src = gpu.get("Source URL")
        else:
            mr = float(cpu["Max RAM (GB)"]); usable, bw, eff, label = mr - min(8, 0.25 * mr), float(cpu["Memory BW (GB/s)"]), 0.55, cpu["CPU / platform"]
            src = cpu.get("Source URL")
        out = []
        for m in models:
            if not all(m.get(k) for k in ("Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim")):
                continue
            w, kv, need = need_gb(m, bits, a.context, a.users)
            if need <= usable:
                out.append({"model": m["Model"], "need_GB": round(need, 1), "tok_s_per_user": round(tps(bw, eff, m, bits, a.users), 1), "evidence": evidence_tag(m)})
        out.sort(key=lambda x: -x["need_GB"])
        res = {"hardware": label, "usable_GB": round(usable, 1), "bandwidth_GBps": bw, "quant": qname, "context": a.context, "users": a.users, "source": src, "fits": out}
        if a.json:
            print(json.dumps(res, indent=1)); return
        print(f"{label}: usable {usable:.1f} GB, {bw:g} GB/s  (source: {src})\n{qname}, {a.context} ctx, {a.users} user(s) -> {len(out)} models fit")
        for x in out[:40]:
            flag = "" if x["tok_s_per_user"] >= a.target_tps else "  (slow)"
            print(f"  {x['model']:<40} {x['need_GB']:>7.1f} GB  ~{x['tok_s_per_user']:>6.1f} tok/s{flag}   [{x['evidence']}]")
        return

    if not a.model:
        ap.error("give a model name, --hw, or --list")
    m = pick(models, "Model", a.model)
    for k in ("Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim"):
        if m.get(k) in (None, ""):
            raise SystemExit(f"{m['Model']}: missing {k}; cannot size it")
    w, kv, need = need_gb(m, bits, a.context, a.users)
    gpu_hits, cpu_hits = [], []
    for g in gpus:
        if a.vendor and str(g["Vendor"]).lower() != a.vendor.lower():
            continue
        if not a.include_legacy and str(g.get("Used / legacy?")).lower() == "yes":
            continue
        try:
            mem, bw = float(g["Memory (GB)"]), float(g["Bandwidth (GB/s)"])
        except (TypeError, ValueError):
            continue
        for qty in range(1, a.max_cards + 1):
            if mem * 0.92 * qty >= need:
                t = tps(bw, eff_for(g["Vendor"]), m, bits, a.users, qty)
                gpu_hits.append({"accelerator": g["Accelerator"], "vendor": g["Vendor"], "cards": qty, "memory_GB": mem, "tok_s_per_user": round(t, 1),
                                 "meets_target": t >= a.target_tps, "tdp_W": g.get("TDP / board power (W)"), "evidence": evidence_tag(g), "source": g.get("Source URL")})
                break
    for c in cpus:
        try:
            mr, bw = float(c["Max RAM (GB)"]), float(c["Memory BW (GB/s)"])
        except (TypeError, ValueError):
            continue
        if mr - min(8, 0.25 * mr) >= need:
            t = tps(bw, 0.55, m, bits, a.users)
            cpu_hits.append({"cpu": c["CPU / platform"], "max_RAM_GB": mr, "tok_s_per_user": round(t, 1), "meets_target": t >= a.target_tps,
                             "evidence": evidence_tag(c), "source": c.get("Source URL")})
    gpu_hits.sort(key=lambda x: (x["cards"], -x["tok_s_per_user"]))
    cpu_hits.sort(key=lambda x: -x["tok_s_per_user"])
    res = {"model": m["Model"], "hf_repo": m.get("HF repo"), "model_source": m.get("Source URL"), "model_evidence": evidence_tag(m),
           "quant": qname, "bits_per_weight": bits, "context": a.context, "users": a.users,
           "weights_GB": round(w, 1), "kv_cache_GB": round(kv, 1), "runtime_GB": RUNTIME_GB, "need_GB": round(need, 1),
           "gpus_that_fit": gpu_hits, "cpus_that_fit": cpu_hits,
           "note": "Planning estimates (bandwidth model, FP16 KV cache); validate on target hardware."}
    if a.json:
        print(json.dumps(res, indent=1, default=str)); return
    print(f"{m['Model']}  ({m.get('Type')}, {m['Total params (B)']}B total / {m['Active params (B)']}B active, {m['Layers']}L x {m['KV heads']} KV x {m['Head dim']}d)")
    print(f"  source: {m.get('HF repo') or '-'}  [{evidence_tag(m)}]")
    print(f"  {qname} ({bits} bpw), {a.context} ctx, {a.users} user(s): weights {w:.1f} GB + KV {kv:.1f} GB + runtime {RUNTIME_GB} GB = {need:.1f} GB needed")
    print(f"\nAccelerators that fit (<= {a.max_cards} card(s)); target {a.target_tps:g} tok/s per user:")
    for x in gpu_hits[:40]:
        print(f"  {'OK ' if x['meets_target'] else 'slow'} {x['accelerator']:<36} x{x['cards']}  {x['memory_GB']:>5g} GB  ~{x['tok_s_per_user']:>6.1f} tok/s  {x['tdp_W']} W  [{x['evidence']}]")
    if not gpu_hits:
        print("  none – raise --max-cards, lower the quant, or shorten the context")
    print("\nCPU-only platforms that fit (RAM):")
    for x in cpu_hits[:15]:
        print(f"  {'OK ' if x['meets_target'] else 'slow'} {x['cpu']:<40} {x['max_RAM_GB']:>6g} GB  ~{x['tok_s_per_user']:>6.1f} tok/s  [{x['evidence']}]")
    print("\n" + res["note"])

if __name__ == "__main__":
    main()
