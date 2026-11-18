"""Combine out/*_evidence.json into an Excel apply-plan (cell ops + Evidence sheet rows) for the selector workbook.

Usage: python scripts/excel/build_plan.py <workbook.xlsx>   -> out/apply_plan.json
The workbook is only read (to map item names to rows); apply_to_excel.ps1 writes to the OPEN workbook.
"""
import json, re, pathlib, warnings, math, sys, datetime
import openpyxl
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[2] / "out"
TODAY = datetime.date.today().isoformat()
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
wb = openpyxl.load_workbook(sys.argv[1], data_only=False)

def row_index(sheet, key_col=1):
    s = wb[sheet]; idx = {}
    for r in range(2, s.max_row + 1):
        v = s.cell(r, key_col).value
        if v is not None and str(v) not in idx:
            idx[str(v)] = r
    return idx

COLMAP = {
    "Models": {"Total params (B)": "C", "Active params (B)": "D", "Layers": "E", "KV heads": "F", "Head dim": "G", "Max context": "H", "License": "J", "HF repo": "K"},
    "GPUs": {"Memory (GB)": "C", "Bandwidth (GB/s)": "D", "TDP / board power (W)": "E", "Interface": "F", "Architecture": "N", "CUDA compute capability": "O"},
    "CPUs": {"Cores": "C", "Base / default power (W)": "D", "Max turbo / max power (W)": "E", "Memory": "F", "Memory BW (GB/s)": "G", "Max RAM (GB)": "H", "CPU PCIe lanes": "I"},
    "Voice Models": {"Total params (B)": "C", "Bits / weight": "D", "License": "G"},
    "Quant": {"Bits/weight (eff.)": "B", "GB per 1B params": "C"},
}
ROWS = {s: row_index(s) for s in COLMAP}
nvme = wb["NVMe Storage"]
nvme_rows = {str(nvme.cell(r, 36).value): r for r in range(1, 12) if nvme.cell(r, 36).value}  # AJ = col 36

GREEN, AMBER, RED, BLUE = 0xC6EFCE, 0xFFEB9C, 0xFFC7CE, 0xDDEBF7
def colour(status):
    if status == "VERIFIED":
        return GREEN
    if status == "MISMATCH":
        return RED
    if status == "ADDED":
        return BLUE
    return AMBER

def fmt_new(param, src):
    if src is None:
        return None
    if isinstance(src, (int, float)):
        if param in ("Layers", "KV heads", "Head dim", "Max context", "Cores", "CPU PCIe lanes", "Max RAM (GB)", "Memory (GB)"):
            return int(round(src))
        if param in ("Total params (B)", "Active params (B)"):
            return round(src, 2)
        return round(src, 1) if abs(src - round(src)) > 1e-9 else int(round(src))
    return str(src)

ops, evidence = [], []
all_recs = []
for f in ["models_evidence.json", "gpus_evidence.json", "cpus_evidence.json", "misc_evidence.json"]:
    if (OUT / f).exists():
        all_recs += json.load(open(OUT / f, encoding="utf-8"))

seen_cells = set()
for r in all_recs:
    sheet, item, param, status = r["Sheet"], r["Item"], r["Parameter"], r["Status"]
    cell = ""
    if sheet in COLMAP and param in COLMAP[sheet] and str(item) in ROWS[sheet]:
        cell = f"{COLMAP[sheet][param]}{ROWS[sheet][str(item)]}"
    elif sheet == "NVMe Storage":
        if item.startswith("MXFP4"):
            cell = "AH7"
        elif item in nvme_rows:
            cell = f"AK{nvme_rows[item]}"
    new_val = None
    if status == "MISMATCH" and r["Source value"] is not None:
        new_val = fmt_new(param, r["Source value"])
    if status == "ADDED" and r["Source value"]:
        new_val = r["Source value"]
    if sheet == "Quant" and param == "GB per 1B params":
        continue  # derived column, no cell op
    comment = f"[{status}] {r['Source type']}" + (f": {r['Note']}" if r.get("Note") else "")
    if r.get("Source URL"):
        comment += f"\nSource: {r['Source URL']}"
    if new_val is not None:
        comment = f"CORRECTED {r['Sheet value']} -> {new_val} on {r['Checked']}. " + comment
    comment += f"\nChecked {r['Checked']}"
    if cell and (sheet, cell) not in seen_cells:
        seen_cells.add((sheet, cell))
        ops.append({"sheet": sheet, "cell": cell, "new": new_val, "fill": colour(status), "comment": comment[:2000]})
    evidence.append([sheet, item, cell, param, r["Sheet value"], new_val if new_val is not None else "", r["Source value"], status,
                     r["Source type"], r.get("Source URL") or "", r.get("Note") or "", r["Checked"]])

# Spec-source / last-synced refresh for Models rows with verified config data
models_verified = {}
for r in all_recs:
    if r["Sheet"] == "Models" and r["Status"] in ("VERIFIED", "MISMATCH") and r["Parameter"] in ("Layers", "KV heads", "Head dim", "Max context"):
        models_verified.setdefault(r["Item"], set()).add(r["Source type"])
spec_ops = []
for item, types in models_verified.items():
    row = ROWS["Models"].get(item)
    if row:
        src = "HF config.json" + (" (mirror/base copy)" if any("mirror" in t for t in types) else "") + " / model card – verified " + TODAY + ""
        spec_ops.append({"sheet": "Models", "cell": f"M{row}", "new": src, "fill": None, "comment": None})
        spec_ops.append({"sheet": "Models", "cell": f"N{row}", "new": TODAY, "fill": None, "comment": None})

# Source-URL columns: one URL per item (first non-empty)
url_cols = {"GPUs": "AG", "CPUs": "AE", "Models": "P"}
status_cols = {"GPUs": "AH", "CPUs": "AF", "Models": "Q"}
url_ops = []
per_item = {}
for r in all_recs:
    if r["Sheet"] in url_cols and str(r["Item"]) in ROWS[r["Sheet"]]:
        key = (r["Sheet"], str(r["Item"]))
        d = per_item.setdefault(key, {"urls": [], "statuses": []})
        if r.get("Source URL") and r["Source URL"] not in d["urls"]:
            d["urls"].append(r["Source URL"])
        d["statuses"].append(r["Status"])
for (sheet, item), d in per_item.items():
    row = ROWS[sheet][item]
    url_ops.append({"sheet": sheet, "cell": f"{url_cols[sheet]}{row}", "new": "; ".join(d["urls"][:3]) if d["urls"] else "(no external source – see Evidence)", "fill": None, "comment": None})
    st = d["statuses"]
    summary = f"{st.count('VERIFIED')} verified / {st.count('MISMATCH')} corrected / {len(st) - st.count('VERIFIED') - st.count('MISMATCH')} estimate-or-unsourced"
    url_ops.append({"sheet": sheet, "cell": f"{status_cols[sheet]}{row}", "new": summary, "fill": None, "comment": None})
for sheet in url_cols:
    url_ops.append({"sheet": sheet, "cell": f"{url_cols[sheet]}1", "new": "Source URL (evidence " + TODAY + ")", "fill": None, "comment": None})
    url_ops.append({"sheet": sheet, "cell": f"{status_cols[sheet]}1", "new": "Evidence status", "fill": None, "comment": None})

# Price placeholders
for sheet, col, hdr in [("GPUs", "G", "Indicative price (USD)"), ("CPUs", "K", "Indicative platform price (USD)")]:
    evidence.append([sheet, "(all rows)", f"{col}2:{col}{wb[sheet].max_row}", hdr, "various", "", None, "PLACEHOLDER – NO SOURCE",
                     "none", "", "Indicative prices are planning placeholders (sheet 1 already labels them 'placeholder prices'); no vendor/distributor price source was used or verified.", TODAY])

plan = {"ops": ops + spec_ops + url_ops, "evidence": evidence,
        "price_cols": [("GPUs", "G"), ("CPUs", "K")]}
json.dump(plan, open(OUT / "apply_plan.json", "w", encoding="utf-8"), ensure_ascii=False, default=str)
from collections import Counter
print("cell ops", len(ops), "spec ops", len(spec_ops), "url ops", len(url_ops), "evidence rows", len(evidence))
print(Counter(e[7] for e in evidence))
print("corrections:", [(o["sheet"], o["cell"], o["new"]) for o in ops if o["new"] is not None][:200])
