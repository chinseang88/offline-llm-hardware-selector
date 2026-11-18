"""Export sanitised, public-shareable data tables from the (internal) selector workbook.

Usage: python scripts/export_public_data.py <workbook.xlsx> [--out data]

Drops pre-production rows (name prefixes in PRIVATE_ROW_PREFIXES) and any text that references
non-public documents, plus the unsourced indicative-price columns.
"""
import argparse, csv, pathlib, re, warnings
import openpyxl

warnings.filterwarnings("ignore")
PRIVATE_ROW_PREFIXES = ("Intel CRI", "NVL", "PTL", "RZL")   # pre-production SKUs kept in the private workbook only
CONFIDENTIAL_ROW = re.compile(r"^(" + "|".join(PRIVATE_ROW_PREFIXES) + r")", re.I)
CONFIDENTIAL_TEXT = re.compile(r"Intel Confidential|playbook|EDS \d+|Gold deck|\b(8|9)\d{5}\b|Xe3P/CRI,?\s?", re.I)

SHEETS = {
    "models": ("Models", ["Model", "Vendor", "Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim", "Max context", "Type", "License", "HF repo", "Year", "Spec source", "Last synced", "Runtime arch", "Source URL", "Evidence status"]),
    "gpus": ("GPUs", ["Accelerator", "Vendor", "Memory (GB)", "Bandwidth (GB/s)", "TDP / board power (W)", "Interface", "Notes / source", "Used / legacy?", "Architecture", "CUDA compute capability", "CUDA toolkit / compute stack", "ROCm support", "Runtime HW class", "Source URL", "Evidence status"]),
    "cpus": ("CPUs", ["CPU / platform", "Class", "Cores", "Base / default power (W)", "Max turbo / max power (W)", "Memory", "Memory BW (GB/s)", "Max RAM (GB)", "CPU PCIe lanes", "AI accel (iGPU/NPU)", "Notes / power source", "Runtime HW class", "Runtimes that run on this CPU/SoC", "Source URL", "Evidence status"]),
    "voice_models": ("Voice Models", ["Stage", "Model", "Total params (B)", "Bits / weight", "Source", "License", "Notes"]),
    "quant": ("Quant", ["Quantization", "Bits/weight (eff.)", "GB per 1B params", "Quality"]),
}

def header_map(ws):
    m = {}
    for c in ws[1]:
        if c.value:
            key = re.sub(r"\s*\(evidence [\d-]+\)", "", str(c.value)).strip()
            m[key] = c.column
    return m

def clean(v):
    if isinstance(v, str):
        v = CONFIDENTIAL_TEXT.sub("", v).strip()
        v = re.sub(r"\(\s*,", "(", v).replace("( ", "(").replace(" ,", ",")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return "" if v is None else v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(a.workbook, data_only=True)
    for fname, (sheet, cols) in SHEETS.items():
        ws = wb[sheet]; hm = header_map(ws)
        missing = [c for c in cols if c not in hm]
        if missing:
            raise SystemExit(f"{sheet}: missing columns {missing}")
        n = 0
        with open(out / f"{fname}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in range(2, ws.max_row + 1):
                key = ws.cell(r, hm[cols[0]]).value
                if key is None or (fname != "voice_models" and str(key).strip() == ""):
                    continue
                if fname == "voice_models" and ws.cell(r, hm["Model"]).value is None:
                    continue
                if CONFIDENTIAL_ROW.search(str(key)):
                    continue
                w.writerow([clean(ws.cell(r, hm[c]).value) for c in cols]); n += 1
        print(f"{fname}.csv: {n} rows")
    # evidence log
    if "Evidence" in wb.sheetnames:
        ws = wb["Evidence"]
        hdr_row = next(r for r in range(1, 20) if ws.cell(r, 1).value == "Sheet")
        hdr = [ws.cell(hdr_row, c).value for c in range(1, 13)]
        n = 0
        with open(out / "evidence.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(hdr)
            for r in range(hdr_row + 1, ws.max_row + 1):
                row = [ws.cell(r, c).value for c in range(1, 13)]
                if row[0] is None:
                    continue
                if CONFIDENTIAL_ROW.search(str(row[1])) or str(row[7]).startswith("PLACEHOLDER") or str(row[7]) == "INTERNAL SOURCE":
                    continue
                w.writerow([clean(v) for v in row]); n += 1
        print(f"evidence.csv: {n} rows")

if __name__ == "__main__":
    main()
