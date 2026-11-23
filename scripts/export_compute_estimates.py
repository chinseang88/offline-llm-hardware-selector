"""Export the compute estimates (TFLOPS) used for time to first token.

  python scripts/export_compute_estimates.py <Offline_LLM_Hardware_Selector_vNN.xlsx>   workbook Power sheet, AA:AB and AD:AE
  python scripts/export_compute_estimates.py --derive                                   keep the workbook rows, redo the derived rows

Writes
  data/compute.csv            workbook estimates for released hardware, plus a derived figure for every other row
  docs/private_compute.json   workbook estimates for pre-release planning rows (git-ignored, loaded by the page only when present)
and replaces the "Compute" records in data/evidence.csv.

The workbook figures are planning estimates with no vendor source. The workbook figure of a row whose
Evidence status says "Pre-release planning" is never written to the published CSV. Such a row gets a
figure derived from columns that are already published (common.derive_compute), so every row has a
time to first token. Every record is logged as DERIVED/ESTIMATE.
"""
import csv
import datetime
import json
import re
import sys
import warnings

import openpyxl

from common import DATA, DERIVED_BASIS, ROOT, derive_compute, load_rows

PLANNING = re.compile(r"Pre-release planning", re.I)
BASIS = "Workbook planning estimate; no vendor source"
NOTE = ("No vendor source. Planning estimate used only for the time-to-first-token (prefill) calculation; "
        "validate with a prefill benchmark on the target hardware.")
DERIVED_NOTE = ("No vendor source and no workbook figure is published for this row. {basis}. Used only for the "
                "time-to-first-token (prefill) calculation; validate with a prefill benchmark on the target hardware.")
FIELDS = ["Sheet", "Item", "TFLOPS (est.)", "Basis", "Evidence status"]
EVIDENCE_FIELDS = ["Sheet", "Item", "Cell", "Parameter", "Value in sheet (before)", "Corrected to", "Source value / quote", "Status",
                   "Source type", "Source URL", "Note", "Checked"]


def read_pairs(sheet, name_col, value_col):
    pairs = {}
    for row in range(2, sheet.max_row + 1):
        name, value = sheet.cell(row, name_col).value, sheet.cell(row, value_col).value
        if name and isinstance(value, (int, float)) and value > 0:
            pairs[str(name)] = float(value)
    return pairs


def table(name, key):
    with open(DATA / f"{name}.csv", encoding="utf-8", newline="") as stream:
        return [(row[key], row.get("Evidence status", "")) for row in csv.DictReader(stream)]


def from_workbook(path):
    warnings.filterwarnings("ignore")
    sheet = openpyxl.load_workbook(path, data_only=True)["Power"]
    column = openpyxl.utils.column_index_from_string
    estimates = {"GPUs": read_pairs(sheet, column("AA"), column("AB")), "CPUs": read_pairs(sheet, column("AD"), column("AE"))}
    public, private = [], []
    for sheet_name, (name, key) in {"GPUs": ("gpus", "Accelerator"), "CPUs": ("cpus", "CPU / platform")}.items():
        for item, status in table(name, key):
            value = estimates[sheet_name].get(item)
            if value is None:
                continue
            if PLANNING.search(status):
                private.append({"Sheet": sheet_name, "Item": item, "TFLOPS (est.)": value})
            else:
                public.append({"Sheet": sheet_name, "Item": item, "TFLOPS (est.)": f"{value:g}", "Basis": BASIS, "Evidence status": "DERIVED/ESTIMATE"})
    (ROOT / "docs" / "private_compute.json").write_text(json.dumps({
        "note": "Local only. Pre-release planning compute estimates; never commit or publish.", "compute": private}, indent=1), encoding="utf-8")
    print(f"docs/private_compute.json: {len(private)} pre-release workbook rows (not published)")
    return public


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--derive":
        with open(DATA / "compute.csv", encoding="utf-8", newline="") as stream:
            public = [row for row in csv.DictReader(stream) if not row["Basis"].startswith(DERIVED_BASIS)]
    else:
        public = from_workbook(sys.argv[1])
    estimates = {(row["Sheet"], row["Item"]): float(row["TFLOPS (est.)"]) for row in public}
    derived = derive_compute({"GPUs": load_rows("gpus"), "CPUs": load_rows("cpus")}, estimates)

    with open(DATA / "compute.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(public + derived)

    parameter = {"GPUs": "GPU FP16 dense TFLOPS (est.)", "CPUs": "CPU TFLOPS at base power (est.)"}
    today = datetime.date.today().isoformat()
    with open(DATA / "evidence.csv", encoding="utf-8", newline="") as stream:
        kept = [row for row in csv.DictReader(stream) if row["Sheet"] != "Compute"]
    records = [{"Sheet": "Compute", "Item": row["Item"], "Cell": "", "Parameter": parameter[row["Sheet"]], "Value in sheet (before)": row["TFLOPS (est.)"],
                "Corrected to": "", "Source value / quote": "", "Status": "DERIVED/ESTIMATE",
                "Source type": DERIVED_BASIS if row in derived else "Workbook planning estimate", "Source URL": "",
                "Note": DERIVED_NOTE.format(basis=row["Basis"]) if row in derived else NOTE, "Checked": today} for row in public + derived]
    with open(DATA / "evidence.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(kept + records)
    print(f"data/compute.csv: {len(public)} workbook rows for released hardware + {len(derived)} rows derived from published columns")
    print(f"data/evidence.csv: {len(records)} Compute records, {len(kept) + len(records)} total")
    covered = {(row["Sheet"], row["Item"]) for row in public + derived}
    missing = [f"{sheet}: {item}" for sheet, (name, key) in {"GPUs": ("gpus", "Accelerator"), "CPUs": ("cpus", "CPU / platform")}.items()
               for item, _ in table(name, key) if (sheet, item) not in covered]
    if missing:
        print("no compute figure for: " + "; ".join(missing))


if __name__ == "__main__":
    main()
