"""Export the workbook's compute estimates (Power sheet, columns AA:AB and AD:AE) used for time to first token.

  python scripts/export_compute_estimates.py <Offline_LLM_Hardware_Selector_vNN.xlsx>

Writes
  data/compute.csv            released hardware only (published)
  docs/private_compute.json   pre-release planning rows (git-ignored, loaded by the page only when present)
and replaces the "Compute" records in data/evidence.csv.

The figures are the workbook's planning estimates. They have no vendor source, so every record is
logged as DERIVED/ESTIMATE. Rows whose Evidence status says "Pre-release planning" are never
written to the published CSV.
"""
import csv
import datetime
import json
import re
import sys
import warnings

import openpyxl

from common import DATA, ROOT

PLANNING = re.compile(r"Pre-release planning", re.I)
BASIS = "Workbook planning estimate; no vendor source"
NOTE = ("No vendor source. Planning estimate used only for the time-to-first-token (prefill) calculation; "
        "validate with a prefill benchmark on the target hardware.")
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


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    warnings.filterwarnings("ignore")
    sheet = openpyxl.load_workbook(sys.argv[1], data_only=True)["Power"]
    column = openpyxl.utils.column_index_from_string
    estimates = {"GPUs": read_pairs(sheet, column("AA"), column("AB")), "CPUs": read_pairs(sheet, column("AD"), column("AE"))}
    parameter = {"GPUs": "GPU FP16 dense TFLOPS (est.)", "CPUs": "CPU TFLOPS at base power (est.)"}
    public, private, missing = [], [], []
    for sheet_name, (name, key) in {"GPUs": ("gpus", "Accelerator"), "CPUs": ("cpus", "CPU / platform")}.items():
        for item, status in table(name, key):
            value = estimates[sheet_name].get(item)
            if value is None:
                missing.append(f"{sheet_name}: {item}")
            elif PLANNING.search(status):
                private.append({"Sheet": sheet_name, "Item": item, "TFLOPS (est.)": value})
            else:
                public.append({"Sheet": sheet_name, "Item": item, "TFLOPS (est.)": f"{value:g}", "Basis": BASIS, "Evidence status": "DERIVED/ESTIMATE"})

    with open(DATA / "compute.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(public)
    (ROOT / "docs" / "private_compute.json").write_text(json.dumps({
        "note": "Local only. Pre-release planning compute estimates; never commit or publish.", "compute": private}, indent=1), encoding="utf-8")

    today = datetime.date.today().isoformat()
    with open(DATA / "evidence.csv", encoding="utf-8", newline="") as stream:
        kept = [row for row in csv.DictReader(stream) if row["Sheet"] != "Compute"]
    records = [{"Sheet": "Compute", "Item": row["Item"], "Cell": "", "Parameter": parameter[row["Sheet"]], "Value in sheet (before)": row["TFLOPS (est.)"],
                "Corrected to": "", "Source value / quote": "", "Status": "DERIVED/ESTIMATE", "Source type": "Workbook planning estimate",
                "Source URL": "", "Note": NOTE, "Checked": today} for row in public]
    with open(DATA / "evidence.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(kept + records)
    print(f"data/compute.csv: {len(public)} released rows; docs/private_compute.json: {len(private)} pre-release rows (not published)")
    print(f"data/evidence.csv: {len(records)} Compute records, {len(kept) + len(records)} total")
    if missing:
        print("no estimate in the workbook for: " + "; ".join(missing))


if __name__ == "__main__":
    main()
