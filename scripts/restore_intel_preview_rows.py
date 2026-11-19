"""Add sanitized Nova Lake (NVL) CPU and Crescent Island (CRI) GPU rows to public CSV data.

The workbook remains private. This script exports only the public-facing product
fields, removes internal document IDs/prices/notes, and marks every row as a
pre-release planning SKU with a pending public source.

Usage: python scripts/restore_intel_preview_rows.py <private-workbook.xlsx>
"""
import csv
import pathlib
import sys

import openpyxl

from common import ROOT

CPU_COLUMNS = [
    "CPU / platform", "Class", "Cores", "Base / default power (W)", "Max turbo / max power (W)",
    "Memory", "Memory BW (GB/s)", "Max RAM (GB)", "CPU PCIe lanes", "AI accel (iGPU/NPU)",
    "Notes / power source", "Runtime HW class", "Runtimes that run on this CPU/SoC", "Source URL", "Evidence status",
]
GPU_COLUMNS = [
    "Accelerator", "Vendor", "Memory (GB)", "Bandwidth (GB/s)", "TDP / board power (W)", "Interface",
    "Notes / source", "Used / legacy?", "Architecture", "CUDA compute capability", "CUDA toolkit / compute stack",
    "ROCm support", "Runtime HW class", "Source URL", "Evidence status",
]
PREVIEW_STATUS = "Pre-release planning / public source pending"


def existing(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    workbook = openpyxl.load_workbook(sys.argv[1], data_only=True)

    cpu_path = ROOT / "data" / "cpus.csv"
    cpus = existing(cpu_path)
    public_cpu_names = {row["CPU / platform"] for row in cpus}
    sheet = workbook["CPUs"]
    for row_number in range(2, sheet.max_row + 1):
        name = str(sheet.cell(row_number, 1).value or "")
        if not name.startswith("NVL") or name in public_cpu_names:
            continue
        cpus.append({
            "CPU / platform": name,
            "Class": sheet.cell(row_number, 2).value,
            "Cores": sheet.cell(row_number, 3).value,
            "Base / default power (W)": sheet.cell(row_number, 4).value,
            "Max turbo / max power (W)": sheet.cell(row_number, 5).value,
            "Memory": sheet.cell(row_number, 6).value,
            "Memory BW (GB/s)": sheet.cell(row_number, 7).value,
            "Max RAM (GB)": sheet.cell(row_number, 8).value,
            "CPU PCIe lanes": sheet.cell(row_number, 9).value,
            "AI accel (iGPU/NPU)": sheet.cell(row_number, 10).value,
            "Notes / power source": "Nova Lake pre-release planning SKU exported from the private selector workbook; internal document IDs and pricing stripped. Max turbo and capacity values may be preliminary; verify before product commitment.",
            "Runtime HW class": sheet.cell(row_number, 18).value,
            "Runtimes that run on this CPU/SoC": sheet.cell(row_number, 20).value,
            "Source URL": "",
            "Evidence status": PREVIEW_STATUS,
        })
    cpus.sort(key=lambda row: (row.get("Class") or "", row.get("CPU / platform") or ""))
    save(cpu_path, CPU_COLUMNS, cpus)

    gpu_path = ROOT / "data" / "gpus.csv"
    gpus = existing(gpu_path)
    public_gpu_names = {row["Accelerator"] for row in gpus}
    sheet = workbook["GPUs"]
    for row_number in range(2, sheet.max_row + 1):
        name = str(sheet.cell(row_number, 1).value or "")
        if not name.startswith("Intel CRI") or name in public_gpu_names:
            continue
        gpus.append({
            "Accelerator": name,
            "Vendor": "Intel",
            "Memory (GB)": sheet.cell(row_number, 3).value,
            "Bandwidth (GB/s)": sheet.cell(row_number, 4).value,
            "TDP / board power (W)": sheet.cell(row_number, 5).value,
            "Interface": sheet.cell(row_number, 6).value,
            "Notes / source": "Crescent Island (CRI) pre-release planning SKU exported from the private selector workbook; internal document IDs, pricing and availability notes stripped. Bandwidth/power may be estimated; verify before product commitment.",
            "Used / legacy?": "No",
            "Architecture": "Xe3P-XPC Crescent Island",
            "CUDA compute capability": "-",
            "CUDA toolkit / compute stack": "No CUDA – oneAPI / SYCL, vLLM XPU, OpenVINO",
            "ROCm support": "-",
            "Runtime HW class": sheet.cell(row_number, 18).value,
            "Source URL": "",
            "Evidence status": PREVIEW_STATUS,
        })
    gpus.sort(key=lambda row: (row["Vendor"] != "Intel", row.get("Architecture") or "", row.get("Memory (GB)") or 0, row["Accelerator"]))
    save(gpu_path, GPU_COLUMNS, gpus)

    evidence_path = ROOT / "data" / "evidence.csv"
    evidence = existing(evidence_path)
    evidence = [row for row in evidence if row.get("Status") != "PLANNING / PUBLIC SOURCE PENDING" or not (str(row.get("Item", "")).startswith("NVL") or str(row.get("Item", "")).startswith("Intel CRI"))]
    today = "2026-11-19"
    for row in cpus:
        name = row["CPU / platform"]
        if not name.startswith("NVL"):
            continue
        for param, value in (("Cores", row["Cores"]), ("Base power (W)", row["Base / default power (W)"]), ("Max power (W)", row["Max turbo / max power (W)"]), ("Memory", row["Memory"]), ("Memory BW (GB/s)", row["Memory BW (GB/s)"]), ("Max RAM (GB)", row["Max RAM (GB)"]), ("PCIe lanes", row["CPU PCIe lanes"])):
            evidence.append({"Sheet": "CPUs", "Item": name, "Cell": "", "Parameter": param, "Value in sheet (before)": value, "Corrected to": "", "Source value / quote": value, "Status": "PLANNING / PUBLIC SOURCE PENDING", "Source type": "Intel pre-release planning row (private workbook)", "Source URL": "", "Note": "Public selectors restored by explicit request; internal document IDs and pricing removed. Verify against a public Intel source before commitment.", "Checked": today})
    for row in gpus:
        name = row["Accelerator"]
        if not name.startswith("Intel CRI"):
            continue
        for param, value in (("Memory (GB)", row["Memory (GB)"]), ("Bandwidth (GB/s)", row["Bandwidth (GB/s)"]), ("TDP (W)", row["TDP / board power (W)"]), ("Interface", row["Interface"]), ("Architecture", row["Architecture"])):
            evidence.append({"Sheet": "GPUs", "Item": name, "Cell": "", "Parameter": param, "Value in sheet (before)": value, "Corrected to": "", "Source value / quote": value, "Status": "PLANNING / PUBLIC SOURCE PENDING", "Source type": "Intel pre-release planning row (private workbook)", "Source URL": "", "Note": "Crescent Island planning spec; internal document IDs, availability, and pricing removed. Verify against a public Intel source before commitment.", "Checked": today})
    evidence_columns = ["Sheet", "Item", "Cell", "Parameter", "Value in sheet (before)", "Corrected to", "Source value / quote", "Status", "Source type", "Source URL", "Note", "Checked"]
    save(evidence_path, evidence_columns, evidence)

    print(f"CPUs: {len(cpus)} rows; NVL: {sum(r['CPU / platform'].startswith('NVL') for r in cpus)}")
    print(f"GPUs: {len(gpus)} rows; CRI: {sum(r['Accelerator'].startswith('Intel CRI') for r in gpus)}")
    print(f"Evidence: {len(evidence)} rows; planning records added for NVL/CRI")


if __name__ == "__main__":
    main()
