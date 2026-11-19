"""Build docs/data.json from the public CSV tables.

The browser app uses the same rows and evidence records as the CLI. Run after any
CSV update, then commit docs/data.json.
"""
import csv
import json
from pathlib import Path

from common import ROOT, load_rows
from selector_parity import calculate as calculate_parity

DOCS = ROOT / "docs"
NUMERIC = {
    "Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim", "Max context", "Year",
    "Memory (GB)", "Bandwidth (GB/s)", "TDP / board power (W)", "CUDA compute capability",
    "Cores", "Base / default power (W)", "Max turbo / max power (W)", "Memory BW (GB/s)",
    "Max RAM (GB)", "CPU PCIe lanes", "Bits / weight", "Bits/weight (eff.)", "GB per 1B params",
    "Total params (B)", "Active params (B)", "Tier", "Reference rank",
}


def evidence_rows():
    with open(ROOT / "data" / "evidence.csv", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def source_facts(rows):
    facts = {}
    for row in rows:
        key = f"{row.get('Sheet', '')}|{row.get('Item', '')}"
        facts.setdefault(key, []).append(row)
    return facts


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    evidence = evidence_rows()
    payload = {
        "generated": True,
        "models": load_rows("models"),
        "gpus": load_rows("gpus"),
        "cpus": load_rows("cpus"),
        "quants": load_rows("quant"),
        "referenceModels": load_rows("reference_models"),
        "evidence": evidence,
        "factsByItem": source_facts(evidence),
    }
    output = DOCS / "data.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    parity_output = DOCS / "parity.json"
    parity_output.write_text(json.dumps(calculate_parity(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")
    print(f"wrote {parity_output} ({parity_output.stat().st_size / 1024:.1f} KiB)")
    print(f"rows: {len(payload['models'])} models, {len(payload['referenceModels'])} reference models, {len(payload['gpus'])} GPUs, {len(payload['cpus'])} CPUs, {len(evidence)} evidence checks")


if __name__ == "__main__":
    main()
