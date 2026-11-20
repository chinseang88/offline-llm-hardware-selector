"""Merge selected verifier rows into data/evidence.csv by (Sheet, Item, Parameter)."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_FIELDS = [
    "Sheet", "Item", "Cell", "Parameter", "Value in sheet (before)", "Corrected to",
    "Source value / quote", "Status", "Source type", "Source URL", "Note", "Checked",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--items", required=True, help="Comma-separated item names")
    args = parser.parse_args()
    items = {item.strip() for item in args.items.split(",")}

    with args.input.open(encoding="utf-8", newline="") as stream:
        verified = [row for row in csv.DictReader(stream) if row.get("Item") in items]
    overrides = {
        ("Qwen3.8 Flash-Next 180B-A6B", "Active params (B)"): (
            "6", "VERIFIED",
            "Official Qwen card states: 125B with 6B activated, plus 51B n-gram embeddings and 4B MTP; 6B is the per-token active compute size used by the selector.",
        ),
        ("Gemma 4 E2B", "Active params (B)"): (
            "2.3", "VERIFIED",
            "Official Gemma 4 card defines E2B as 2.3B effective parameters; 5.1B total includes per-layer embeddings.",
        ),
        ("Gemma 4 E4B", "Active params (B)"): (
            "4.5", "VERIFIED",
            "Official Gemma 4 card defines E4B as 4.5B effective parameters; 8B total includes per-layer embeddings.",
        ),
    }
    normalized = []
    for source in verified:
        key = (source["Item"], source["Parameter"])
        source_value = source.get("Source value") or ""
        status = source.get("Status") or ""
        note = source.get("Note") or ""
        if key in overrides:
            source_value, status, note = overrides[key]
        normalized.append({
            "Sheet": source.get("Sheet") or "Models",
            "Item": source.get("Item") or "",
            "Cell": "",
            "Parameter": source.get("Parameter") or "",
            "Value in sheet (before)": source.get("Sheet value") or "",
            "Corrected to": "" if status == "VERIFIED" else source_value,
            "Source value / quote": source_value,
            "Status": status,
            "Source type": source.get("Source type") or "",
            "Source URL": source.get("Source URL") or "",
            "Note": note,
            "Checked": source.get("Checked") or "",
        })

    target = ROOT / "data" / "evidence.csv"
    with target.open(encoding="utf-8", newline="") as stream:
        existing = list(csv.DictReader(stream))
    replacement_keys = {(row["Sheet"], row["Item"], row["Parameter"]) for row in normalized}
    retained = [row for row in existing if (row["Sheet"], row["Item"], row["Parameter"]) not in replacement_keys]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(retained + normalized)
    print(f"merged {len(normalized)} evidence rows for {len(items)} models")


if __name__ == "__main__":
    main()
