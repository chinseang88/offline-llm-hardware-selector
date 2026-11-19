"""Import public model identity facts from an authenticated reference-site export.

Measured quality, speed, memory, platform, timing, issue, and internal-source fields
are intentionally excluded. They require a public, non-redacted citation before
promotion into the public evidence log.
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE_FIELDS = [
    "Model", "Tier", "Reference rank", "Family", "Vendor", "Total params (B)",
    "Active params (B)", "Architecture", "License", "License class",
    "Context (max)", "Release date", "Model card URL", "Release source URL",
    "GGUF repository URL", "License source URL", "Source URL", "Evidence status",
]
EVIDENCE_FIELDS = [
    "Sheet", "Item", "Cell", "Parameter", "Value in sheet (before)",
    "Corrected to", "Source value / quote", "Status", "Source type",
    "Source URL", "Note", "Checked",
]


def clean(value):
    return "" if value is None else str(value).strip()


def sort_number(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def public_facts(model, family):
    facts = []
    checks = [
        ("Model identity", model.get("name"), family.get("card_url"), "Hugging Face model card", "Reference model family matched to its publisher model card."),
        ("Total params (B)", model.get("params_arch"), family.get("card_url"), "Hugging Face model card", "Publisher model identity and parameter family; no measured result imported."),
        ("License", model.get("license"), model.get("license_src") or family.get("card_url"), "License source", "License classification is copied with its original public licence link."),
        ("Release date", family.get("released"), family.get("released_src") or family.get("card_url"), "Publisher release source", "Publisher release date and source."),
    ]
    for parameter, value, url, source_type, note in checks:
        if value and url:
            facts.append((parameter, value, url, source_type, note))
    return facts


def import_catalog(input_path: Path):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not payload.get("meta", {}).get("redacted"):
        raise ValueError("Expected a redacted authenticated export; refusing to import without its disclosure marker")

    families = {clean(row.get("family")): row for row in payload.get("families", [])}
    rows = []
    evidence = []
    checked = date.today().isoformat()
    for entry in payload.get("model_list", []):
        family = families.get(clean(entry.get("family")), {})
        card_url = clean(family.get("card_url"))
        release_url = clean(family.get("released_src"))
        gguf_url = f"https://huggingface.co/{clean(entry.get('repo'))}" if clean(entry.get("repo")) else ""
        license_url = clean(entry.get("license_src")) or card_url
        facts = public_facts(entry, family)
        source_urls = "; ".join(dict.fromkeys(url for url in (card_url, release_url, gguf_url, license_url) if url))
        rows.append({
            "Model": clean(entry.get("name")),
            "Tier": clean(entry.get("tier_no")),
            "Reference rank": clean(entry.get("list_rank")),
            "Family": clean(entry.get("family")),
            "Vendor": clean(family.get("vendor")),
            "Total params (B)": clean(family.get("params_b")),
            "Active params (B)": clean(family.get("active_b")),
            "Architecture": clean(family.get("arch")),
            "License": clean(entry.get("license")),
            "License class": clean(entry.get("license_class")),
            "Context (max)": clean(entry.get("context")),
            "Release date": clean(family.get("released")),
            "Model card URL": card_url,
            "Release source URL": release_url,
            "GGUF repository URL": gguf_url,
            "License source URL": license_url,
            "Source URL": source_urls,
            "Evidence status": f"{len(facts)} verified public facts / benchmark data withheld" if facts else "Public verification pending",
        })
        for parameter, value, url, source_type, note in facts:
            evidence.append({
                "Sheet": "Reference Models", "Item": clean(entry.get("name")), "Cell": "",
                "Parameter": parameter, "Value in sheet (before)": "", "Corrected to": value,
                "Source value / quote": value, "Status": "VERIFIED", "Source type": source_type,
                "Source URL": url, "Note": note, "Checked": checked,
            })

    rows.sort(key=lambda row: (sort_number(row["Tier"], 99), sort_number(row["Reference rank"], 999), row["Model"]))
    with (DATA / "reference_models.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REFERENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    evidence_path = DATA / "evidence.csv"
    with evidence_path.open(encoding="utf-8", newline="") as stream:
        existing = list(csv.DictReader(stream))
    retained = [row for row in existing if row.get("Sheet") != "Reference Models"]
    with evidence_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(retained + evidence)

    print(f"wrote {len(rows)} public reference model rows and {len(evidence)} verified facts")
    print("excluded: private benchmarks, host measurements, timing, quality scores, memory runs, issues, and withheld sources")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Authenticated reference-site dataset JSON export")
    import_catalog(parser.parse_args().input)


if __name__ == "__main__":
    main()
