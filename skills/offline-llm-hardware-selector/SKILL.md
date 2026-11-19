---
name: offline-llm-hardware-selector
description: Size and select hardware for running LLMs offline (GPU / CPU / unified-memory boxes) and verify every spec against its primary source. Use whenever someone asks "what hardware runs model X", "will model X fit on GPU Y", "how much VRAM / RAM / tokens-per-second", "which Intel Arc / NVIDIA / AMD card for a 70B model", "KV cache size", "quantization size", or wants to add, refresh or audit GPU/CPU/LLM spec data (memory, bandwidth, TDP, layers, KV heads, context, licence) with evidence from Hugging Face, Intel ARK, AMD, NVIDIA, Apple or standards bodies. Also use for the Offline_LLM_Hardware_Selector Excel workbook (Evidence sheet, Source URL columns) and for the data/*.csv tables in this repo.
---

# Offline LLM Hardware Selector

Evidence-backed sizing of LLM hardware. All numbers come from `data/*.csv` (one row per model / GPU / CPU, each with a `Source URL` and `Evidence status` column) and every value was checked against its vendor or Hugging Face source; the log is `data/evidence.csv`.

Repo layout (paths relative to the repo root; find it with `**/offline-llm-hardware-selector/scripts/selector.py`):

| Path | Use |
|---|---|
| `scripts/selector.py` | Answer sizing questions (model → hardware, hardware → models) |
| `scripts/build_web_data.py` | Rebuild the GitHub Pages dataset in `docs/data.json` after CSV edits |
| `docs/index.html` | Interactive model/hardware fit page with source-evidence drawers |
| `scripts/export_public_data.py`, `scripts/excel/*` | Workbook ↔ CSV round trip (only when the Excel workbook is available) |
| `data/*.csv` | models, gpus, cpus, voice_models, quant + evidence log |
| `references/conventions.md` | Sizing formulas, status legend, hybrid/MLA KV conventions |
| `references/sources.md` | Where each parameter comes from, URL patterns, known pitfalls |

Run scripts with the workspace Python (needs `openpyxl`; `pymupdf` optional for NVIDIA PDF datasheets). No API keys.

## Workflow A – answer a sizing / selection question

1. For visual, interactive browsing, serve the static app and use the controls in the page:
   ```bash
   python scripts/build_web_data.py
   python -m http.server 8000 -d docs
   ```
   Open `http://localhost:8000`. The live public page is `https://chinseang88.github.io/offline-llm-hardware-selector/`.
2. Identify model, quantization (default `Q4_K_M`), context length (default 8192), concurrent users (default 1), vendor preference, and target tok/s (default 20). If the question names hardware instead of a model, use `--hw`.
2. Run, e.g.:
   ```
   python scripts/selector.py "Qwen3 32B" --quant Q4_K_M --context 8192 --users 1 --vendor Intel
   python scripts/selector.py --hw "Arc Pro B60" --quant Q5_K_M
   python scripts/selector.py "Llama 3.3 70B" --max-cards 2 --json
   ```
   Unknown model → `--list models` and pick the closest; if it is not in the table, add it (Workflow C) before answering.
3. Report: memory breakdown (weights + KV + runtime), the accelerators/CPUs that fit, estimated tok/s vs target, and the **evidence status** shown in brackets. Quote the Source URL for the model and for any hardware you recommend. Always state that tok/s figures are bandwidth-model estimates, not benchmarks.
4. If the user has an Excel workbook (`Offline_LLM_Hardware_Selector_v*.xlsx`), the same answers exist on sheets `1 Plan a Build` / `2 What Can My HW Run`; the CSVs mirror its `Models`, `GPUs`, `CPUs`, `Voice Models`, `Quant` sheets.

## Workflow B – verify / refresh evidence

1. Run the verifier for the affected table (a name filter limits the run):
   ```
   python scripts/verify_models.py            # all models (≈170 rows, HF API)
   python scripts/verify_gpus.py "RTX 5090"   # one GPU
   python scripts/verify_cpus.py              # all CPUs
   python scripts/verify_misc.py              # voice models, quant table, NVMe constants
   ```
2. Read `out/<table>_evidence.csv`. Status meanings are in `references/conventions.md`. Treat `MISMATCH` as "source disagrees with the table": correct the CSV cell to the source value, keep the old value in the evidence note. Treat `NOT ON PAGE` / `SOURCE UNAVAILABLE` / `DERIVED/ESTIMATE` as unverified – say so; never upgrade them to verified without a quote.
3. Update the row's `Source URL` / `Evidence status` columns and append the new records to `data/evidence.csv` (same 12 columns).
4. Excel workbook present and open? Use `scripts/excel/build_plan.py` then `apply_to_excel.ps1` (writes corrections, cell notes, colour fills, Evidence sheet). Read the script headers first; they attach to the running Excel instance via COM.

## Workflow C – add a model / GPU / CPU

1. Find the primary source first (see `references/sources.md`): HF repo for models (`config.json`, safetensors total, licence); Intel ARK SKU page, AMD/NVIDIA product page or datasheet for hardware.
2. Append the row to the CSV with the column conventions in `references/conventions.md` (hybrid models → full-attention layers; MLA models → 1 KV head × 288 dim; unified-memory boxes → GPU-usable share).
3. Run the matching verifier with the name filter; fix anything not `VERIFIED`; copy the records into `data/evidence.csv`.
4. For GPU/CPU rows also add the URL to the `URLS` map in `verify_gpus.py` / `verify_cpus.py` so future refreshes cover it.

## Guard-rails

- Never invent a spec. If the vendor page is JS-only or retired (many GeForce 10/16/20 pages, Qualcomm), say "vendor page unavailable" and mark the row amber.
- Pre-production or NDA hardware must not enter the public CSVs; keep such rows in a private copy.
- Prices are not tracked here (the workbook's indicative prices were unsourced placeholders).
- Quote values with units exactly as the source states them (GB/s vs GiB, TGP vs TBP, cTDP ranges).
