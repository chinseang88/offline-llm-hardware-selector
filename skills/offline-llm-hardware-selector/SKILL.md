---
name: offline-llm-hardware-selector
description: Size and select hardware for running LLMs offline (GPU / CPU / unified-memory boxes) and verify every spec against its primary source. Use whenever someone asks "what hardware runs model X", "will model X fit on GPU Y", "how much VRAM / RAM / tokens-per-second", "which Intel Arc / NVIDIA / AMD card for a 70B model", "KV cache size", "quantization size", or wants to add, refresh or audit GPU/CPU/LLM spec data (memory, bandwidth, TDP, layers, KV heads, context, licence) with evidence from Hugging Face, Intel ARK, AMD, NVIDIA, Apple or standards bodies. Also use for the Offline_LLM_Hardware_Selector Excel workbook (Evidence sheet, Source URL columns) and for the data/*.csv tables in this repo.
---

# Offline LLM Hardware Selector

Evidence-backed sizing of LLM hardware. All numbers come from `data/*.csv` (one row per model / GPU / CPU, each with a `Source URL` and `Evidence status` column) and every value was checked against its vendor or Hugging Face source; the log is `data/evidence.csv`.

Repo layout (paths relative to the repo root; find it with `**/offline-llm-hardware-selector/scripts/selector.py`):

| Path | Use |
|---|---|
| `scripts/selector.py` | Answer sizing questions (model → hardware, hardware → models); `--runtime`, `--kv-bits`, `--task` |
| `scripts/formulas.py` | The sizing formulas (Python mirror of the web app's `docs/formulas.js`) |
| `scripts/reverify_all.py` | One-command re-verification: data, cross-references, page wiring, themes, formulas; `--online` adds Hugging Face repos, source links and newer releases |
| `scripts/build_web_data.py` | Rebuild the GitHub Pages dataset in `docs/data.json` and Excel-parity audit in `docs/parity.json` after CSV edits |
| `scripts/selector_parity.py` | Quantify web/CLI vs Excel memory and throughput formulas for 8K/32K workloads |
| `scripts/export_public_data.py`, `scripts/excel/*` | Workbook ↔ CSV round trip (only when the Excel workbook is available) |
| `scripts/build_public_workbook.py` | Rebuild `excel/Offline_LLM_Hardware_Selector_public.xlsx`, the public Excel edition, from `data/*.csv` only; then `scripts/excel/finalize_public_workbook.ps1` on Windows. Never commit the internal planner workbook |
| `scripts/export_compute_estimates.py` | Refresh `data/compute.csv` (compute estimates for time to first token) from the workbook's Power sheet; the workbook figure of a pre-release row goes only to the ignored `docs/private_compute.json`. `--derive` redoes the figures derived from published columns for rows without a workbook estimate |
| `data/*.csv` | models, gpus, cpus, voice_models, quant, libraries (runtime efficiency), arch_support, task_rules, compute (TFLOPS estimates) + evidence log |
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
Use the website's catalog dropdowns exactly; do not permit arbitrary CPU/GPU/memory/context/user/speed/KV values that are not in the workbook's data-validation lists. Intel-only must prune both CPU and GPU options and show detailed product labels (generation/Lake, core count, base/turbo watts, memory type/bandwidth/RAM, PCIe lanes, GPU VRAM/bandwidth/TDP/interface/architecture).
2. Run, e.g.:
   ```
   python scripts/selector.py "Qwen3 32B" --quant Q4_K_M --context 8192 --users 1 --vendor Intel
   python scripts/selector.py --hw "Arc Pro B60" --quant Q5_K_M
   python scripts/selector.py "Llama 3.3 70B" --max-cards 2 --json
   ```
   Unknown model → `--list models` and pick the closest; if it is not in the table, add it (Workflow C) before answering.
3. Report: memory breakdown (weights + KV + runtime), the accelerators/CPUs that fit, estimated tok/s vs target, and the **evidence status** shown in brackets. Quote the Source URL for the model and for any hardware you recommend. Always state that tok/s figures are bandwidth-model estimates, not benchmarks.
4. If the user has an Excel workbook (`Offline_LLM_Hardware_Selector_v*.xlsx`), comparable answers exist on sheets `1 Plan a Build` / `2 What Can My HW Run`; the CSVs mirror its `Models`, `GPUs`, `CPUs`, `Voice Models`, `Quant`, `Libraries`, `Arch Support` and `Task Rules` sheets. The workbook's runtime allowance and speed denominator differ from the app and CLI (see `references/conventions.md`), so expect small differences and say which one you are quoting.

## Workflow B – verify / refresh evidence

1. Run the verifier for the affected table (a name filter limits the run):
   ```
   python scripts/verify_models.py            # all models (HF API)
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
5. Rebuild and re-check before committing: `python scripts/build_web_data.py`, then `python scripts/reverify_all.py` (must exit 0). A model's `Runtime arch` must have a row in `data/arch_support.csv`.

## Workflow D – re-verify everything after any change

```
python scripts/reverify_all.py            # offline: data, cross-references, page wiring, themes, formulas
python scripts/reverify_all.py --online   # plus Hugging Face repos, source links, newer releases to review
```
Failures must be fixed. "To review" items are judgement calls (for example recent repositories not in the catalog: most are fine-tunes or variants; add only official releases, through Workflow C). When changing a formula, change `docs/formulas.js` and `scripts/formulas.py` together. For the web app, also open the page with `?selftest=1`, which exercises every page and control in the light and dark themes.

## Guard-rails

- Never invent a spec. If the vendor page is JS-only or retired (many GeForce 10/16/20 pages, Qualcomm), say "vendor page unavailable" and mark the row amber.
- Nova Lake (NVL) and Crescent Island (CRI) may be shown only as explicitly labelled `Pre-release planning / public source pending` rows, with internal IDs, pricing, availability notes and private source references removed. PTL/RZL and other private SKUs remain excluded.
- Prices are not tracked here (the workbook's indicative prices were unsourced placeholders).
- Compute (TFLOPS) figures are planning estimates used only for time to first token; always call them estimates. Never add the workbook's compute figure for a `Pre-release planning` row to `data/compute.csv`, the site, tests or documentation: it stays in the ignored `docs/private_compute.json`. Such a row may carry only the figure derived from its published columns (`export_compute_estimates.py --derive`), labelled “derived compute”.
- Quote values with units exactly as the source states them (GB/s vs GiB, TGP vs TBP, cTDP ranges).
