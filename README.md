# Offline LLM Hardware Selector

Evidence-backed sizing tables and tools for running LLMs **offline** on GPUs, CPUs and unified-memory boxes.

* `data/models.csv` – 170 open-weight LLMs (params, active params, layers, KV heads, head dim, context, licence, HF repo)
* `data/gpus.csv` – 101 accelerators, including 7 explicitly marked Crescent Island (CRI) pre-release planning rows
* `data/cpus.csv` – 56 CPUs / SoCs, including 15 explicitly marked Nova Lake (NVL) pre-release planning rows
* `data/reference_models.csv` – 75 public reference-catalog models with tier, rank, parameter scale, architecture, licence, release, model-card and GGUF links
* `data/voice_models.csv`, `data/quant.csv` – ASR/VAD/diarization checkpoints and quantization bits-per-weight
* `data/evidence.csv` – 2,432 checks: every published parameter with its source URL, quoted source value and status

Every row carries a `Source URL` and an `Evidence status` ("7 verified / 0 corrected / 0 estimate-or-unsourced"). Sources are Hugging Face metadata/configs/model cards, Intel ARK, AMD and NVIDIA product pages, whitepapers and datasheets, Apple tech specs, llama.cpp, and the PCI-SIG / SATA-IO / USB-IF / JEDEC / OCP specifications. Nothing is copied from third-party spec aggregators.

## Browser comparison page

Open the live selector at **https://chinseang88.github.io/offline-llm-hardware-selector/**.

- **Model → Hardware:** exact catalog dropdowns for model/vendor/type, quantization, context/users, target speed, runtime, primary application, KV precision, prompt/answer lengths, CPU offload, host CPU/RAM, GPU quantity/VRAM, host power and PCIe width. Intel-only rebuilds the CPU and GPU dropdowns, results, Database and product-evidence views.
- **Hardware → Models:** independently select CPU and installed RAM, GPU and physical VRAM, GPU quantity, runtime, application, KV precision, offload, headroom and bandwidth utilization, then list models that fit. Intel mode includes 15 Nova Lake planning CPUs and 7 Crescent Island planning GPUs, labelled with pending public-source verification.
- **Database / Evidence:** search, inspect, sort every column, filter, and download the public model/reference-model/GPU/CPU and source-check tables.
- **Know more:** model 360 details explain parameter scale, architecture, context, current memory fit, release/lineage, tier/rank, publisher card, GGUF repository, licence source, and individual fact citations. Internal benchmark results are not published without a public source and complete machine/runtime/workload provenance.
- **Output sorting:** sort every result column; sorting direction toggles by clicking the column header. GPU selections synchronize across model-led, hardware-led and mixed catalog controls.
- **Excel parity:** each result shows the page vs Excel decode estimate, and the Method view summarizes aggregate 8K/32K memory/throughput deltas from the same workbook formulas. A comparison with another external site requires that site's URL.

The static app lives in `docs/`. Rebuild its dataset after changing any CSV:

```bash
python scripts/build_web_data.py
```

With access to the authenticated Client Edge AI reference export, `scripts/import_reference_catalog.py` imports 75 public model rows and 300 public source checks. It also preserves every structured reference record locally under ignored `out/reference_private/` and builds an ignored `docs/private_reference.json`; the local Database view then exposes private leaderboard, timing, runtime, KV, drafter, platform, source-lineage, docs, findings and issue tables. Private/redacted measurements never enter Git or GitHub Pages.

## Command-line selector

```bash
pip install -r requirements.txt

# Which hardware runs Qwen3 32B at Q4_K_M, 8K context, 1 user?
python scripts/selector.py "Qwen3 32B"

# Only Intel accelerators, 2 cards allowed, 32K context, 4 users, Q8
python scripts/selector.py "Llama 3.3 70B" --vendor Intel --max-cards 2 --context 32768 --users 4 --quant Q8_0

# What fits on a given card / CPU?
python scripts/selector.py --hw "Arc Pro B60"
python scripts/selector.py --hw "Ryzen AI Max+ 395" --quant Q5_K_M

# JSON for programs / agents
python scripts/selector.py "gpt-oss-120b" --json
```

Output shows the memory breakdown (weights + FP16 KV cache + runtime), the accelerators and CPU platforms that fit, an estimated decode tok/s per user, and the evidence status of each row. The formulas are listed in [skills/offline-llm-hardware-selector/references/conventions.md](skills/offline-llm-hardware-selector/references/conventions.md). They are **planning estimates**, not benchmarks.

## Re-verifying the data

```bash
python scripts/verify_models.py            # Hugging Face API + config.json/params.json + model card
python scripts/verify_gpus.py  "RTX 5090"  # Intel ARK / AMD / NVIDIA / Apple pages (+ NVIDIA whitepapers)
python scripts/verify_cpus.py              # Intel ARK / AMD / Jetson / Raspberry Pi / Rockchip
python scripts/verify_misc.py              # voice models, quant bits, storage-interface ceilings
```

Each writes `out/<table>_evidence.csv|json` with one record per (row, parameter): sheet value, source value/quote, status, source type, URL, note. Responses are cached in `out/cache/`. Status legend and column conventions (hybrid-model layer counting, MLA KV representation, unified-memory usable share, etc.) are in `references/conventions.md`; URL patterns and known pitfalls per vendor are in `references/sources.md`.

## Using it as an agent skill

`skills/offline-llm-hardware-selector/SKILL.md` packages the workflows for VS Code Copilot / Claude Code style agents: copy that folder into your skills directory (e.g. `~/.copilot/skills/` or `.claude/skills/`) and the agent will use the CSVs + scripts to answer sizing questions with citations, refresh evidence, or add new models/hardware with sources.

## Excel workbook companion

The tables originate from an Excel planner (`Offline_LLM_Hardware_Selector_v27.xlsx`, not included). `scripts/export_public_data.py` exports the CSVs from it; `scripts/excel/` holds the scripts that write corrections, cell notes, colour fills and an *Evidence* sheet back into the open workbook via Excel COM (Windows only).

## Scope and limitations

* Private document IDs, internal pricing and NDA source notes are excluded. Nova Lake and Crescent Island planning rows are included only because they were explicitly requested and are labelled **Pre-release planning / public source pending**; verify against a future public Intel source before commitment. PTL/RZL and other private rows remain excluded.
* Prices are deliberately not tracked.
* Some legacy vendor pages are retired or JavaScript-only (older GeForce, Qualcomm); those rows are marked amber (`SOURCE UNAVAILABLE` / `NOT ON PAGE`) rather than filled from secondary sites.
* Token-rate estimates use a memory-bandwidth model with fixed efficiency factors; real throughput depends on runtime, kernels, batch size and thermals.

## Contributing

Add a row to the relevant CSV with a primary source URL, run the matching `verify_*.py` with a name filter, fix anything that is not `VERIFIED`, and append the new records to `data/evidence.csv`. Please do not submit values without a vendor or Hugging Face source.

## License

MIT – see [LICENSE](LICENSE). Spec data remain © their respective vendors; quoted for interoperability/planning purposes.
