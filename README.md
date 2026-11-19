# Offline LLM Hardware Selector

Evidence-backed sizing tables and tools for running LLMs **offline** on GPUs, CPUs and unified-memory boxes.

* `data/models.csv` – 170 open-weight LLMs (params, active params, layers, KV heads, head dim, context, licence, HF repo)
* `data/gpus.csv` – 94 accelerators (memory, bandwidth, TDP, interface, architecture, CUDA cc)
* `data/cpus.csv` – 41 CPUs / SoCs (cores, base & max power, memory type, bandwidth, max RAM, PCIe lanes)
* `data/voice_models.csv`, `data/quant.csv` – ASR/VAD/diarization checkpoints and quantization bits-per-weight
* `data/evidence.csv` – ~2,000 checks: every parameter above with its source URL, quoted source value and status

Every row carries a `Source URL` and an `Evidence status` ("7 verified / 0 corrected / 0 estimate-or-unsourced"). Sources are Hugging Face metadata/configs/model cards, Intel ARK, AMD and NVIDIA product pages, whitepapers and datasheets, Apple tech specs, llama.cpp, and the PCI-SIG / SATA-IO / USB-IF / JEDEC / OCP specifications. Nothing is copied from third-party spec aggregators.

## Browser comparison page

Open the live selector at **https://chinseang88.github.io/offline-llm-hardware-selector/**.
It compares a selected model’s memory requirement (weights + context KV cache + runtime) against public GPU and CPU rows, then ranks the candidates by estimated decode throughput. Each model and hardware row opens its primary-source evidence trail.

The static app lives in `docs/`. Rebuild its dataset after changing any CSV:

```bash
python scripts/build_web_data.py
```

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

* Pre-production / NDA hardware and any internal documents are excluded from this public dataset.
* Prices are deliberately not tracked.
* Some legacy vendor pages are retired or JavaScript-only (older GeForce, Qualcomm); those rows are marked amber (`SOURCE UNAVAILABLE` / `NOT ON PAGE`) rather than filled from secondary sites.
* Token-rate estimates use a memory-bandwidth model with fixed efficiency factors; real throughput depends on runtime, kernels, batch size and thermals.

## Contributing

Add a row to the relevant CSV with a primary source URL, run the matching `verify_*.py` with a name filter, fix anything that is not `VERIFIED`, and append the new records to `data/evidence.csv`. Please do not submit values without a vendor or Hugging Face source.

## License

MIT – see [LICENSE](LICENSE). Spec data remain © their respective vendors; quoted for interoperability/planning purposes.
