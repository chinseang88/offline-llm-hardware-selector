# Data dictionary

All files are UTF-8 CSV with a header row. Blank = unknown/not applicable. Units are in the column name.

| File | Key column | Columns |
|---|---|---|
| `models.csv` | Model | Vendor · Total params (B) · Active params (B) · Layers · KV heads · Head dim · Max context · Type · License · HF repo · Year · Spec source · Last synced · Runtime arch · Source URL · Evidence status |
| `reference_models.csv` | Model | Tier · Reference rank · Family · Vendor · Total params (B) · Active params (B) · Architecture · License · License class · Context (max) · Release date · Model card URL · Release source URL · GGUF repository URL · License source URL · Source URL · Evidence status |
| `gpus.csv` | Accelerator | Vendor · Memory (GB) · Bandwidth (GB/s) · TDP / board power (W) · Interface · Notes / source · Used / legacy? · Architecture · CUDA compute capability · CUDA toolkit / compute stack · ROCm support · Runtime HW class · Source URL · Evidence status |
| `cpus.csv` | CPU / platform | Class · Cores · Base / default power (W) · Max turbo / max power (W) · Memory · Memory BW (GB/s) · Max RAM (GB) · CPU PCIe lanes · AI accel (iGPU/NPU) · Notes / power source · Runtime HW class · Runtimes that run on this CPU/SoC · Source URL · Evidence status |
| `voice_models.csv` | Model | Stage · Total params (B) · Bits / weight · Source · License · Notes |
| `quant.csv` | Quantization | Bits/weight (eff.) · GB per 1B params · Quality |
| `evidence.csv` | (Sheet, Item, Parameter) | Sheet · Item · Cell · Parameter · Value in sheet (before) · Corrected to · Source value / quote · Status · Source type · Source URL · Note · Checked |

`Evidence status` summarises `evidence.csv` for the row: `<n> verified / <n> corrected / <n> estimate-or-unsourced`.
`Cell` in `evidence.csv` refers to the originating Excel workbook and can be ignored for CSV use.

Column semantics (hybrid-model layer counting, MLA KV heads, unified-memory usable share, AMD usable PCIe lanes, etc.) are defined in `../skills/offline-llm-hardware-selector/references/conventions.md`.

`data/reference_models.csv` contains only public model identity, tier, release, licence and source facts. Authenticated redacted benchmark exports are imported by `scripts/import_reference_catalog.py` into ignored `out/reference_private/`; the optional ignored `docs/private_reference.json` enables those 45+ measurement/audit tables only in the local preview.
