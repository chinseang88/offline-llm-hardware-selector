# Conventions: sizing formulas, status legend, column rules

## Sizing formulas

One implementation, two mirrors: `docs/formulas.js` (web app) and `scripts/formulas.py` (CLI, checks). `scripts/reverify_all.py` fails if they disagree. Three profiles are kept apart on purpose.

### App and CLI (`selector.py`, Model → Hardware, Hardware → Models)

| Quantity | Formula |
|---|---|
| Weights (GB) | `total_params_B × bits_per_weight / 8 × 1.03` (3 % tensor/metadata overhead) |
| KV cache (GB) | `2 × layers × kv_heads × head_dim × kv_bits / 8 / 1e9 × context × users` (KV bits 4, 8 or 16; default 8) |
| Runtime (GB) | `0.07 × weights` |
| Memory needed (GB) | `weights + KV cache + runtime` |
| GPU usable memory | `memory_GB × 0.92 × cards` (headroom adjustable on the hardware page) |
| CPU usable RAM | `RAM − min(8, 0.25 × RAM)` |
| CPU offload pool | `max(0, min(installed RAM, platform max RAM) − 8)`; only llama.cpp, Ollama, LM Studio and IPEX-LLM; never on unified-memory systems |
| Decode tok/s per user | `eff × bandwidth_GBps × (1 + 0.8 × (cards − 1)) / (active_params_B × bits / 8) / users` |
| Decode with offload | `1 / (bytes_per_token × (gpu_share / gpu_rate + cpu_share / cpu_rate)) / users`, `gpu_share = gpu_usable / memory_needed`, `rate = eff × bandwidth` |
| Efficiency `eff` | `data/libraries.csv` cell for the runtime and the hardware's `Runtime HW class` (leading number). NVIDIA below the runtime's minimum CUDA compute capability, or AMD without ROCm, falls back to `Vulkan fallback eff.`; a dash means the runtime does not run there |
| Prefill tok/s | `0.35 × TFLOPS × (1 + 0.8 × (cards − 1)) × 1000 / (2 × active_params_B) / users` on a GPU; `0.35 × 0.6 × TFLOPS × 1000 / (2 × active_params_B) / users` on a CPU (workbook Power sheet: MFU 0.35, 2 GFLOPs per billion active params). With CPU offload the two shares run in series, as for decode |
| Time to first token (s) | `prompt_tokens / prefill_tok_s + 1 / decode_tok_s`. Unknown when the row has no compute estimate |
| Compute `TFLOPS` | `data/compute.csv`: planning estimates without a vendor source (`DERIVED/ESTIMATE`). Pre-release planning rows are never published; they load from the ignored `docs/private_compute.json` when present |
| Task floor | `data/task_rules.csv`: minimum bits by size band (<4B, 4–30B, ≥30B); a vision task also needs a VLM. A model below the floor is not counted as fitting |
| Runtime can load model | `data/arch_support.csv` cell for the model's `Runtime arch`: `N` blocks, `P` warns |

### Workbook "1 Plan a Build" / "2 What Can My HW Run" (shown on the site as the Excel comparison)

| Quantity | Formula |
|---|---|
| Memory needed (GB) | `total × bits / 8 + KV + max(1, 0.1 × (weights + KV))`, context capped at the model's max context |
| Decode tok/s per user | `eff × bandwidth × min(cards, 8) × (0.8 if cards > 1) / (active × bits / 8 + KV cache) / users` |

The app is lower on memory for small models (no 1 GB runtime floor) and higher on speed (the workbook adds the whole KV allocation to the bytes read per token). `scripts/selector_parity.py` reports the size of both differences.

### Workbook "Multi-Stage Pipeline Fit" and "Concurrent Model Selector" (Multi-LLM page)

| Quantity | Formula |
|---|---|
| Usable memory | `memory_GB × cards × (1 − headroom)` |
| Effective bandwidth | `bandwidth × (1 + 0.8 × (cards − 1)) × utilization` (utilization = runtime efficiency unless set by hand) |
| Voice reserve | per GPU stage: resident weights (`params × bits / 8`, or a measured value) + 5 GB activation reserve; 5 GB/s bandwidth. The 5 / 5 figures are planning assumptions. A stage without published weights counts as 0 GB and is flagged `WEIGHTS PENDING` |
| LLM slot memory | `total × bits / 8 × 1.03 + KV cache (1 user, KV bits selectable, FP16 in the workbook) + 0.07 × weights` |
| LLM slot bandwidth | `active × bits / 8 × 1.03 × target tok/s` |
| Fit | slots are added in order on top of the voice reserve: `NO - GPU MEMORY` when the running memory passes usable, else `NO - BANDWIDTH` when running bandwidth passes effective (sum of slots when all answer at once, the largest slot when one at a time) |
| One set | Stage 1 VAD + Stage 2 diarization + Stage 3 ASR + the Stage 4 LLM (one per set), each set with its own loaded copy of every model. Memory per set = voice reserve + the LLM; bandwidth per set = voice bandwidth + LLM bandwidth |
| Concurrent sets (Stage 5) | `min(floor(usable memory / memory per set), floor(effective bandwidth / bandwidth per set))`; no answer while an input needs review. Serving several sets from one shared copy of the weights would allow more and is not modelled |
| First token, alone | `prompt_tokens / prefill_tok_s + 1 / (bandwidth left after voice / bytes per token)` |
| First token, every set busy | the LLMs of all sets share the prefill compute; each decodes at its target: `prompt_tokens / (prefill_tok_s / sets) + 1 / target`. Voice-stage latency is not modelled |

These are planning estimates (memory-bandwidth bound decode, no prefill, no speculative decoding). Always say so.

## Bits per weight (`data/quant.csv`)

From llama.cpp `tools/quantize/quantize.cpp` QUANT_OPTIONS (file sizes in GiB for Llama-3-8B, 8.03 B params → bits = GiB×2^30×8/params): Q8_0 8.5, Q6_K 6.6, Q5_K_M 5.7, Q4_K_M 4.85, Q3_K_M 3.9, Q2_K ≈3.0–3.2 (model-dependent). FP16/BF16 = 16, FP8 = 8 by definition; MXFP4 = 4.25 (OCP MX spec: 32×FP4 + one E8M0 scale).

## Column rules for `data/models.csv`

- **Total params (B)** – safetensors `total` from the HF API (includes embeddings and vision towers). Marketing names (e.g. "0.6B") are *not* used when the checkpoint differs (Qwen3-0.6B = 0.75 B).
- **Active params (B)** – dense: = total. MoE: the vendor's stated active count (model card / repo name `…-A3B`). If only experts/top-k are known the value is `DERIVED/ESTIMATE`.
- **Layers** – `num_hidden_layers`, **except hybrid models** (Gated-DeltaNet / Mamba / linear-attention mixes: Qwen3.5/3.6/3.8, LFM, Granite-H, Nemotron-H, Kimi K3, Jamba): use the number of *full-attention* layers (`layer_types`, `hybrid_override_pattern`, `linear_attn_config.full_attn_layers`, `attn_layer_period`). The per-token KV cache scales only with those; the fixed-size linear-attention state is ignored.
- **KV heads / Head dim** – `num_key_value_heads` / `head_dim` (or hidden/heads). **MLA models** (DeepSeek-V2/V3/V3.2, Kimi K2, GLM-5.3, Mistral Large 3/4): KV heads = 1, head dim = `(kv_lora_rank + qk_rope_head_dim)/2` = 288 so that `2 × 1 × 288 = 576` values/token/layer matches the compressed cache. DeepSeek-V4 (`num_key_value_heads=1, head_dim=512`) uses the config values directly. Sliding-window layers (Gemma) count as normal attention layers (conservative).
- **Max context** – `max_position_embeddings`; where the model card states a larger YaRN/RoPE-scaled length (Qwen2.5 "131,072 tokens") the card value is accepted and noted.
- **License** – HF `cardData.license` (+ `license_name` for `other`). Placeholders ("Check licence", "Model-specific") are not allowed.
- **HF repo** – the official vendor repo; gated repos are fine (metadata is still public).

## Column rules for `data/gpus.csv`

- Memory / Bandwidth / TDP / Interface from the vendor spec page or datasheet; GeForce pages give TGP and bus width but not bandwidth → use the architecture whitepaper (Turing, GA102, Ada, RTX Blackwell) or mark `NOT ON PAGE`.
- Unified-memory systems (Mac, DGX Spark, Ryzen AI Max): **Memory (GB) is the GPU-usable share**, not the system total (noted as `DERIVED/ESTIMATE` with the vendor total in the note). TDP for Apple = typical-load estimate; Apple only publishes "maximum continuous power".
- HGX/OAM modules: per-GPU values = system total ÷ 8.
- CUDA compute capability from developer.nvidia.com/cuda-gpus; SKUs missing from the current list inherit the architecture value (`NOT ON PAGE`).

## Column rules for `data/cpus.csv`

- Intel: ARK "Processor Base Power" / "Maximum Turbo Power" (N-series list only TDP → max power is a board-vendor PL2, amber). Max RAM = ARK "Max Memory Size". PCIe lanes = ARK "Max # of PCI Express Lanes" (includes SoC-tile lanes on Meteor/Arrow Lake H).
- AMD: "Default TDP" / cTDP upper bound; PPT (1.35×TDP) only when the note says so (`DERIVED/ESTIMATE`). PCIe lanes = *usable* count from "Native PCIe Lanes (Total/Usable)". Memory bandwidth is derived (MT/s × 8 B × channels) – AMD does not publish it.
- Embedded boards (Raspberry Pi, Rockchip, Jetson): power is the vendor power-mode range or PSU rating, flagged in the note.

## Evidence status legend (`Evidence status` column, `data/evidence.csv`, workbook cell fills)

| Status | Fill | Meaning |
|---|---|---|
| VERIFIED | green | Value matches the primary source (≤3 % numeric tolerance; licence/interface by text) |
| MISMATCH | red | Source disagreed; value **corrected** to the source, old value kept in the note |
| ADDED | blue | Missing HF repo located and filled |
| DERIVED/ESTIMATE | amber | Computed from vendor data or an explicit modelling choice (see note) |
| NOT ON PAGE | amber | Vendor page reachable but does not state the value |
| SOURCE UNAVAILABLE | amber | Page retired/404/JS-only or gated config |
| PARTIAL | amber | Type verified, speed grade not stated |
| REVIEW | amber | Vendor sources disagree (e.g. ARK "up to x16" vs product brief x8) |
| INTERNAL SOURCE | amber | Pre-production data from an internal document (private copies only) |
| USER-SUPPLIED / NOT APPLICABLE | amber | Placeholder rows / not meaningful for this row |
