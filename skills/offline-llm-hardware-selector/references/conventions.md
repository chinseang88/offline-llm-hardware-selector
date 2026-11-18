# Conventions: sizing formulas, status legend, column rules

## Sizing formulas (identical to the workbook and `scripts/selector.py`)

| Quantity | Formula |
|---|---|
| Weights (GB) | `total_params_B × bits_per_weight / 8 × 1.03` (3 % tensor/metadata overhead) |
| KV cache per token (GB) | `2 × layers × kv_heads × head_dim × 2 bytes / 1e9` (FP16 K and V) |
| Memory needed (GB) | `weights + kv_per_token × context × users + 1.5` (runtime/activations) |
| GPU usable memory | `memory_GB × 0.92` |
| CPU usable RAM | `max_RAM − min(8, 0.25 × max_RAM)` |
| Decode tok/s per user | `eff × bandwidth_GBps × min(cards,8) × (0.8 if cards>1) / (active_params_B × bits/8 + kv_per_token) / users` |
| Efficiency `eff` | Intel GPU 0.55 · Apple 0.65 · other GPUs 0.60 · any CPU 0.55 |
| Largest Q4 model on a card (B params) | `(usable − 3) / 0.667` |

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
