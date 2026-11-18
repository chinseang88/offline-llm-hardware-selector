# Sources: where each parameter comes from and how to fetch it

All fetches are plain HTTPS GETs (no API keys). `scripts/common.py` caches every response under `out/cache/`; delete a cache file to force a refresh.

## Models – Hugging Face

| Need | URL pattern | Notes |
|---|---|---|
| Total params, licence, dtype, base_model, gguf metadata | `https://huggingface.co/api/models/<org>/<repo>` | Works for **gated** repos too (`safetensors.total`, `cardData.license`, `cardData.license_name`, `cardData.base_model`, `gguf.architecture`) |
| Layers / KV heads / head dim / context | `https://huggingface.co/<org>/<repo>/resolve/main/config.json` | 401 when gated → try a non-gated mirror (`unsloth/…`, `NousResearch/…`, sometimes with `Meta-` prefix). Vision/hybrid models keep the LM config under `text_config` / `language_config`. |
| Mistral native checkpoints | `…/resolve/main/params.json` | Keys `n_layers`, `n_kv_heads`, `head_dim`, `kv_lora_rank`, `qk_rope_head_dim`, `moe.num_experts`, `max_position_embeddings` |
| Active params, YaRN context statements | `https://huggingface.co/<org>/<repo>/raw/main/README.md` | Regex for "…B activated/active" and "context length … 128K / 1M" |
| Repo discovery | `https://huggingface.co/api/models?search=<q>&author=<org>&sort=downloads&direction=-1&limit=12` | Prefer the vendor org; GGUF-only repos (prism-ml Bonsai) declare `base_model` → take the architecture from there |

Mirror prefixes tried automatically: `unsloth`, `NousResearch`, `RedHatAI`, `mlx-community`.

## GPUs

| Vendor | Source | Extraction hints |
|---|---|---|
| Intel Arc / Flex / Max | ARK `https://www.intel.com/content/www/us/en/products/sku/<id>/x/specifications.html` (slug ignored) | Server-rendered. Fields: "Memory Size", "Graphics Memory Bandwidth", "TBP", "PCI Express Configurations". Series listings: Arc A 227957, Arc B 240391, Arc Pro A 241358, Arc Pro B 242616, Flex 230021, Max 232874 → `/ark/products/series/<id>/x.html` lists SKU ids. ARK search/autocomplete are JS-only. |
| NVIDIA GeForce 30/40/50 | `nvidia.com/en-us/geforce/graphics-cards/<series>/<family>/` | "Standard Memory Config", "Memory Interface Width", "Total Graphics Power (W) N" (multi-SKU family pages list several values after one label). **No bandwidth** – use whitepapers: Turing, GA102, Ada, RTX Blackwell (`images.nvidia.com/aem-dam/...pdf`; tables in GiB / bare numbers after "Memory Bandwidth (GB/sec)"). 10/16/20-series product pages are retired (404 / redirect). |
| NVIDIA RTX workstation | `nvidia.com/en-us/design-visualization/rtx-<model>/`, `…/products/workstations/professional-desktop-gpus/rtx-pro-<n>/` | "Memory Bandwidth N GB/sec", "Max Power Consumption N W", "Graphics Bus" |
| NVIDIA data-centre | `nvidia.com/en-us/data-center/<l4|l40s|a100|h100|h200|a40|dgx-b200|dgx-b300>/` + datasheet PDFs (A100, T4, P4, P40, P100, V100) | H100/H200 pages list SXM and NVL columns ("Up to 700W (configurable) 350-400W"); DGX pages give 8-GPU totals |
| NVIDIA compute capability | `https://developer.nvidia.com/cuda-gpus` | Text is "CC name name … CC name …"; legacy list is separate |
| AMD Radeon / PRO / Instinct | `amd.com/en/products/graphics/...` (slug uses `rx-7700-xt`, not `rx-7700xt`) | "Max Memory Size", "Memory Bandwidth Up to N GB/s", "Typical Board Power (Desktop) N W"; consumer pages omit the bus; Instinct/PRO have "Bus Type PCIe® 4.0 x16" (strip ®/™) |
| Apple | `support.apple.com/en-us/<id>` tech-spec pages (Mac mini 2024: 121555, Mac Studio 2025: 122211) | "NNNGB/s memory bandwidth", "Maximum continuous power: N W"; no LPDDR type published |

## CPUs

| Vendor | Source | Extraction hints |
|---|---|---|
| Intel Core / Xeon / N-series | ARK SKU pages (series: Core Ultra 200 241071, Meteor Lake 236803, 12th gen 217838, 14th gen 236170, N-series 231819 / 231807, Xeon W 125035, SPR 228622, Xeon 6 240357) | "Total Cores", "Processor Base Power N W", "Maximum Turbo Power" value appears **after a tooltip** as `( 115 W`; "Max Memory Size (dependent on memory type) N GB/TB", "Memory Types Up to LPDDR5/X 7467 MT/s", "Max # of PCI Express Lanes N" |
| AMD Ryzen / EPYC / Threadripper | `amd.com/en/products/processors/<laptop|desktops|server|workstations>/...` | "# of CPU Cores", "Default TDP 55W", "AMD Configurable TDP (cTDP) 45-120W", "Max. Memory 256 GB", "Native PCIe® Lanes (Total/Usable) 28 , 24" (TR PRO adds "/PCIe 5.0"), EPYC "PCIe® 5.0 x128" |
| NVIDIA Jetson | `nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/` | One row per module: "8GB 128-bit LPDDR5 102GB/s", power modes "7W - 15W - 25W" |
| Raspberry Pi 5 | `raspberrypi.com/products/raspberry-pi-5/` | cores, LPDDR4X-4267, PCIe 2.0 x1; no TDP (PSU 27 W) |
| Rockchip RK3588 | `rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html` | "quad-core Cortex-A76 + quad-core Cortex-A55"; memory/lanes not stated |
| Qualcomm Snapdragon X | JS-only product page | `SOURCE UNAVAILABLE` – use the product brief PDF manually |

## Constants

- Quantization bits: `https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp` (raw.githubusercontent.com may hang behind corporate proxies – use the blob page). MXFP4: OCP MX Specification v1.0.
- Storage interface ceilings: PCI-SIG FAQ (Gen3/4/5 per-lane rates), SATA-IO (6 Gb/s → 600 MB/s), USB-IF (USB 3.2 Gen 2, USB4), JEDEC JESD84-B51 (eMMC 5.1 HS400 400 MB/s).

## Pitfalls seen in practice

- Internet Archive snapshots of retired NVIDIA pages are JS shells – do not cite them.
- techpowerup.com returns 403 to scripts and is not a primary source anyway.
- HF `safetensors.total` includes vision encoders; marketing sizes (e.g. "9B") can be 7–25 % lower.
- A single NVIDIA/AMD page may contain footnote figures ("128GB RAM reference board") – anchor on the spec label, not on any nearby number.
- Excel COM: `Comment.Shape.TextFrame.AutoSize` raises 0x800A03EC – set `Shape.Width/Height` instead; `Interior.Color` is BGR.
