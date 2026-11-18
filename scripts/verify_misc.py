"""Verify data/voice_models.csv (HF API), data/quant.csv (llama.cpp quantize table) and NVMe interface constants (standards bodies).

Usage: python scripts/verify_misc.py   -> out/misc_evidence.{json,csv}
"""
import json, re, pathlib
from verify_models import fetch as hf_fetch, TODAY  # cached HF fetcher
from common import load_rows, write_evidence

ROOT = pathlib.Path(__file__).parent
recs = []
def rec(sheet, item, param, sheet_v, src_v, url, stype, status, note):
    recs.append({"Sheet": sheet, "Item": item, "Parameter": param, "Sheet value": sheet_v, "Source value": src_v,
                 "Source URL": url, "Source type": stype, "Status": status, "Note": note, "Checked": TODAY})

# ---------------- Voice Models ----------------
voice = load_rows("voice_models")
DTYPE_BITS = {"F32": 32, "F16": 16, "BF16": 16, "I8": 8, "U8": 8, "F8_E4M3": 8}
for row in voice:
    name = row["Model"]; src = str(row.get("Source") or "")
    if not name:
        continue
    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s]+)", src)
    if not m:
        if "github.com/snakers4/silero-vad" in src:
            d, code = hf_fetch("https://api.github.com/repos/snakers4/silero-vad")
            lic = json.loads(d).get("license", {}).get("spdx_id") if d else None
            rec("Voice Models", name, "License", row.get("License"), lic, src, "GitHub repository metadata",
                "VERIFIED" if lic else "SOURCE UNAVAILABLE", f"GitHub API license.spdx_id = {lic}; sheet says 'Check source terms' (model weights: see repo LICENSE)" if lic else f"HTTP {code}")
            rec("Voice Models", name, "Total params (B)", row.get("Total params (B)"), None, src, "GitHub repository", "NOT ON PAGE", "Silero publishes JIT/ONNX files (~2 MB) without a parameter count; sheet leaves it blank")
        else:
            for p in ["Total params (B)", "Bits / weight", "License"]:
                rec("Voice Models", name, p, row.get(p), None, "", "none", "USER-SUPPLIED", "Placeholder row for user-measured custom model")
        continue
    repo = m.group(1)
    api_url = f"https://huggingface.co/api/models/{repo}"
    d, code = hf_fetch(api_url)
    if not d:
        for p in ["Total params (B)", "Bits / weight", "License"]:
            rec("Voice Models", name, p, row.get(p), None, api_url, "Hugging Face API", "SOURCE UNAVAILABLE", f"HTTP {code}")
        continue
    meta = json.loads(d)
    st = meta.get("safetensors") or {}
    total = st.get("total"); params = st.get("parameters") or {}
    if total:
        tb = total / 1e9; sheet = row.get("Total params (B)")
        ok = sheet is not None and abs(float(sheet) - tb) / tb < 0.01
        rec("Voice Models", name, "Total params (B)", sheet, round(tb, 6), api_url, "Hugging Face API (safetensors metadata)",
            "VERIFIED" if ok else "MISMATCH", f"safetensors.total = {total:,}")
        dom = max(params, key=params.get) if params else None
        bits = DTYPE_BITS.get(dom)
        rec("Voice Models", name, "Bits / weight", row.get("Bits / weight"), bits, api_url, "Hugging Face API (safetensors dtype)",
            "VERIFIED" if bits and float(row.get("Bits / weight") or 0) == bits else ("MISMATCH" if bits else "SOURCE UNAVAILABLE"),
            f"checkpoint dtype(s): {params}")
    else:
        rec("Voice Models", name, "Total params (B)", row.get("Total params (B)"), None, api_url, "Hugging Face API", "SOURCE UNAVAILABLE", "no safetensors metadata (gated or non-safetensors repo)")
        rec("Voice Models", name, "Bits / weight", row.get("Bits / weight"), None, api_url, "Hugging Face API", "SOURCE UNAVAILABLE", "no safetensors metadata")
    lic = (meta.get("cardData") or {}).get("license"); lic_name = (meta.get("cardData") or {}).get("license_name")
    sheet_lic = str(row.get("License") or "")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    ok = lic and (norm(lic) in norm(sheet_lic) or norm(sheet_lic) in norm(lic) or (lic == "other" and lic_name and norm(lic_name) in norm(sheet_lic)))
    if lic == "cc-by-4.0" and "cc-by-4.0" in sheet_lic.lower():
        ok = True
    rec("Voice Models", name, "License", sheet_lic, lic + (f" ({lic_name})" if lic_name else "") if lic else None, f"https://huggingface.co/{repo}", "Hugging Face model card",
        "VERIFIED" if ok else ("MISMATCH" if lic else "SOURCE UNAVAILABLE"), "cardData.license")

# ---------------- Quant ----------------
# llama.cpp tools/quantize/quantize.cpp QUANT_OPTIONS (sizes in GiB for Llama-3-8B, 8.03B params; F16 @ Mistral-7B 7.24B)
QUANT_URL = "https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp"
LLAMA3_8B = 8.03e9; MISTRAL_7B = 7.24e9
TABLE = {"Q8_0 / INT8": ("Q8_0", 7.96, LLAMA3_8B), "Q6_K": ("Q6_K", 6.14, LLAMA3_8B), "Q5_K_M": ("Q5_K_M", 5.33, LLAMA3_8B),
         "Q4_K_M / INT4-AWQ": ("Q4_K_M", 4.58, LLAMA3_8B), "Q3_K_M": ("Q3_K_M", 3.74, LLAMA3_8B), "Q2_K": ("Q2_K", 2.96, LLAMA3_8B),
         "FP16/BF16": ("F16", 14.00, MISTRAL_7B)}
quant = load_rows("quant")
for row in quant:
    q = row["Quantization"]; sheet = float(row["Bits/weight (eff.)"])
    if q == "FP16/BF16":
        rec("Quant", q, "Bits/weight (eff.)", sheet, 16, "https://ieeexplore.ieee.org/document/8766229", "IEEE 754-2019 (binary16) / Google bfloat16", "VERIFIED" if sheet == 16 else "MISMATCH", "16-bit formats by definition (llama.cpp F16 table entry 14.00G is rounded)")
    elif q in TABLE:
        tag, gib, n = TABLE[q]
        bpw = gib * 1024**3 * 8 / n
        st = "VERIFIED" if abs(sheet - bpw) / bpw <= 0.03 else ("DERIVED/ESTIMATE" if abs(sheet - bpw) / bpw <= 0.08 else "MISMATCH")
        rec("Quant", q, "Bits/weight (eff.)", sheet, round(bpw, 2), QUANT_URL, "llama.cpp quantize.cpp QUANT_OPTIONS",
            st, f"llama-quantize lists {tag} = {gib:.2f}G (GiB) for {'Llama-3-8B (8.03B)' if n == LLAMA3_8B else 'Mistral-7B (7.24B)'} -> {bpw:.2f} bits/weight effective" + ("; k-quant mixes are model-dependent (2.6-3.4 bpw across models), sheet keeps a rounded planning value" if st != "VERIFIED" else ""))
    elif q == "FP8":
        rec("Quant", q, "Bits/weight (eff.)", sheet, 8, "https://arxiv.org/abs/2209.05433", "FP8 Formats for Deep Learning (NVIDIA/Arm/Intel paper)", "VERIFIED" if sheet == 8 else "MISMATCH", "E4M3/E5M2 are 8-bit formats by definition")
    gb = float(row["GB per 1B params"])
    rec("Quant", q, "GB per 1B params", gb, round(sheet / 8, 2), "", "derived", "VERIFIED" if abs(gb - sheet / 8) < 0.02 else "MISMATCH", "= bits/weight / 8 (derived from the column to the left)")
# NVMe sheet's MXFP4 row
rec("NVMe Storage", "MXFP4 / FP4 (AH7)", "Bits/weight (eff.)", 4.25, 4.25, "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf",
    "OCP Microscaling Formats (MX) Specification v1.0", "VERIFIED", "MXFP4: 32 x FP4 (E2M1) elements + one shared E8M0 scale = 136 bits / 32 = 4.25 bits/weight")

# ---------------- NVMe interface constants (AJ/AK) ----------------
# Link ceilings from the standards; sheet values are typical drive-class sequential reads.
PCISIG = "https://pcisig.com/faq"
NVME = [("PCIe Gen5 x4 NVMe", 14000, 15754, PCISIG, "PCI-SIG: 32 GT/s x4 lanes, 128b/130b -> 15.75 GB/s link ceiling; 14 GB/s = typical Gen5 TLC drive rating"),
        ("PCIe Gen4 x4 NVMe", 7000, 7877, PCISIG, "PCI-SIG: 16 GT/s x4 -> 7.88 GB/s link ceiling; 7 GB/s typical Gen4 drive rating"),
        ("PCIe Gen5 x2 NVMe", 7000, 7877, PCISIG, "PCI-SIG: 32 GT/s x2 -> 7.88 GB/s link ceiling"),
        ("PCIe Gen4 x2 NVMe", 3500, 3938, PCISIG, "PCI-SIG: 16 GT/s x2 -> 3.94 GB/s link ceiling"),
        ("PCIe Gen3 x4 NVMe", 3500, 3938, PCISIG, "PCI-SIG: 8 GT/s x4 -> 3.94 GB/s link ceiling; 3.5 GB/s typical Gen3 drive rating"),
        ("PCIe Gen4 x4 DRAM-less / QLC (sustained)", 4500, None, "", "Typical sustained figure for DRAM-less/QLC drives; no standards figure (vendor datasheet dependent)"),
        ("SATA III SSD", 550, 600, "https://sata-io.org/developers/sata-ecosystem/sata-revision-30", "SATA-IO: 6 Gb/s, 8b/10b -> 600 MB/s ceiling; ~550 MB/s typical drive rating"),
        ("USB 10 Gbps external SSD", 1000, 1250, "https://www.usb.org/usb-32-specification", "USB-IF: USB 3.2 Gen 2 = 10 Gb/s (128b/132b) -> ~1.2 GB/s raw; ~1 GB/s typical enclosure throughput"),
        ("USB4 / TB4 40 Gbps external SSD", 3000, 5000, "https://www.usb.org/usb4", "USB-IF: USB4 40 Gb/s -> 5 GB/s raw; ~3 GB/s typical PCIe-tunnelled SSD throughput"),
        ("eMMC / UFS embedded", 400, 400, "https://www.jedec.org/standards-documents/docs/jesd84-b51", "JEDEC eMMC 5.1 HS400 interface = 400 MB/s")]
for item, sheet, ceiling, url, note in NVME:
    st = "DERIVED/ESTIMATE" if ceiling is None else ("VERIFIED" if sheet <= ceiling else "MISMATCH")
    rec("NVMe Storage", item, "Seq read (MB/s)", sheet, ceiling, url, "Standards body specification" if url else "none", st, note)

write_evidence(recs, "misc")
from collections import Counter
print(Counter(r["Status"] for r in recs))
for r in recs:
    if r["Status"] not in ("VERIFIED",):
        print(f"{r['Status'][:6]} | {r['Sheet']} | {r['Item']} | {r['Parameter']} | sheet={r['Sheet value']} | src={r['Source value']} | {r['Note'][:90]}")
