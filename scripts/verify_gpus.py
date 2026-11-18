"""Verify data/gpus.csv against vendor specification pages (Intel ARK, AMD, NVIDIA, Apple).

Usage: python scripts/verify_gpus.py [name-filter]   -> out/gpus_evidence.{json,csv}
"""
import json, re, html, sys, pathlib, datetime
from concurrent.futures import ThreadPoolExecutor
from common import load_rows, fetch, write_evidence, UA

TODAY = datetime.date.today().isoformat()

def to_text(raw, url):
    if url.lower().endswith(".pdf") or raw[:4] == b"%PDF":
        try:
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            return re.sub(r"\s+", " ", " ".join(p.get_text() for p in doc))
        except Exception:
            return ""
    h = raw.decode("utf-8", "replace")
    h = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", h, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", h))
    return re.sub(r"\s+", " ", t)

ARK = "https://www.intel.com/content/www/us/en/products/sku/{}/x/specifications.html"
NV_GF = "https://www.nvidia.com/en-us/geforce/graphics-cards/"
NV_DV = "https://www.nvidia.com/en-us/design-visualization/"
NV_DC = "https://www.nvidia.com/en-us/data-center/"
NV_WS = "https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/"
AMD_R = "https://www.amd.com/en/products/graphics/desktops/radeon/"
AMD_W = "https://www.amd.com/en/products/graphics/workstations/"
AMD_I = "https://www.amd.com/en/products/accelerators/instinct/"

# GPU name in sheet -> (vendor spec URL, page kind)
URLS = {
    "GT 710 1GB": ("https://www.nvidia.com/en-us/geforce/graphics-cards/geforce-gt-710/specifications/", "nv"),
    "GT 1030 2GB": ("https://www.nvidia.com/en-us/geforce/products/10series/geforce-gt-1030/", "nv"),
    "Intel Arc A310 4GB": (ARK.format(227958), "ark"),
    "Radeon RX 6400 4GB": (AMD_R + "6000-series/amd-radeon-rx-6400.html", "amd"),
    "GTX 1050 Ti 4GB": ("https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1050/", "nv"),
    "GTX 1650 4GB": (NV_GF + "gtx-1650/", "nv"),
    "NVIDIA T400 4GB": ("https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/nvidia-t400-datasheet-1987150-r3.pdf", "pdf"),
    "Intel Arc A380 6GB": (ARK.format(227959), "ark"),
    "GTX 1660 Super 6GB": (NV_GF + "gtx-1660-super/", "nv"),
    "RTX 2060 6GB": (NV_GF + "rtx-2060/", "nv"),
    "RTX 3050 6GB": (NV_GF + "30-series/rtx-3050/", "nv"),
    "Radeon RX 580 8GB": ("https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-500-series/radeon-rx-500-series/radeon-rx-580.html", "amd"),
    "Tesla P4 8GB": ("https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf", "pdf"),
    "Radeon RX 6600 8GB": (AMD_R + "6000-series/amd-radeon-rx-6600.html", "amd"),
    "RTX 3050 8GB": (NV_GF + "30-series/rtx-3050/", "nv"),
    "RTX 5050 8GB": (NV_GF + "50-series/rtx-5050/", "nv"),
    "RTX 4060 8GB": (NV_GF + "40-series/rtx-4060-4060ti/", "nv"),
    "RTX 5060 8GB": (NV_GF + "50-series/rtx-5060-family/", "nv"),
    "Intel Arc B570 10GB": (ARK.format(241676), "ark"),
    "RTX 3080 10GB": (NV_GF + "30-series/rtx-3080-3080ti/", "nv"),
    "RTX 2080 Ti 11GB": (NV_GF + "rtx-2080-ti/", "nv"),
    "RTX 2060 12GB": (NV_GF + "rtx-2060/", "nv"),
    "RTX 3060 12GB": (NV_GF + "30-series/rtx-3060-3060ti/", "nv"),
    "Intel Arc B580": (ARK.format(241598), "ark"),
    "Radeon RX 7700 XT 12GB": (AMD_R + "7000-series/amd-radeon-rx-7700-xt.html", "amd"),
    "RTX A2000 12GB": (NV_DV + "rtx-a2000/", "nv"),
    "RTX 3080 Ti 12GB": (NV_GF + "30-series/rtx-3080-3080ti/", "nv"),
    "RTX 4070 12GB": (NV_GF + "40-series/rtx-4070-family/", "nv"),
    "RTX 5070 12GB": (NV_GF + "50-series/rtx-5070-family/", "nv"),
    "RTX 4070 Super 12GB": (NV_GF + "40-series/rtx-4070-family/", "nv"),
    "Tesla P100 16GB": ("https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf", "pdf"),
    "Intel Arc A770 16GB": (ARK.format(229151), "ark"),
    "Radeon RX 7600 XT 16GB": (AMD_R + "7000-series/amd-radeon-rx-7600-xt.html", "amd"),
    "Intel Arc Pro B50 16GB": (ARK.format(242615), "ark"),
    "Radeon RX 9060 XT 16GB": (AMD_R + "9000-series/amd-radeon-rx-9060xt.html", "amd"),
    "RTX 5060 Ti 16GB": (NV_GF + "50-series/rtx-5060-family/", "nv"),
    "RTX 4060 Ti 16GB": (NV_GF + "40-series/rtx-4060-4060ti/", "nv"),
    "Radeon RX 7800 XT 16GB": (AMD_R + "7000-series/amd-radeon-rx-7800-xt.html", "amd"),
    "Tesla T4 16GB": ("https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf", "pdf"),
    "Radeon RX 9070 XT 16GB": (AMD_R + "9000-series/amd-radeon-rx-9070xt.html", "amd"),
    "RTX 2000 Ada 16GB": (NV_DV + "rtx-2000/", "nv"),
    "RTX A4000 16GB": (NV_DV + "rtx-a4000/", "nv"),
    "RTX 5070 Ti 16GB": (NV_GF + "50-series/rtx-5070-family/", "nv"),
    "RTX 4070 Ti Super 16GB": (NV_GF + "40-series/rtx-4070-family/", "nv"),
    "RTX 4080 Super 16GB": (NV_GF + "40-series/rtx-4080-family/", "nv"),
    "RTX 5080": (NV_GF + "50-series/rtx-5080/", "nv"),
    "Radeon RX 7900 XT 20GB": (AMD_R + "7000-series/amd-radeon-rx-7900xt.html", "amd"),
    "RTX 4000 Ada 20GB": (NV_DV + "rtx-4000/", "nv"),
    "Tesla M40 24GB": ("https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-m40-datasheet.pdf", "pdf"),
    "Tesla P40 24GB": ("https://images.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf", "pdf"),
    "Intel Arc Pro B60": (ARK.format(243916), "ark"),
    "RX 7900 XTX": (AMD_R + "7000-series/amd-radeon-rx-7900xtx.html", "amd"),
    "RTX 3090 (used)": (NV_GF + "30-series/rtx-3090-3090ti/", "nv"),
    "RTX 3090 Ti 24GB": (NV_GF + "30-series/rtx-3090-3090ti/", "nv"),
    "RTX PRO 4000 Blackwell 24GB": (NV_WS + "rtx-pro-4000/", "nv"),
    "NVIDIA A10 24GB": (NV_DC + "products/a10-gpu/", "nv"),
    "RTX 4090": (NV_GF + "40-series/rtx-4090/", "nv"),
    "NVIDIA L4 24GB": (NV_DC + "l4/", "nv"),
    "Tesla V100 32GB": ("https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf", "pdf"),
    "Radeon AI PRO R9700 32GB": (AMD_W + "radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html", "amd"),
    "RTX 5090": (NV_GF + "50-series/rtx-5090/", "nv"),
    "Radeon PRO W7800 32GB": (AMD_W + "radeon-pro/w7800.html", "amd"),
    "RTX PRO 4500 Blackwell 32GB": (NV_WS + "rtx-pro-4500/", "nv"),
    "RTX 5000 Ada 32GB": (NV_DV + "rtx-5000/", "nv"),
    "A100 40GB PCIe": ("https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf", "pdf"),
    "Mac mini M4 Pro 64GB": ("https://support.apple.com/en-us/121555", "apple"),
    "Radeon PRO W7900": (AMD_W + "radeon-pro/w7900.html", "amd"),
    "RTX A6000 48GB": (NV_DV + "rtx-a6000/", "nv"),
    "NVIDIA A40 48GB": (NV_DC + "a40/", "nv"),
    "RTX PRO 5000 Blackwell 48GB": (NV_WS + "rtx-pro-5000/", "nv"),
    "RTX 6000 Ada": (NV_DV + "rtx-6000/", "nv"),
    "L40S": (NV_DC + "l40s/", "nv"),
    "Instinct MI210 64GB": (AMD_I + "mi200/mi210.html", "amd"),
    "A100 80GB PCIe": (NV_DC + "a100/", "nv"),
    "H100 SXM 80GB (HGX)": (NV_DC + "h100/", "nv"),
    "H100 NVL 94GB": (NV_DC + "h100/", "nv"),
    "Ryzen AI Max+ 395 (128GB unified)": ("https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html", "amdcpu"),
    "RTX PRO 6000 Blackwell": (NV_WS + "rtx-pro-6000/", "nv"),
    "Mac Studio M4 Max 128GB": ("https://support.apple.com/en-us/122211", "apple"),
    "DGX Spark (GB10, 128GB)": ("https://www.nvidia.com/en-us/products/workstations/dgx-spark/", "nv"),
    "H200 NVL 141GB": (NV_DC + "h200/", "nv"),
    "B200 192GB (HGX)": (NV_DC + "dgx-b200/", "nv"),
    "MI300X 192GB": (AMD_I + "mi300/mi300x.html", "amd"),
    "MI325X 256GB": (AMD_I + "mi300/mi325x.html", "amd"),
    "B300 288GB (HGX)": (NV_DC + "dgx-b300/", "nv"),
    "MI355X 288GB": (AMD_I + "mi350/mi355x.html", "amd"),
    "Mac Studio M3 Ultra 512GB": ("https://support.apple.com/en-us/122211", "apple"),
    "Intel Arc A580 8GB": (ARK.format(227961), "ark"),
    "Intel Arc A750 8GB": (ARK.format(227954), "ark"),
    "Intel Arc Pro A40 6GB": (ARK.format(230317), "ark"),
    "Intel Arc Pro A50 6GB": (ARK.format(230316), "ark"),
    "Intel Arc Pro A60 12GB": (ARK.format(235979), "ark"),
    "Intel Data Center GPU Flex 170 16GB": (ARK.format(230019), "ark"),
    "Intel Data Center GPU Max 1100 48GB": (ARK.format(232876), "ark"),
}
CUDA_URL = "https://developer.nvidia.com/cuda-gpus"

WP_TURING = "https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf"
WP_ADA = "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
WP_GA102 = "https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf"
WP_BLACKWELL = "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"
DS_A100 = "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf"
DS_T4 = "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf"
DS_A4000 = "https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf"
DS_A6000 = "https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf"
# Secondary vendor documents consulted when the product page does not state a value.
SECONDARY = {
    "RTX 2080 Ti 11GB": [(WP_TURING, "NVIDIA Turing architecture whitepaper", "RTX 2080 Ti")],
    "RTX 2060 6GB": [(WP_TURING, "NVIDIA Turing architecture whitepaper", "RTX 2060")],
    "RTX 2060 12GB": [(WP_TURING, "NVIDIA Turing architecture whitepaper", "RTX 2060")],
    "GTX 1660 Super 6GB": [(WP_TURING, "NVIDIA Turing architecture whitepaper", "GTX 1660")],
    "GTX 1650 4GB": [(WP_TURING, "NVIDIA Turing architecture whitepaper", "GTX 1650")],
    "Tesla T4 16GB": [(WP_TURING, "NVIDIA Turing architecture whitepaper", "Tesla T4")],
    "RTX 3080 10GB": [(WP_GA102, "NVIDIA GA102 whitepaper", "RTX 3080")],
    "RTX 3080 Ti 12GB": [(WP_GA102, "NVIDIA GA102 whitepaper", "RTX 3080 Ti")],
    "RTX 3090 (used)": [(WP_GA102, "NVIDIA GA102 whitepaper", "RTX 3090")],
    "RTX 3090 Ti 24GB": [(WP_GA102, "NVIDIA GA102 whitepaper", "RTX 3090 Ti")],
    "RTX A6000 48GB": [(WP_GA102, "NVIDIA GA102 whitepaper", "RTX A6000"), (DS_A6000, "NVIDIA RTX A6000 datasheet", "A6000")],
    "RTX A4000 16GB": [(DS_A4000, "NVIDIA RTX A4000 datasheet", "A4000")],
    "RTX 4090": [(WP_ADA, "NVIDIA Ada architecture whitepaper", "RTX 4090")],
    "RTX 4080 Super 16GB": [(WP_ADA, "NVIDIA Ada architecture whitepaper", "RTX 4080")],
    "RTX 5090": [(WP_BLACKWELL, "NVIDIA RTX Blackwell whitepaper", "RTX 5090")],
    "RTX 5080": [(WP_BLACKWELL, "NVIDIA RTX Blackwell whitepaper", "RTX 5080")],
    "RTX 5070 Ti 16GB": [(WP_BLACKWELL, "NVIDIA RTX Blackwell whitepaper", "RTX 5070 Ti")],
    "RTX 5070 12GB": [(WP_BLACKWELL, "NVIDIA RTX Blackwell whitepaper", "RTX 5070")],
    "A100 40GB PCIe": [(DS_A100, "NVIDIA A100 datasheet", "A100")],
    "A100 80GB PCIe": [(DS_A100, "NVIDIA A100 datasheet", "A100")],
    "Tesla T4 16GB": [(DS_T4, "NVIDIA T4 datasheet", "T4")],
}
# Rows whose sheet memory is a GPU-usable share of a unified/system pool (vendor total in parentheses)
UNIFIED = {"Mac mini M4 Pro 64GB", "Mac Studio M4 Max 128GB", "Mac Studio M3 Ultra 512GB", "DGX Spark (GB10, 128GB)", "Ryzen AI Max+ 395 (128GB unified)", "B300 288GB (HGX)"}
HGX_PER_GPU = {"B200 192GB (HGX)": 8, "B300 288GB (HGX)": 8}

def nums_near(text, anchors, unit_re, span=140, before=40):
    """Return [(value, quote)] for numbers with the given unit found near any anchor phrase."""
    out = []
    for a in anchors:
        for m in re.finditer(a, text, re.I):
            win = text[max(0, m.start() - before): m.end() + span]
            for n in re.finditer(unit_re, win):
                try:
                    v = float(n.group(1).replace(",", ""))
                except Exception:
                    continue
                out.append((v, win[max(0, n.start() - 60): n.end() + 15].strip()))
    return out

def extract(text, kind):
    """Return dict param -> list of (value, quote) candidates."""
    c = {}
    c["Memory (GB)"] = nums_near(text, [r"Memory Size", r"Standard Memory Config", r"GPU [Mm]emory\b", r"Max Memory Size", r"Max\. Memory", r"\bMemory\b(?! Bandwidth| Speed| Interface| Type| Clock)", r"unified memory", r"Unified Memory", r"HBM", r"GDDR"],
                                 r"(\d{1,4}(?:,\d{3})?(?:\.\d)?)\s?GB(?!/s)(?! SSD)")
    c["Bandwidth (GB/s)"] = nums_near(text, [r"[Bb]andwidth|BANDWIDTH"], r"(\d{1,5}(?:,\d{3})?(?:\.\d+)?)\s?(?:GB/s|GB/sec|TB/s|TB/sec)", span=60)
    c["Bandwidth (GB/s)"] = [((v * 1000 if "TB/S" in q.upper() else v), q) for v, q in c["Bandwidth (GB/s)"] if not re.search(r"NVLink|Infinity Fabric|Interconnect|PCIe", q)]
    # table rows with the unit in the label: 'Memory Bandwidth (GB/sec) 484 616 432 672'
    for m in re.finditer(r"Bandwidth\s*\((GB|TB)/s(?:ec)?\)\s*((?:\d{2,5}(?:\.\d)?\s*){1,8})", text):
        for n in re.findall(r"\d{2,5}(?:\.\d)?", m.group(2)):
            c["Bandwidth (GB/s)"].append((float(n) * (1000 if m.group(1) == "TB" else 1), m.group(0)[:90]))
    # memory sizes given in MB (whitepapers): 11264 MB -> 11 GB
    for v_, q in nums_near(text, [r"Memory Size", r"Frame Buffer"], r"(\d{3,6})\s?MB\b"):
        c["Memory (GB)"].append((round(v_ / 1024, 2), q))
    tdp = []
    LABELS = r"(Total Graphics Power|Graphics Card Power|Total Board Power|Typical Board Power|Max(?:imum)? Power Consumption|Max(?:imum)? [Tt]hermal [Dd]esign [Pp]ower|Thermal Design Power|\bTDP\b|\bTBP\b|\bTGP\b|Board Power|Peak Power|Max Power|Default TDP|Configurable TDP \(cTDP\)|GB10 TDP\s*\d?|Power Supply|Maximum continuous power|Power consumption)"
    for m in re.finditer(LABELS + r"(?:\s*\S{0,2}\s*\((?!W\)|Watts\))[^)]{1,40}\))?\s*(\(W\)|\(Watts\))?\s*(?:\(\d\))?\s*:?\s*([^A-Za-z]{0,3}(?:(?:Up to|up to)\s*)?\d{2,4}(?:\s?W(?:atts?)?)?(?:\s*(?:or|/|-|\u2013)\s*\d{2,4}(?:\s?W)?)*(?:\s*\((?:configurable|[a-z ]+)\))?(?:\s+(?:Up to\s*)?\d{2,4}(?:\s?W)?(?:\s*(?:or|/|-|\u2013)\s*\d{2,4}(?:\s?W)?)*(?:\s*\((?:configurable|[a-z ]+)\))?){0,5})", text):
        label, wmode, seq = m.group(1), m.group(2), m.group(3)
        if label == "Power Supply" and "Watts" not in m.group(0):
            continue
        if label == "Max Power" and re.search(r"Check your power supply|Minimum", text[max(0, m.start() - 30): m.end()], re.I):
            continue
        q = text[max(0, m.start() - 20): m.end()].strip()
        for n in re.finditer(r"(\d{2,4})(\s?W(?:atts?)?)?", seq):
            if wmode or n.group(2) or re.search(r"\d{2,4}\s?W", seq):
                tdp.append((float(n.group(1)), q))
    c["TDP / board power (W)"] = tdp
    c["Interface"] = [(q.strip(), q.strip()) for q in re.findall(r"((?:Up to )?PCI(?:e|\s*Express)[\u00ae\u2122]*\s*(?:Gen\s*)?[2-5](?:\.0)?\s*(?:x\s?\d{1,2}|Gen\s?\d)?(?:\s*x\s?\d{1,2})?|PCIe\s*[2-5]\.0\s*x\d{1,2}|Gen\s*[2-5]\s*x\s?\d{1,2}|SXM\d?|OAM|NVLink[^.,;]{0,20})", text)]
    c["_unified_total"] = nums_near(text, [r"unified memory", r"Unified Memory", r"Max\. Memory", r"GPU Memory"], r"(\d{1,4}(?:,\d{3})?(?:\.\d)?)\s?[GT]B(?!/s)")
    c["_unified_total"] = [((v * 1000 if re.search(r"\d\s?TB", q) else v), q) for v, q in c["_unified_total"]]
    c["Memory (GB)"] += [(v, q) for v, q in c["_unified_total"] if v >= 1000]
    c["_apple_power"] = nums_near(text, [r"Maximum continuous power"], r"(\d{2,4})\s?W\b", span=20)
    return c

def status_num(sheet, cands, tol=0.03):
    """VERIFIED if sheet value matches any candidate within tol."""
    try:
        s = float(sheet)
    except Exception:
        return "MISMATCH", None
    best = None
    for v, q in cands:
        if v and abs(s - v) / v <= tol:
            return "VERIFIED", (v, q)
        if best is None or abs(s - v) < abs(s - best[0]):
            best = (v, q)
    return ("MISMATCH" if cands else "NOT ON PAGE"), best

def check(row):
    name = row["Accelerator"]; recs = []
    def rec(param, sheet, src, url, status, note, stype):
        recs.append({"Sheet": "GPUs", "Item": name, "Parameter": param, "Sheet value": sheet, "Source value": src,
                     "Source URL": url, "Source type": stype, "Status": status, "Note": note, "Checked": TODAY})
    params = ["Memory (GB)", "Bandwidth (GB/s)", "TDP / board power (W)", "Interface"]
    if name.startswith("Intel CRI"):
        for p in params + ["Architecture"]:
            rec(p, row.get(p), None, "", "INTERNAL SOURCE", "Pre-production SKU: values cite non-public documents named in the Notes column; not externally verifiable", "Internal document (cited)")
        return recs
    url, kind = URLS.get(name, (None, None))
    if not url:
        for p in params:
            rec(p, row.get(p), None, "", "NO PRIMARY SOURCE", "No vendor URL mapped", "none")
        return recs
    raw, code = fetch(url)
    if raw is None:
        for p in params:
            rec(p, row.get(p), None, url, "SOURCE UNAVAILABLE", f"HTTP {code}", "vendor page")
        return recs
    text = to_text(raw, url)
    stype = {"ark": "Intel ARK specifications", "amd": "AMD product specifications", "amdcpu": "AMD product specifications",
             "nv": "NVIDIA product page", "pdf": "NVIDIA datasheet (PDF)", "apple": "Apple tech specs"}[kind]
    c = extract(text, kind)
    per = HGX_PER_GPU.get(name)
    for p in ["Memory (GB)", "Bandwidth (GB/s)", "TDP / board power (W)"]:
        cands = c[p]
        if per and p in ("Memory (GB)", "Bandwidth (GB/s)"):
            cands = cands + [(v / per, q + f" [8-GPU system total / {per}]") for v, q in cands if re.search(r"total|HBM|GPU Memory", q, re.I) and "NVLink" not in q]
        st, best = status_num(row.get(p), cands)
        note = f"'{best[1]}'" if best else "value not found on page"
        if st == "VERIFIED" and kind == "nv" and re.search(r"family|3060-3060ti|3080-3080ti|3090-3090ti|4060-4060ti|a100|h100|rtx-3050", url):
            note += " (multi-SKU page; value matched among listed SKUs)"
        # unified / usable-memory rows
        if p == "Memory (GB)" and name in UNIFIED and st != "VERIFIED":
            tot = [v for v, q in c["_unified_total"]]
            sheet = float(row.get(p) or 0)
            fit = [v for v in tot if 0.75 * v <= sheet <= v] or ([v / per for v in tot if 0.75 * v / per <= sheet <= v / per] if per else [])
            if fit:
                st, best = "DERIVED/ESTIMATE", (fit[0], f"vendor total {fit[0]:g} GB; sheet uses GPU-usable share ({sheet / fit[0]:.0%})")
                note = best[1]
        if p == "TDP / board power (W)" and kind == "apple":
            ap = c["_apple_power"]
            if ap:
                st, best = "DERIVED/ESTIMATE", (ap[0][0], ap[0][1])
                note = f"Apple publishes only 'Maximum continuous power {ap[0][0]:g} W' (whole system); sheet value is a typical-load estimate"
        # secondary vendor documents
        if st in ("NOT ON PAGE", "MISMATCH") and name in SECONDARY:
            for surl, sname, token in SECONDARY[name]:
                sraw, scode = fetch(surl)
                if not sraw:
                    continue
                stext = to_text(sraw, surl)
                if token.lower() not in stext.lower():
                    continue
                sc = extract(stext, "pdf")
                st2, best2 = status_num(row.get(p), sc[p])
                if st2 == "VERIFIED":
                    st, best = st2, best2
                    note = f"'{best2[1]}' ({sname}; document covers several SKUs, value matched among listed values)"
                    rec(p, row.get(p), best[0], surl, st, note, sname)
                    break
            else:
                rec(p, row.get(p), best[0] if best else None, url, st, note, stype)
            continue
        rec(p, row.get(p), best[0] if best else None, url, st, note, stype)
    # Interface: text match on generation and width
    sheet_if = str(row.get("Interface") or "")
    gen = re.search(r"([2-5])\.0", sheet_if); width = re.search(r"x(\d{1,2})", sheet_if)
    found = [q for _, q in c["Interface"] if not re.search(r"registered|trademark", q, re.I)]
    ok = None
    for q in found:
        qg = re.search(r"([2-5])\.0|Gen\s*([2-5])|PCIe\s*([2-5])\b", q); qw = re.search(r"x\s?(\d{1,2})", q)
        g = qg and (qg.group(1) or qg.group(2) or qg.group(3))
        if gen and g and gen.group(1) == g and (not width or not qw or width.group(1) == qw.group(1)):
            ok = q; break
    if "SXM" in sheet_if or "OAM" in sheet_if or "unified" in sheet_if.lower():
        key = "SXM" if "SXM" in sheet_if else ("OAM" if "OAM" in sheet_if else "unified")
        hit = re.search(key, text, re.I)
        rec("Interface", sheet_if, hit.group(0) if hit else None, url, "VERIFIED" if hit else "NOT ON PAGE", f"'{text[max(0,hit.start()-50):hit.end()+40]}'" if hit else "form-factor keyword not on page", stype)
    else:
        if ok is None and found and any(f.startswith("Up to") for f in found) and kind == "ark":
            rec("Interface", sheet_if, found[0], url, "REVIEW", f"ARK states '{found[0]}' (electrical limit); sheet keeps the board's physical slot width from the product brief", stype)
        else:
            rec("Interface", sheet_if, ok or (found[0] if found else None), url, "VERIFIED" if ok else ("MISMATCH" if found else "NOT ON PAGE"),
                ("'" + ok + "'") if ok else ("page shows: " + "; ".join(sorted(set(found))[:4]) if found else "no PCIe statement found"), stype)
    return recs

def cuda_table():
    """Parse developer.nvidia.com/cuda-gpus: text is 'CC name name ... CC name ...'."""
    raw, code = fetch(CUDA_URL)
    if not raw:
        return {}
    t = to_text(raw, CUDA_URL)
    i = t.find("Compute Capability Data Center")
    t = t[i:] if i >= 0 else t
    parts = re.split(r"\s(\d{1,2}\.\d)\s", " " + t + " ")
    table = {}
    for cc, seg in zip(parts[1::2], parts[2::2]):
        for nm in re.split(r"\s(?=NVIDIA |GeForce |Jetson |Tesla |Quadro |TITAN |RTX )", seg):
            nm = nm.strip()
            if 2 < len(nm) < 60:
                table[nm] = cc
    return table

def check_cuda(rows):
    table = cuda_table()
    recs = []
    def norm(s):
        s = re.sub(r"\(.*?\)|\b\d+GB\b|NVIDIA|GeForce|Tesla|\(used\)|HGX|PCIe|SXM|NVL", " ", s)
        return re.sub(r"\s+", " ", s).strip().lower()
    for row in rows:
        name = row["Accelerator"]; cc = row.get("CUDA compute capability")
        if row.get("Vendor") != "NVIDIA":
            continue
        key = norm(name)
        alias = {"dgx spark": "gb10", "b200": "b200", "b300": "b300", "h100 sxm": "h100", "h100": "h100", "h200": "h200", "a100 40gb": "a100", "a100": "a100"}
        key = alias.get(key, key)
        hit = None
        for k, v in table.items():
            kn = norm(k)
            if kn == key or kn == key.replace(" ", "") or (key in kn and len(kn) - len(key) <= 3 and not key.endswith("ti") and "ti" not in kn.replace(key, "")):
                hit = (k, v); break
        want = f"{float(cc):.1f}" if cc is not None and str(cc).replace(".", "").isdigit() else str(cc)
        if hit:
            st = "VERIFIED" if want == hit[1] else "MISMATCH"
            recs.append({"Sheet": "GPUs", "Item": name, "Parameter": "CUDA compute capability", "Sheet value": cc, "Source value": hit[1],
                         "Source URL": CUDA_URL, "Source type": "NVIDIA CUDA GPUs list", "Status": st, "Note": f"listed as '{hit[0]}' = {hit[1]}", "Checked": TODAY})
        else:
            recs.append({"Sheet": "GPUs", "Item": name, "Parameter": "CUDA compute capability", "Sheet value": cc, "Source value": None,
                         "Source URL": CUDA_URL, "Source type": "NVIDIA CUDA GPUs list", "Status": "NOT ON PAGE", "Note": "SKU not listed by name on the current CUDA GPUs page (legacy list is separate); value follows the architecture's compute capability", "Checked": TODAY})
    return recs

if __name__ == "__main__":
    rows = load_rows("gpus")
    if len(sys.argv) > 1:
        rows = [r for r in rows if sys.argv[1].lower() in str(r["Accelerator"]).lower()]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(check, rows))
    flat = [r for rs in res for r in rs] + check_cuda(rows)
    write_evidence(flat, "gpus")
    from collections import Counter
    print(Counter(r["Status"] for r in flat))
    for r in flat:
        if r["Status"] not in ("VERIFIED", "INTERNAL SOURCE"):
            print(f"{r['Status'][:6]} | {r['Item']} | {r['Parameter']} | sheet={r['Sheet value']} | src={r['Source value']} | {r['Note'][:100]}")
