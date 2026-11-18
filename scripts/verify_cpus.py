"""Verify data/cpus.csv against vendor specification pages (Intel ARK, AMD, NVIDIA, Apple, Raspberry Pi, Rockchip, Qualcomm).

Usage: python scripts/verify_cpus.py [name-filter]   -> out/cpus_evidence.{json,csv}
"""
import json, re, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
from verify_gpus import fetch, to_text, nums_near, ARK, TODAY
from common import load_rows, write_evidence

ROOT = pathlib.Path(__file__).parent
AMD_L = "https://www.amd.com/en/products/processors/laptop/ryzen/"
AMD_D = "https://www.amd.com/en/products/processors/desktops/ryzen/"
AMD_S = "https://www.amd.com/en/products/processors/server/epyc/"
AMD_W = "https://www.amd.com/en/products/processors/workstations/ryzen-threadripper/"
JETSON = "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/"

URLS = {
    "Raspberry Pi 5 8GB": ("https://www.raspberrypi.com/products/raspberry-pi-5/", "rpi"),
    "Raspberry Pi 5 16GB": ("https://www.raspberrypi.com/products/raspberry-pi-5/", "rpi"),
    "Intel N100 mini-PC": (ARK.format(231803), "ark"),
    "Intel N150 mini-PC": (ARK.format(241636), "ark"),
    "Jetson Orin Nano Super 8GB": (JETSON, "jetson"),
    "Rockchip RK3588 SBC 32GB": ("https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html", "rk"),
    "Jetson Orin NX 16GB": (JETSON, "jetson"),
    "Intel N305 mini-PC": (ARK.format(231805), "ark"),
    "NVIDIA Jetson AGX Orin 64GB": (JETSON, "jetson"),
    "Core Ultra 7 258V (Lunar Lake)": (ARK.format(240957), "ark"),
    "Apple M4 Mac mini 32GB": ("https://support.apple.com/en-us/121555", "apple"),
    "Snapdragon X Elite X1E-80": ("https://www.qualcomm.com/products/mobile/snapdragon/laptops-and-tablets/snapdragon-x-elite", "qc"),
    "Core Ultra 5 125H (Meteor Lake)": (ARK.format(236848), "ark"),
    "Ryzen AI 7 350": (AMD_L + "ai-300-series/amd-ryzen-ai-7-350.html", "amd"),
    "Ryzen AI 9 HX 370": (AMD_L + "ai-300-series/amd-ryzen-ai-9-hx-370.html", "amd"),
    "Apple M4 Pro Mac mini 64GB": ("https://support.apple.com/en-us/121555", "apple"),
    "Mac Studio M4 Max 128GB": ("https://support.apple.com/en-us/122211", "apple"),
    "Ryzen 7 8845HS": (AMD_L + "8000-series/amd-ryzen-7-8845hs.html", "amd"),
    "Core Ultra 9 285H": (ARK.format(241747), "ark"),
    "Ryzen AI Max 385 (64GB)": (AMD_L + "ai-300-series/amd-ryzen-ai-max-385.html", "amd"),
    "Core Ultra 9 275HX": (ARK.format(242293), "ark"),
    "Ryzen AI Max+ 395": (AMD_L + "ai-300-series/amd-ryzen-ai-max-plus-395.html", "amd"),
    "Core i5-12400": (ARK.format(134586), "ark"),
    "Ryzen 5 7600": (AMD_D + "7000-series/amd-ryzen-5-7600.html", "amd"),
    "Ryzen 7 8700G": (AMD_D + "8000-series/amd-ryzen-7-8700g.html", "amd"),
    "Ryzen 7 9700X": (AMD_D + "9000-series/amd-ryzen-7-9700x.html", "amd"),
    "Core i7-14700": (ARK.format(236781), "ark"),
    "NVIDIA DGX Spark (GB10)": ("https://www.nvidia.com/en-us/products/workstations/dgx-spark/", "nv"),
    "Core Ultra 7 265K": (ARK.format(241063), "ark"),
    "Core Ultra 9 285K": (ARK.format(241060), "ark"),
    "Ryzen 9 9950X": (AMD_D + "9000-series/amd-ryzen-9-9950x.html", "amd"),
    "EPYC 4564P": (AMD_S + "4th-generation-4004-series/amd-epyc-4564p.html", "amd"),
    "Xeon w5-2555X (W-2500)": (ARK.format(236881), "ark"),
    "Xeon w7-2595X (W-2500)": (ARK.format(236878), "ark"),
    "Xeon Gold 6430 (SPR, 1S)": (ARK.format(231737), "ark"),
    "Threadripper 9960X": (AMD_W + "9000-series/amd-ryzen-threadripper-9960x.html", "amd"),
    "Threadripper PRO 9975WX": (AMD_W + "9000-wx-series/amd-ryzen-threadripper-pro-9975wx.html", "amd"),
    "EPYC 9555 (Turin, 1S)": (AMD_S + "9005-series/amd-epyc-9555.html", "amd"),
    "Xeon w9-3595X (W-3500)": (ARK.format(240482), "ark"),
    "Xeon 6 6980P (1S)": (ARK.format(240777), "ark"),
    "EPYC 9965 (Turin Dense, 1S)": (AMD_S + "9005-series/amd-epyc-9965.html", "amd"),
}
STYPE = {"ark": "Intel ARK specifications", "amd": "AMD product specifications", "nv": "NVIDIA product page", "jetson": "NVIDIA Jetson Orin module specs",
         "apple": "Apple tech specs", "rpi": "Raspberry Pi product page", "rk": "Rockchip product page", "qc": "Qualcomm product page"}
PARAMS = ["Cores", "Base / default power (W)", "Max turbo / max power (W)", "Memory", "Memory BW (GB/s)", "Max RAM (GB)", "CPU PCIe lanes"]

def extract(text, kind, name):
    c = {p: [] for p in PARAMS}
    # cores
    c["Cores"] += nums_near(text, [r"Total Cores", r"# of CPU Cores", r"CPU Cores", r"-core CPU", r"core (?:Arm|ARM)", r"Cortex", r"Oryon", r"CPU\b"], r"(?<![\d.])(\d{1,3})(?=\s*(?:cores?|-core|x\s*(?:Arm|ARM|Cortex)|$|\W))", span=30, before=60)
    c["Cores"] += [(float(m.group(1)), m.group(0)) for m in re.finditer(r"(\d{1,3})[- ]core (?:CPU|processor|Arm|ARM)", text)]
    c["Cores"] += [(float(m.group(1)), m.group(0)) for m in re.finditer(r"(?:Total Cores|# of CPU Cores)\D{0,8}(\d{1,3})\b", text)]
    c["Cores"] += [(float(m.group(1)), m.group(0)) for m in re.finditer(r"(\d{1,2})x\s*(?:Arm|ARM)?\s*Cortex", text)]
    c["Cores"] += [(float(m.group(1)) + float(m.group(2)), m.group(0)) for m in re.finditer(r"(\d)x\s*(?:Arm\s*)?Cortex-A\d+\D{0,30}?(\d)x\s*(?:Arm\s*)?Cortex-A\d+", text)]
    WORDS = {"dual": 2, "quad": 4, "hexa": 6, "octa": 8, "deca": 10}
    for m in re.finditer(r"(dual|quad|hexa|octa|deca)-core", text, re.I):
        c["Cores"].append((float(WORDS[m.group(1).lower()]), m.group(0)))
    for m in re.finditer(r"(dual|quad|hexa|octa)-core[^.]{0,40}?(dual|quad|hexa|octa)-core", text, re.I):
        c["Cores"].append((float(WORDS[m.group(1).lower()] + WORDS[m.group(2).lower()]), m.group(0)[:80]))
    # power: label followed by one or more values / ranges (Jetson lists several modules after one label)
    pw = []
    for m in re.finditer(r"(Processor Base Power|Maximum Turbo Power|Max Turbo Power|Minimum Assured Power|Maximum Assured Power|\bTDP\b|Default TDP|Configurable TDP \(cTDP\)|cTDP|Thermal Design Power|Power|Max(?:imum)? continuous power|Power Supply|power modes?|Power consumption)\s*(?:\(TDP\))?\s*:?\s*((?:[\s,|/\-\u2013]{0,3}(?:Up to\s*)?\d{1,4}\s?W?(?:\s*(?:-|\u2013|to)\s*\d{1,4}\s?W)?\s?(?:W(?:atts?)?)?){1,30})", text):
        seq = m.group(2)
        if "W" not in seq:
            continue
        q = text[max(0, m.start() - 20): m.end() + 5].strip()
        for n in re.findall(r"\d{1,4}", seq):
            pw.append((float(n), q))
    # ARK renders a tooltip between the label and the value: 'Maximum Turbo Power ... ( 115 W Minimum Assured Power'
    for m in re.finditer(r"\(\s*(\d{1,4})\s?W\b", text):
        back = text[max(0, m.start() - 700): m.start()]
        labs = list(re.finditer(r"Maximum Turbo Power|Processor Base Power|Minimum Assured Power|Maximum Assured Power|\bTDP\b", back))
        if labs:
            pw.append((float(m.group(1)), f"{labs[-1].group(0)} ... ( {m.group(1)} W"))
    c["Base / default power (W)"] = pw; c["Max turbo / max power (W)"] = pw
    # memory type
    c["Memory"] = [(m.group(0).replace("/", ""), m.group(0)) for m in re.finditer(r"(?:LPDDR5/?X?|LPDDR4/?X?|DDR5|DDR4)(?:[- ]?\d{4})?(?:\s*(?:MT/s|MHz))?", text, re.I)]
    # bandwidth
    c["Memory BW (GB/s)"] = nums_near(text, [r"Bandwidth", r"bandwidth"], r"(\d{1,4}(?:\.\d+)?)\s?GB/s", span=40, before=60)
    c["Memory BW (GB/s)"] += [(float(m.group(1)), m.group(0)) for m in re.finditer(r"LPDDR5X?(?: \(\+ ECC\))?\s*(\d{2,4}(?:\.\d)?)\s?GB/s", text)]
    # max ram: prefer explicit max-memory labels; fall back to generic mentions
    strong = []
    for m in re.finditer(r"(?:Max Memory Size(?: \(dependent on memory type\))?|Max\.? Memory|Max Memory|Memory Capacity|Maximum memory)\D{0,6}(\d{1,4})\s?(GB|TB)\b", text):
        strong.append((float(m.group(1)) * (1024 if m.group(2) == "TB" else 1), m.group(0)))
    weak = []
    for m in re.finditer(r"(\d{1,4})\s?(GB|TB)(?!/s)(?! SSD)(?! storage)", text):
        ctx = text[max(0, m.start() - 70): m.end() + 30]
        if re.search(r"unified memory|LPDDR|RAM|memory", ctx, re.I):
            weak.append((float(m.group(1)) * (1024 if m.group(2) == "TB" else 1), ctx.strip()))
    strong += [(float(m.group(1)), m.group(0)) for m in re.finditer(r"(\d{1,3})GB \d{2,3}-bit LPDDR5", text)]
    c["Max RAM (GB)"] = strong or weak
    # pcie lanes (AMD gives Total/Usable[/PCIe 5.0]: accept each)
    lanes = []
    for m in re.finditer(r"(?:Max # of PCI Express Lanes|Native PCIe[\u00ae\u2122]* Lanes \([^)]*\)|PCIe[\u00ae\u2122]* Lanes|PCI Express Lanes)\D{0,6}(\d{1,3})(?:\s*[/,]\s*(\d{1,3}))?(?:\s*[/,]\s*(\d{1,3}))?", text):
        for g, tag in ((1, ""), (2, " [usable]"), (3, " [PCIe 5.0 lanes]")):
            if m.group(g):
                lanes.append((float(m.group(g)), m.group(0) + tag))
    for m in re.finditer(r"(?:PCI Express(?: Gen)? \d(?:\.0)?|PCIe[\u00ae\u2122]* (?:Gen ?)?\d(?:\.0)?|PCIe)\s*x\s?(\d{1,3})\b|(\d{1,3})\s*(?:lanes|-lane)s?\s+(?:of\s+)?PCIe|PCIe[^.]{0,25}?(\d{1,3})\s*(?:lanes|-lane)", text):
        v = m.group(1) or m.group(2) or m.group(3)
        lanes.append((float(v), m.group(0)))
    c["CPU PCIe lanes"] = lanes
    return c

def status_num(sheet, cands, tol=0.03):
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
    name = row["CPU / platform"]; recs = []
    def rec(param, sheet, src, url, status, note, stype):
        recs.append({"Sheet": "CPUs", "Item": name, "Parameter": param, "Sheet value": sheet, "Source value": src,
                     "Source URL": url, "Source type": stype, "Status": status, "Note": note, "Checked": TODAY})
    if re.match(r"(NVL|PTL|RZL)", name):
        for p in PARAMS:
            rec(p, row.get(p), None, "", "INTERNAL SOURCE", "Pre-production SKU: values cite non-public documents named in the Notes column; not externally verifiable", "Internal document (cited)")
        return recs
    url, kind = URLS.get(name, (None, None))
    if not url:
        for p in PARAMS:
            rec(p, row.get(p), None, "", "NO PRIMARY SOURCE", "No vendor URL mapped", "none")
        return recs
    raw, code = fetch(url)
    if raw is None:
        for p in PARAMS:
            rec(p, row.get(p), None, url, "SOURCE UNAVAILABLE", f"HTTP {code}", STYPE[kind])
        return recs
    text = to_text(raw, url)
    c = extract(text, kind, name)
    notes = str(row.get("Notes / power source") or "")
    for p in PARAMS:
        sheet = row.get(p)
        if p == "Memory":
            s = str(sheet)
            fam = re.search(r"LPDDR5X|LPDDR5|LPDDR4X|LPDDR4|DDR5|DDR4", s)
            spd = re.search(r"(\d{4})", s)
            hits = [n for n, _ in c["Memory"] if fam and n.upper().startswith(fam.group(0))]
            spd_hit = [q for q in hits if spd and spd.group(1) in q]
            if spd_hit:
                rec(p, sheet, spd_hit[0], url, "VERIFIED", f"page: '{spd_hit[0]}' (type + speed)", STYPE[kind])
            elif hits:
                rec(p, sheet, hits[0], url, "VERIFIED" if not spd else "PARTIAL", f"page states memory type '{hits[0]}'" + ("; speed grade in sheet not stated on page" if spd else ""), STYPE[kind])
            else:
                rec(p, sheet, None, url, "NOT ON PAGE", "memory type not found on page", STYPE[kind])
            continue
        if p == "Memory BW (GB/s)" and kind in ("amd", "ark") and not c[p]:
            rec(p, sheet, None, url, "DERIVED/ESTIMATE", "Vendor page gives memory type/speed and channels only; bandwidth = MT/s x 8 B x channels (derived)", STYPE[kind])
            continue
        st, best = status_num(sheet, c[p])
        if p in ("Base / default power (W)", "Max turbo / max power (W)") and kind in ("ark", "amd"):
            key = r"Turbo|cTDP|Configurable" if p.startswith("Max") else r"Base Power|Default TDP|\bTDP\b"
            sub = [x for x in c[p] if re.search(key, x[1])]
            if sub:
                st, best = status_num(sheet, sub)
                if p.startswith("Max") and kind == "ark" and st != "VERIFIED":
                    turbo = [x for x in sub if "Turbo" in x[1] and "(" in x[1]]
                    if turbo:
                        best = turbo[-1]
                if p.startswith("Max") and kind == "amd" and st != "VERIFIED":
                    best = max(sub, key=lambda x: x[0])
        note = f"'{best[1]}'" if best else "value not found on page"
        if best and re.search(r"reference board|footnote", best[1], re.I) and st == "MISMATCH":
            st, best, note = "NOT ON PAGE", None, "vendor page does not state this value (only marketing footnotes mention memory)"
        if p == "Max turbo / max power (W)" and kind == "ark" and st == "MISMATCH" and not re.search(r"Maximum Turbo Power\b.{0,600}?\(\s*\d", text):
            st = "DERIVED/ESTIMATE"; note = f"ARK lists only TDP {best[0]:g} W for this SKU (no Maximum Turbo Power); sheet value is the board-vendor PL2 commonly used"
        if p == "Memory BW (GB/s)" and kind == "ark" and st == "MISMATCH" and "DDR4" in str(row.get("Memory")) and best:
            st = "DERIVED/ESTIMATE"; note = f"ARK max bandwidth {best[0]:g} GB/s is for the fastest supported DDR5; sheet assumes {row.get('Memory')} -> derived"
        if p == "Max turbo / max power (W)" and kind == "amd" and st != "VERIFIED" and re.search(r"PPT", notes):
            st = "DERIVED/ESTIMATE"; note = "AMD page lists TDP/cTDP only; PPT (socket power limit) = 1.35 x TDP per AMD platform rules: " + notes[:80]
        if p == "Max RAM (GB)" and st == "MISMATCH" and re.search(r"\(\d+GB\)", name):
            st = "DERIVED/ESTIMATE"; note = f"Row models a fixed {re.search(r'\((\d+GB)\)', name).group(1)} configuration; vendor max is {best[0]:g} GB ('{best[1][:60]}')"
        if p == "Max turbo / max power (W)" and st != "VERIFIED" and re.search(r"ESTIMATE|PSU max|system/PSU|power-mode", notes, re.I):
            st = "DERIVED/ESTIMATE"; note = "Sheet marks this as an estimate / system figure: " + notes[:100]
        if p == "Base / default power (W)" and st != "VERIFIED" and re.search(r"system/PSU|vendor publishes no TDP", notes, re.I):
            st = "DERIVED/ESTIMATE"; note = "Vendor publishes no TDP; sheet uses a typical system power estimate: " + notes[:100]
        if p == "CPU PCIe lanes" and kind == "apple":
            st = "NOT APPLICABLE"; note = "SoC with no exposed CPU PCIe lanes for a dGPU (Thunderbolt only)"
        rec(p, sheet, best[0] if best else None, url, st, note, STYPE[kind])
    return recs

if __name__ == "__main__":
    rows = load_rows("cpus")
    if len(sys.argv) > 1:
        rows = [r for r in rows if sys.argv[1].lower() in str(r["CPU / platform"]).lower()]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(check, rows))
    flat = [r for rs in res for r in rs]
    write_evidence(flat, "cpus")
    from collections import Counter
    print(Counter(r["Status"] for r in flat))
    for r in flat:
        if r["Status"] not in ("VERIFIED", "INTERNAL SOURCE"):
            print(f"{r['Status'][:6]} | {r['Item']} | {r['Parameter']} | sheet={r['Sheet value']} | src={r['Source value']} | {r['Note'][:100]}")
