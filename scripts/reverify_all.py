"""Re-verify the whole project in one run: data, cross-references, page wiring, themes and every formula.

  python scripts/reverify_all.py              # offline checks (no network)
  python scripts/reverify_all.py --online     # also re-check model repos, source links and newer releases
  python scripts/reverify_all.py --no-write   # do not update docs/verification.json

Groups
  Data tables        required columns, unique keys, numeric sanity, evidence status, public-only content
  Cross-references   evidence <-> tables, model arch <-> arch support, runtime names, defaults named in the code
  Published build    docs/data.json and docs/parity.json match the CSV tables; private files are not tracked
  Page wiring        HTML ids <-> JavaScript, pages, labels, assets
  Themes             every colour token exists in light and dark, no hard-coded colours, text contrast
  Formulas           hand-computed cases and cached workbook values (v28 sheets)
  Python = browser   scripts/formulas.py against docs/formulas.js over the full catalog (needs Node.js)
  Online             Hugging Face repo metadata, source-link reachability, newer releases to review

Exit code 1 when any check fails. Items "to review" never fail the run. The summary is written to
docs/verification.json (shown on the site) and the full detail to out/reverify_report.json.
"""
import argparse
import collections
import concurrent.futures
import csv
import datetime
import json
import math
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

import formulas as F
from common import DATA, DERIVED_BASIS, OUT, ROOT, UA, derive_compute, load_rows

DOCS = ROOT / "docs"
SCRIPTS = ROOT / "scripts"
JS_FILES = ("formulas.js", "theme.js", "pipeline.js", "app.js", "selftest.js")
GROUPS = ("Data tables", "Cross-references", "Published build", "Page wiring", "Themes", "Formulas", "Python = browser", "Online")
SHEET_TABLES = {"Models": ("models", "Model"), "GPUs": ("gpus", "Accelerator"), "CPUs": ("cpus", "CPU / platform"),
                "Voice Models": ("voice_models", "Model"), "Quant": ("quant", "Quantization"), "Reference Models": ("reference_models", "Model"),
                "Compute": ("compute", "Item")}
REQUIRED = {
    "models": ["Model", "Vendor", "Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim", "Max context", "Type", "License",
               "HF repo", "Year", "Runtime arch", "Source URL", "Evidence status"],
    "gpus": ["Accelerator", "Vendor", "Memory (GB)", "Bandwidth (GB/s)", "TDP / board power (W)", "Interface", "Architecture",
             "CUDA compute capability", "ROCm support", "Runtime HW class", "Source URL", "Evidence status"],
    "cpus": ["CPU / platform", "Class", "Cores", "Memory BW (GB/s)", "Max RAM (GB)", "CPU PCIe lanes", "Runtime HW class", "Source URL", "Evidence status"],
    "quant": ["Quantization", "Bits/weight (eff.)"],
    "voice_models": ["Stage", "Model", "Total params (B)", "Bits / weight"],
    "libraries": ["Library", *F.HW_CLASSES, "Typical eff.", "Vulkan fallback eff.", "Min CUDA compute capability"],
    "arch_support": ["Arch key"],
    "task_rules": ["Task", "Base min bits (4-30B)", "Small model <4B min bits", "Large model >=30B min bits", "VLM required?"],
    "compute": ["Sheet", "Item", "TFLOPS (est.)", "Basis", "Evidence status"],
    "evidence": ["Sheet", "Item", "Parameter", "Status", "Source URL"],
}
PRIVATE_MARKERS = re.compile(r"Intel Confidential|Intel Top Secret|sharepoint\.com|\.ith\.intel\.com|hsdes\.intel\.com|speed\.intel\.com|goto\.intel\.com", re.I)
PLANNING = re.compile(r"Pre-release planning|PLANNING / PUBLIC SOURCE PENDING", re.I)
QUANT_REPO = re.compile(r"gguf|awq|gptq|bnb|[-_.]fp8|[-_.]int[48]|[-_.]\d+bit|mlx|onnx|openvino|exl[23]|quantized|[-_.]lora|adapter", re.I)


class Report:
    def __init__(self):
        self.rows = []

    def check(self, group, name, ok, detail=""):
        self.rows.append({"group": group, "name": name, "status": "pass" if ok else "fail", "detail": "" if ok else str(detail)})
        return ok

    def review(self, group, name, detail):
        self.rows.append({"group": group, "name": name, "status": "review", "detail": str(detail)})

    def tally(self, group=None):
        rows = [row for row in self.rows if group is None or row["group"] == group]
        count = collections.Counter(row["status"] for row in rows)
        return {"pass": count["pass"], "fail": count["fail"], "warn": count["review"]}


def read_csv(name):
    with open(DATA / f"{name}.csv", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sample(items, limit=6):
    items = list(items)
    return ", ".join(str(item) for item in items[:limit]) + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def close(a, b, rel=1e-9):
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


# ---------------------------------------------------------------- data tables
def check_data(report, tables):
    group = "Data tables"
    for name, columns in REQUIRED.items():
        rows = tables[name]
        missing = [column for column in columns if rows and column not in rows[0]]
        report.check(group, f"{name}.csv has its required columns", rows and not missing, f"missing {missing}" if rows else "table is empty")
    for name, key in (("models", "Model"), ("gpus", "Accelerator"), ("cpus", "CPU / platform"), ("quant", "Quantization"),
                      ("libraries", "Library"), ("arch_support", "Arch key"), ("task_rules", "Task")):
        counts = collections.Counter(row[key] for row in tables[name])
        duplicates = [value for value, count in counts.items() if count > 1 or not value]
        report.check(group, f"{name}.csv keys are unique and non-empty", not duplicates, sample(duplicates))
    voice_keys = collections.Counter((row["Stage"], row["Model"]) for row in tables["voice_models"])
    report.check(group, "voice_models.csv stage + model pairs are unique", all(count == 1 for count in voice_keys.values()), sample(k for k, c in voice_keys.items() if c > 1))

    models = load_rows("models")
    unsized = [m["Model"] for m in models if not F.has_sizing_specs(m)]
    report.check(group, "every model has parameters, layers, KV heads and head dimension", not unsized, sample(unsized))
    bad_active = [m["Model"] for m in models if F.has_sizing_specs(m) and m["Active params (B)"] > m["Total params (B)"] * 1.0001]
    report.check(group, "active parameters never exceed total parameters", not bad_active, sample(bad_active))
    no_context = [m["Model"] for m in models if not F.positive(m.get("Max context"))]
    report.check(group, "every model has a published context limit", not no_context, sample(no_context))
    bad_year = [m["Model"] for m in models if not (2017 <= F.positive(m.get("Year")) <= datetime.date.today().year + 1)]
    report.check(group, "model release years are plausible", not bad_year, sample(bad_year))

    gpus, cpus = load_rows("gpus"), load_rows("cpus")
    bad_gpu = [g["Accelerator"] for g in gpus if not (F.positive(g.get("Memory (GB)")) and F.positive(g.get("Bandwidth (GB/s)")) and F.positive(g.get("TDP / board power (W)")))]
    report.check(group, "every GPU has memory, bandwidth and power", not bad_gpu, sample(bad_gpu))
    bad_cpu = [c["CPU / platform"] for c in cpus if not (F.positive(c.get("Memory BW (GB/s)")) and F.positive(c.get("Max RAM (GB)")))]
    report.check(group, "every CPU has memory bandwidth and a RAM limit", not bad_cpu, sample(bad_cpu))
    bits = [F.positive(q["Bits/weight (eff.)"]) for q in load_rows("quant")]
    report.check(group, "quantization bits are between 1 and 16", all(1 <= b <= 16 for b in bits), bits)
    cells = [(lib["Library"], hw, lib[hw]) for lib in tables["libraries"] for hw in F.HW_CLASSES]
    bad_cells = [f"{lib}/{hw}={cell!r}" for lib, hw, cell in cells if not (0 <= F.parse_efficiency(cell) <= 1) or (F.parse_efficiency(cell) == 0 and not str(cell).strip().startswith(("—", "-")))]
    report.check(group, "runtime efficiency cells are a number from 0 to 1 or a dash", not bad_cells, sample(bad_cells))
    arch_values = {value for row in tables["arch_support"] for key, value in row.items() if key not in ("Arch key", "Models (examples)", "Notes")}
    report.check(group, "arch support cells are Y, P or N", arch_values <= {"Y", "P", "N"}, sorted(arch_values))

    compute = tables["compute"]
    compute_keys = collections.Counter((row["Sheet"], row["Item"]) for row in compute)
    report.check(group, "compute.csv sheet + item pairs are unique", all(count == 1 for count in compute_keys.values()), sample(k for k, c in compute_keys.items() if c > 1))
    bad_compute = [row["Item"] for row in compute if not (0 < F.positive(row["TFLOPS (est.)"]) < 100000) or row["Sheet"] not in ("GPUs", "CPUs")]
    report.check(group, "compute estimates are positive numbers for GPUs or CPUs", not bad_compute, sample(bad_compute))
    unlabelled = [row["Item"] for row in compute if row["Evidence status"] != "DERIVED/ESTIMATE"]
    report.check(group, "compute estimates are all labelled as estimates", not unlabelled, sample(unlabelled))
    workbook = {(row["Sheet"], row["Item"]): F.positive(row["TFLOPS (est.)"]) for row in compute if not row["Basis"].startswith(DERIVED_BASIS)}
    planning = {(sheet, row[key]) for sheet, name, key in (("GPUs", "gpus", "Accelerator"), ("CPUs", "cpus", "CPU / platform"))
                for row in tables[name] if PLANNING.search(row.get("Evidence status", ""))}
    leaked = sorted(item for sheet, item in set(workbook) & planning)
    report.check(group, "no workbook compute estimate is published for pre-release planning hardware", not leaked, sample(leaked))
    expected = {(row["Sheet"], row["Item"]): (row["TFLOPS (est.)"], row["Basis"]) for row in derive_compute({"GPUs": gpus, "CPUs": cpus}, workbook)}
    derived = {(row["Sheet"], row["Item"]): (row["TFLOPS (est.)"], row["Basis"]) for row in compute if row["Basis"].startswith(DERIVED_BASIS)}
    stale = sorted(item for sheet, item in set(expected) | set(derived) if expected.get((sheet, item)) != derived.get((sheet, item)))
    report.check(group, "derived compute figures equal the published-column formula (run export_compute_estimates.py --derive)", not stale, sample(stale))

    for name, key in (("models", "Model"), ("gpus", "Accelerator"), ("cpus", "CPU / platform")):
        rows = tables[name]
        no_status = [row[key] for row in rows if not row.get("Evidence status", "").strip()]
        report.check(group, f"{name}.csv rows all carry an evidence status", not no_status, sample(no_status))
        bad_url = [row[key] for row in rows if any(not re.match(r"https?://", part.strip()) for part in row.get("Source URL", "").split(";") if part.strip())]
        report.check(group, f"{name}.csv source links are http(s) URLs", not bad_url, sample(bad_url))
        unsourced = [row[key] for row in rows if not row.get("Source URL", "").strip() and not PLANNING.search(row.get("Evidence status", ""))]
        if unsourced:
            report.review(group, f"{name}.csv rows without a source link", sample(unsourced, 10))
    pending = [row["Item"] for row in tables["evidence"] if row["Status"] in ("REVIEW", "SOURCE UNAVAILABLE", "PARTIAL")]
    if pending:
        report.review(group, "evidence records still marked review / source unavailable / partial", f"{len(pending)} records, e.g. {sample(sorted(set(pending)), 5)}")

    leaks = []
    for path in [*DATA.glob("*.csv"), *(DOCS / name for name in ("index.html", "data.json", *JS_FILES))]:
        if path.exists() and PRIVATE_MARKERS.search(path.read_text(encoding="utf-8", errors="replace")):
            leaks.append(path.name)
    report.check(group, "no confidential markers or internal hosts in published files", not leaks, sample(leaks))
    statuses = collections.Counter(row["Status"] for row in tables["evidence"])
    return {"models": len(models), "gpus": len(gpus), "cpus": len(cpus), "evidence": len(tables["evidence"]), "evidence_verified": statuses["VERIFIED"]}


# ----------------------------------------------------------- cross-references
def js_text(name):
    path = DOCS / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def check_references(report, tables):
    group = "Cross-references"
    evidence = tables["evidence"]
    for sheet, (table, key) in SHEET_TABLES.items():
        names = {row[key] for row in tables[table]}
        items = {row["Item"] for row in evidence if row["Sheet"] == sheet}
        report.check(group, f"evidence for {sheet} points at existing rows", items <= names, sample(sorted(items - names)))
        if sheet in ("Models", "GPUs", "CPUs", "Compute"):
            report.check(group, f"every {sheet} row has at least one evidence record", names <= items, sample(sorted(names - items)))
    for sheet, name, key in (("GPUs", "gpus", "Accelerator"), ("CPUs", "cpus", "CPU / platform")):
        known = {row[key] for row in tables[name]}
        orphans = sorted(row["Item"] for row in tables["compute"] if row["Sheet"] == sheet and row["Item"] not in known)
        report.check(group, f"compute estimates for {sheet} point at existing rows", not orphans, sample(orphans))
        uncovered = sorted(known - {row["Item"] for row in tables["compute"] if row["Sheet"] == sheet})
        report.check(group, f"every {sheet} row has a compute estimate for time to first token", not uncovered, sample(uncovered))
    unknown_sheets = {row["Sheet"] for row in evidence} - set(SHEET_TABLES) - {"NVMe Storage"}
    report.check(group, "evidence sheets are all known tables", not unknown_sheets, sorted(unknown_sheets))

    libraries = [row["Library"] for row in tables["libraries"]]
    arch_columns = [c for c in tables["arch_support"][0] if c not in ("Arch key", "Models (examples)", "Notes")]
    report.check(group, "arch support has one column per runtime library", arch_columns == libraries, f"arch {arch_columns} vs libraries {libraries}")
    arches = {row["Arch key"] for row in tables["arch_support"]}
    missing_arch = sorted({row["Runtime arch"] for row in tables["models"]} - arches)
    report.check(group, "every model architecture has an arch support row", not missing_arch, sample(missing_arch))
    for table, key in (("gpus", "Accelerator"), ("cpus", "CPU / platform")):
        bad = [row[key] for row in tables[table] if row["Runtime HW class"] not in F.HW_CLASSES]
        report.check(group, f"{table}.csv runtime class matches a runtime table column", not bad, sample(bad))
    report.check(group, "CPU-offload runtimes exist in the runtime table", set(F.OFFLOAD_RUNTIMES) <= set(libraries), F.OFFLOAD_RUNTIMES)
    stages = {row["Stage"] for row in tables["voice_models"]}
    report.check(group, "voice catalog covers VAD, Diarization and ASR", {"VAD", "Diarization", "ASR"} <= stages, sorted(stages))
    report.check(group, "quantization table contains Q4_K_M", any("Q4_K_M" in row["Quantization"] for row in tables["quant"]))
    report.check(group, "task rules contain the default task", any(row["Task"] == F.DEFAULT_TASK for row in tables["task_rules"]))

    app, pipeline, html = js_text("app.js"), js_text("pipeline.js"), js_text("index.html")
    model_names = {row["Model"] for row in tables["models"]}
    gpu_names = {row["Accelerator"] for row in tables["gpus"]}
    cpu_names = {row["CPU / platform"] for row in tables["cpus"]}
    quant_names = {row["Quantization"] for row in tables["quant"]}
    defaults = [("default model", re.search(r'row\.Model === "([^"]+)"\) \|\| state\.data\.models\[0\]', app), model_names),
                ("default GPU", re.search(r'row\.Accelerator === "([^"]+)"\) \|\| state\.data\.gpus\[0\]', app), gpu_names),
                ("default CPU", re.search(r'row\["CPU / platform"\] === "([^"]+)"\) \|\| state\.data\.cpus\[0\]', app), cpu_names),
                ("pipeline default GPU", re.search(r'const DEFAULT_GPU = "([^"]+)"', pipeline), gpu_names),
                ("pipeline default quantization", re.search(r'const DEFAULT_QUANT = "([^"]+)"', pipeline), quant_names)]
    for label, match, names in defaults:
        report.check(group, f"{label} named in the code exists in the data", bool(match) and match.group(1) in names, match.group(1) if match else "not found in source")
    llm_default = re.search(r'const DEFAULT_LLM = "([^"]+)"', pipeline)
    report.check(group, "Multi-LLM default LLM exists in the model table", bool(llm_default) and llm_default.group(1) in model_names, llm_default.group(1) if llm_default else "not found in source")
    voice_defaults = re.search(r"const DEFAULT_VOICE = \[(.+?)\];", pipeline)
    pairs = re.findall(r'\["([^"]+)", "([^"]+)"\]', voice_defaults.group(1)) if voice_defaults else []
    voice_pairs = {(row["Stage"], row["Model"]) for row in tables["voice_models"]}
    report.check(group, "pipeline default voice models exist in the voice table", bool(pairs) and set(pairs) <= voice_pairs, sample(sorted(set(pairs) - voice_pairs)) or "not found in source")
    generation = re.search(r"function cpuGeneration\(name\) \{(.+?)\n\}", app, re.S)
    stale = sorted(set(re.findall(r'^\s+"([^"]+)": "', generation.group(1), re.M)) - cpu_names) if generation else []
    if stale:
        report.review(group, "CPU generation labels in app.js for CPUs no longer in the table", sample(stale))
    table_options = set(re.findall(r'<option value="([A-Za-z]+)">', re.search(r'<select id="database-select">(.+?)</select>', html, re.S).group(1)))
    payload_keys = {"models", "gpus", "cpus", "quants", "voiceModels", "libraries", "archSupport", "taskRules", "compute", "referenceModels"}
    report.check(group, "database page lists exactly the published tables", table_options == payload_keys, f"page {sorted(table_options)} vs data {sorted(payload_keys)}")


# ------------------------------------------------------------ published build
def check_build(report):
    group = "Published build"
    from build_web_data import build_payload
    from selector_parity import calculate
    fresh = json.loads(json.dumps(build_payload(), ensure_ascii=False))
    published = json.loads((DOCS / "data.json").read_text(encoding="utf-8"))
    stale = sorted(key for key in fresh if fresh[key] != published.get(key))
    report.check(group, "docs/data.json matches the CSV tables", not stale and set(fresh) == set(published), f"stale sections: {stale}. Run python scripts/build_web_data.py")
    parity = json.loads((DOCS / "parity.json").read_text(encoding="utf-8"))
    report.check(group, "docs/parity.json matches the current formulas", parity == json.loads(json.dumps(calculate())), "Run python scripts/build_web_data.py")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    report.check(group, ".gitignore excludes private reference data, workbooks and scratch output",
                 all(line in ignore for line in ("docs/private_reference.json", "docs/private_compute.json", "*.xlsx", "out/")))
    if shutil.which("git"):
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.splitlines()
        private = [path for path in tracked if path.endswith((".xlsx", ".xlsm")) or path.startswith("out/") or path in ("docs/private_reference.json", "docs/private_compute.json")]
        report.check(group, "no private or scratch file is tracked by git", not private, sample(private))
    else:
        report.review(group, "git not found", "could not confirm that private files are untracked")


# ---------------------------------------------------------------- page wiring
def check_wiring(report):
    group = "Page wiring"
    html = js_text("index.html")
    ids = re.findall(r'\bid="([^"]+)"', html)
    duplicates = sorted(value for value, count in collections.Counter(ids).items() if count > 1)
    report.check(group, "HTML ids are unique", not duplicates, sample(duplicates))
    id_set = set(ids)
    scripts = {name: js_text(name) for name in JS_FILES}
    missing_files = [name for name, text in scripts.items() if not text]
    report.check(group, "all page scripts exist", not missing_files, sample(missing_files))
    all_js = "\n".join(scripts.values())
    referenced = set(re.findall(r'(?:byId|getElementById)\("([^"]+)"\)', all_js))
    report.check(group, "every element the scripts look up exists in the page", referenced <= id_set, sample(sorted(referenced - id_set)))
    literals = set(re.findall(r'"([A-Za-z][\w-]*)"', all_js))
    controls = set(re.findall(r'<(?:select|input|button|output|table|dialog)\b[^>]*\bid="([^"]+)"', html))
    orphan = sorted(controls - literals)
    report.check(group, "every control with an id is used by a script", not orphan, sample(orphan))
    for attribute in ("aria-labelledby", "for"):
        targets = set(re.findall(rf'\b{attribute}="([^"]+)"', html))
        report.check(group, f"every {attribute} target exists", targets <= id_set, sample(sorted(targets - id_set)))
    pages = re.findall(r'class="nav-page[^"]*" data-page="([^"]+)"', html)
    report.check(group, "every navigation button has a page section", all(f"{page}-page" in id_set for page in pages), sample(p for p in pages if f"{p}-page" not in id_set))
    handled = set(re.findall(r'if \(page === "([^"]+)"\)', scripts["app.js"])) | {"model"}
    report.check(group, "every page is rendered when opened", set(pages) <= handled, sample(sorted(set(pages) - handled)))
    tabs = set(re.findall(r'data-knowledge-page="([^"]+)"', html))
    panels = set(re.findall(r'data-knowledge-page-content="([^"]+)"', html))
    report.check(group, "every knowledge tab has a panel", tabs == panels, f"tabs {sorted(tabs)} vs panels {sorted(panels)}")
    guide = re.search(r'data-knowledge-page-content="ttft".*?</section>', html, re.S)
    constant = lambda name: (re.search(rf"const {name} = ([\d.]+);", scripts["formulas.js"]) or [None, "missing"])[1]
    wanted = [f"{constant('PREFILL_GFLOPS_PER_B')} × active parameters", f"{constant('PREFILL_MFU')} × TFLOPS × 1000", f"× {constant('CPU_PREFILL_FACTOR')}", f"{constant('EXTRA_CARD_SCALE')} × (cards", "prompt tokens ÷ prefill tok/s + 1 ÷ decode tok/s"]
    absent = [text for text in wanted if not guide or text not in guide.group(0)]
    report.check(group, "Know more shows the first-token formula with the constants the scripts use", not absent, sample(absent))
    opened = set(re.findall(r'data-open-page="([^"]+)"', html))
    report.check(group, "in-page links open existing pages", opened <= set(pages), sample(sorted(opened - set(pages))))
    assets = re.findall(r'(?:src|href)="((?!https?:|#|\./)[^"]+)"', html)
    missing = [asset for asset in assets if not (DOCS / asset.split("?")[0]).exists()]
    report.check(group, "local scripts and stylesheets exist", not missing, sample(missing))
    versions = set(re.findall(r'\?v=([\w.-]+)"', html))
    report.check(group, "local assets share one cache key", len(versions) == 1, sorted(versions))
    tags = collections.Counter(re.findall(r"<(/?)(section|details|dialog|table|main|header|aside|nav)\b", html))
    unbalanced = [tag for tag in ("section", "details", "dialog", "table", "main", "header", "aside", "nav") if tags[("", tag)] != tags[("/", tag)]]
    report.check(group, "structural tags are balanced", not unbalanced, sample(unbalanced))
    report.check(group, "page sets its theme before first paint", "dataset.theme" in html.split("</head>")[0] and 'id="theme-toggle"' in html)
    for name in ("runtime-select", "hardware-runtime-select", "task-select", "hardware-task-select"):
        report.check(group, f"{name} is filled from the data tables", re.search(rf'"{name}"', all_js) is not None)
    stage_fields = re.findall(r'data-field=\\?"(runtime|kvBits)\\?"', scripts["pipeline.js"])
    report.check(group, "Multi-LLM runtime and KV cache are set inside the stages", {"runtime", "kvBits"} <= set(stage_fields) and "ctx.runtimes()" in scripts["pipeline.js"]
                 and not {"pipe-runtime-select", "pipe-kv-select"} & id_set, sorted(set(stage_fields)))


# --------------------------------------------------------------------- themes
def parse_color(value):
    value = value.strip()
    match = re.fullmatch(r"#([0-9a-fA-F]{6})", value)
    if match:
        return tuple(int(match.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    match = re.fullmatch(r"#([0-9a-fA-F]{3})", value)
    return tuple(int(ch * 2, 16) for ch in match.group(1)) if match else None


def luminance(rgb):
    channel = lambda c: (c / 255 / 12.92) if c / 255 <= 0.03928 else ((c / 255 + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def check_themes(report):
    group = "Themes"
    css = js_text("styles.css")
    blocks = re.findall(r':root(\[data-theme="dark"\])?\s*\{([^}]*)\}', css)
    themes = {"dark" if marker else "light": dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body)) for marker, body in blocks}
    report.check(group, "stylesheet defines a light and a dark palette", set(themes) == {"light", "dark"}, sorted(themes))
    light, dark = themes.get("light", {}), themes.get("dark", {})
    fonts = {"--mono", "--sans"}
    report.check(group, "dark palette overrides every colour token", set(light) - fonts == set(dark), sample(sorted((set(light) - fonts) ^ set(dark))))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    report.check(group, "every token used in the stylesheet is defined", used <= set(light), sample(sorted(used - set(light))))
    unused = sorted(set(light) - used)
    if unused:
        report.review(group, "tokens defined but not used", sample(unused))
    body = re.sub(r':root(\[data-theme="dark"\])?\s*\{[^}]*\}', "", css)
    hard = sorted(set(re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\([^)]*\)|\b(?:white|black)\b(?!-)", body)))
    report.check(group, "no hard-coded colours outside the token palettes", not hard, sample(hard))
    report.check(group, "both palettes declare color-scheme", "color-scheme: light" in css and "color-scheme: dark" in css)
    report.check(group, "reduced-motion preference is respected", "prefers-reduced-motion" in css)
    report.check(group, "keyboard focus stays visible", ":focus-visible" in css)
    pairs = [("--ink", "--paper"), ("--ink", "--surface"), ("--muted", "--paper"), ("--muted", "--surface"), ("--panel-ink", "--panel"),
             ("--panel-muted", "--panel"), ("--lime", "--panel"), ("--on-accent", "--lime"), ("--teal-ink", "--teal-soft"),
             ("--amber-ink", "--amber-soft"), ("--red-ink", "--red-soft"), ("--red-ink", "--amber-soft"), ("--red-ink", "--surface"),
             ("--ink", "--lime-soft"), ("--ink", "--amber-soft"), ("--coral-ink", "--paper"), ("--amber-ink", "--paper"), ("--coral", "--panel"),
             ("--blue", "--surface"), ("--teal", "--surface"), ("--teal", "--paper"), ("--paper", "--ink")]
    for theme, tokens in themes.items():
        for front, back in pairs:
            a, b = parse_color(tokens.get(front, "")), parse_color(tokens.get(back, ""))
            if not a or not b:
                report.check(group, f"{theme}: {front} on {back} are plain hex colours", False, f"{tokens.get(front)} / {tokens.get(back)}")
                continue
            ratio = contrast(a, b)
            name = f"{theme}: {front[2:]} on {back[2:]} is readable"
            if ratio < 3:
                report.check(group, name, False, f"contrast {ratio:.2f}:1 is below 3:1")
            elif ratio < 4.5:
                report.check(group, name, True)
                report.review(group, f"{theme}: {front[2:]} on {back[2:]} contrast {ratio:.2f}:1", "meets 3:1 (large text, icons) but not 4.5:1 for small text")
            else:
                report.check(group, name, True)


# ------------------------------------------------------------------- formulas
def workbook_pipeline_case():
    """Cached values of the workbook's Multi-Stage Pipeline Fit sheet (v28, 2026-11-20)."""
    model = lambda name, total, active, layers, heads, dim: {"Model": name, "Total params (B)": total, "Active params (B)": active, "Layers": layers,
                                                              "KV heads": heads, "Head dim": dim, "Max context": 0}
    slot = lambda m: {"model": m, "bits": 4, "context": 8192, "target": 40}
    args = {"memoryGb": 160, "bandwidthGbps": 1712, "quantity": 1, "headroomPct": 10, "utilization": 0.5, "kvBits": 16, "activity": "simultaneous",
            "voice": [{"stage": "VAD", "name": "Silero VAD", "placement": "GPU", "weightsGb": 0.002, "extraGb": 5, "bwGbps": 5},
                      {"stage": "Diarization", "name": "pyannote community-1", "placement": "GPU", "weightsGb": None, "extraGb": 5, "bwGbps": 5},
                      {"stage": "ASR", "name": "Qwen3-ASR-1.7B", "placement": "GPU", "weightsGb": 4.698434816, "extraGb": 5, "bwGbps": 5}],
            "llms": [slot(model("LLM 1 (4B)", 4, 4, 36, 8, 128)), slot(model("LLM 2 (9.65B)", 9.65, 9, 32, 2, 128)),
                     slot(model("LLM 3 (12B)", 12, 12, 48, 8, 256)), slot(model("LLM 4 (32.8B)", 32.8, 32.8, 64, 8, 128))]}
    expected = {"usableGb": 144, "effectiveBwGbps": 856, "voiceMemoryGb": 19.700434816, "voiceBwGbps": 15, "llmsFit": 3,
                "weights": [2.06, 4.96975, 6.18, 16.892], "kvOverhead": [1.352159552, 0.616317956, 3.653825472, 3.329923648],
                "total": [3.412159552, 5.586067956, 9.833825472, 20.221923648], "bwNeeded": [82.4, 185.4, 247.2, 675.68],
                "cumulativeMemory": [23.112594368, 28.698662324, 38.532487796, 58.754411444], "cumulativeBw": [97.4, 282.8, 530, 1205.68],
                "status": ["FITS", "FITS", "FITS", "NO - BANDWIDTH"], "tokAlone0": 408.252427184466}
    return args, expected


def run_pipeline(args):
    return F.pipeline_fit(args["memoryGb"], args["bandwidthGbps"], args["utilization"], args["voice"], args["llms"], args["quantity"],
                          args["headroomPct"], args["kvBits"], args["activity"], args.get("tflops", 0), args.get("promptTokens", 0), args.get("mfu"))


def check_formulas(report, tables):
    group = "Formulas"
    near = lambda name, got, want, rel=1e-9: report.check(group, name, close(got, want, rel), f"got {got!r}, expected {want!r}")

    qwen = {"Total params (B)": 32.8, "Active params (B)": 32.8, "Layers": 64, "KV heads": 8, "Head dim": 128, "Max context": 40960, "Type": "Dense"}
    need = F.memory_need(qwen, 4.85, 8192, 1, 8)
    near("weights = total x bits / 8 x 1.03", need["weights"], 32.8 * 4.85 / 8 * 1.03)
    near("KV cache = 2 x L x H x D x kvBits / 8 x context x users", need["kv"], 1.073741824)
    near("runtime = 7% of weights", need["runtime"], need["weights"] * 0.07)
    near("memory need is the sum of its parts", need["total"], 20.48155 + 1.073741824 + 1.4337085)
    near("KV cache doubles from 8-bit to 16-bit", F.memory_need(qwen, 4.85, 8192, 1, 16)["kv"], 2 * need["kv"])
    near("KV cache scales with users", F.memory_need(qwen, 4.85, 8192, 4, 8)["kv"], 4 * need["kv"])
    near("decode = eff x bandwidth / (active x bits / 8)", F.decode_tok_per_second(32.8, 4.85, 0.5, 456), 0.5 * 456 / (32.8 * 4.85 / 8))
    near("each extra card adds 80% bandwidth", F.decode_tok_per_second(32.8, 4.85, 0.5, 456, 1, 2), 1.8 * 0.5 * 456 / (32.8 * 4.85 / 8))
    near("decode divides by concurrent users", F.decode_tok_per_second(32.8, 4.85, 0.5, 456, 4), 0.5 * 456 / (32.8 * 4.85 / 8) / 4)
    near("unsupported runtime gives zero speed", F.decode_tok_per_second(32.8, 4.85, 0, 456), 0)
    near("GPU usable = memory x 0.92 x cards", F.gpu_usable_gb(24, 2), 24 * 0.92 * 2)
    near("GPU headroom is adjustable", F.gpu_usable_gb(24, 1, 10), 21.6)
    near("CPU usable reserves 8 GB on large hosts", F.cpu_usable_gb(128), 120)
    near("CPU usable reserves 25% on small hosts", F.cpu_usable_gb(16), 12)
    near("split decode runs GPU and CPU shares in series", F.split_decode_tok_per_second(2, 0.5, 100, 20), 1 / (2 * (0.5 / 100 + 0.5 / 20)))
    near("split decode with everything on GPU equals GPU speed", F.split_decode_tok_per_second(2, 1, 100, 0), 50)
    near("split decode needs a working CPU runtime for the overflow", F.split_decode_tok_per_second(2, 0.5, 100, 0), 0)
    report.check(group, "CPU offload only for runtimes that can split a model", F.offload_supported("llama.cpp") and not F.offload_supported("vLLM"))
    report.check(group, "unified-memory systems are detected from the interface", F.is_unified_memory({"Interface": "Unified memory"}) and not F.is_unified_memory({"Interface": "PCIe 5.0 x16"}))

    rules = F.build_task_rules(load_rows("task_rules"))
    coding = F.task_rule(rules, "Coding")
    sized = lambda total, kind="Dense": {"Total params (B)": total, "Type": kind}
    report.check(group, "task floor: small models (<4B) need more bits", F.required_bits(sized(3), coding) == 8.5)
    report.check(group, "task floor: mid models (4-30B) use the base bits", F.required_bits(sized(20), coding) == 5.7)
    report.check(group, "task floor: large models (>=30B) may use Q4", F.required_bits(sized(30), coding) == 4.85)
    vision = F.task_rule(rules, "Vision (image + text)")
    report.check(group, "vision task requires a vision-language model", not F.task_allows(sized(12), vision, 8.5) and F.task_allows(sized(12, "Dense (VLM)"), vision, 8.5))
    report.check(group, "unknown task falls back to general chat", F.task_rule(rules, "no such task")["task"] == F.DEFAULT_TASK)

    table = F.build_efficiency_table(load_rows("libraries"))
    intel = {"Vendor": "Intel", "Runtime HW class": "Intel GPU / iGPU", "CUDA compute capability": "-", "ROCm support": "-"}
    old_nvidia = {"Vendor": "NVIDIA", "Runtime HW class": "NVIDIA GPU", "CUDA compute capability": 3.5, "ROCm support": "-"}
    new_nvidia = dict(old_nvidia, **{"CUDA compute capability": 8.9})
    apple = {"Vendor": "Apple", "Runtime HW class": "Apple Silicon", "CUDA compute capability": "-", "ROCm support": "-"}
    report.check(group, "efficiency: native backend reads the runtime table cell", F.gpu_runtime_efficiency(table, "llama.cpp", intel) == {"efficiency": 0.5, "backend": "native"})
    report.check(group, "efficiency: CUDA below the runtime minimum falls back to Vulkan", F.gpu_runtime_efficiency(table, "llama.cpp", old_nvidia) == {"efficiency": 0.45, "backend": "vulkan"})
    report.check(group, "efficiency: supported CUDA uses the native cell", F.gpu_runtime_efficiency(table, "llama.cpp", new_nvidia)["backend"] == "native")
    report.check(group, "efficiency: no fallback when the runtime has none", F.gpu_runtime_efficiency(table, "TensorRT-LLM", intel)["efficiency"] == 0)
    report.check(group, "efficiency: Apple never uses the Vulkan fallback", F.gpu_runtime_efficiency(table, "vLLM", apple)["efficiency"] == 0)
    report.check(group, "efficiency: unknown runtime is unsupported", F.gpu_runtime_efficiency(table, "nope", intel)["backend"] == "unsupported")

    # Workbook "2 What Can My HW Run" (v28): Llama 2 7B row, KV 4-bit, 2048 context, 4 users, vLLM on a 160 GB / 1712 GB/s accelerator.
    llama = {"Total params (B)": 6.74, "Active params (B)": 6.74, "Layers": 32, "KV heads": 32, "Head dim": 128, "Max context": 4096}
    near("workbook: KV cache cell F62", F.excel_memory_need(llama, 16, 2048, 4, 4)["kv"], 1.073741824)
    near("workbook: memory at FP16 cell M62", F.excel_memory_need(llama, 16, 2048, 4, 4)["total"], 16.009116006400003)
    near("workbook: memory at Q8 cell J62", F.excel_memory_need(llama, 8.5, 2048, 4, 4)["total"], 9.234991824)
    near("workbook: tok/s at FP16 cell T62", F.excel_decode_tok_per_second(llama, 16, 2048, 0.55, 1712, 4, 1, 4), 16.174534552468916)
    near("workbook: tok/s at Q8 cell K62", F.excel_decode_tok_per_second(llama, 8.5, 2048, 0.55, 1712, 4, 1, 4), 28.58533499862768)
    near("workbook: context is capped at the model limit", F.excel_memory_need(llama, 16, 32768, 1, 8)["kv"], F.excel_memory_need(llama, 16, 4096, 1, 8)["kv"])
    near("workbook: usable GPU memory cell F7", F.gpu_usable_gb(160, 1), 147.20000000000002)
    near("workbook: multi-card bandwidth scale cell F8", F.excel_quantity_scale(2), 1.6)
    near("workbook: card scale stops at 8 cards", F.excel_quantity_scale(12), 6.4)
    near("workbook: offload RAM cell F16", F.host_offload_gb(128, 256), 120)
    near("workbook: offload RAM is capped at the platform limit", F.host_offload_gb(512, 256), 248)
    near("workbook: vLLM on Intel GPU cell F11", F.gpu_runtime_efficiency(table, "vLLM", intel)["efficiency"], 0.55)
    near("workbook: vLLM on x86 CPU cell F14", F.cpu_runtime_efficiency(table, "vLLM", {"Runtime HW class": "x86 CPU"}), 0.4)
    rule_rows = {row["Task"]: row for row in load_rows("task_rules")}
    report.check(group, "workbook: task rule floors match the Task Rules sheet", [
        (task, F.positive(row["Base min bits (4-30B)"]), F.positive(row["Small model <4B min bits"]), F.positive(row["Large model >=30B min bits"]), row["VLM required?"])
        for task, row in rule_rows.items()] == [
        ("General chat / Q&A", 4.85, 6.6, 4.85, "N"), ("Summarisation / RAG", 4.85, 6.6, 4.85, "N"), ("Coding", 5.7, 8.5, 4.85, "N"),
        ("Maths / reasoning", 5.7, 8.5, 4.85, "N"), ("Agents / tool calling", 5.7, 8.5, 4.85, "N"), ("Vision (image + text)", 4.85, 6.6, 4.85, "Y"),
        ("Classification / extraction", 4.85, 4.85, 4.85, "N"), ("Translation / multilingual", 5.7, 8.5, 4.85, "N")])

    args, want = workbook_pipeline_case()
    got = run_pipeline(args)
    for key in ("usableGb", "effectiveBwGbps", "voiceMemoryGb", "voiceBwGbps"):
        near(f"workbook pipeline: {key}", got[key], want[key])
    for key in ("weights", "kvOverhead", "total", "bwNeeded", "cumulativeMemory", "cumulativeBw"):
        ok = all(close(row[key], value) for row, value in zip(got["llms"], want[key]))
        report.check(group, f"workbook pipeline: {key} for all four LLM slots", ok, f"got {[row[key] for row in got['llms']]}, expected {want[key]}")
    report.check(group, "workbook pipeline: slot statuses", [row["status"] for row in got["llms"]] == want["status"], [row["status"] for row in got["llms"]])
    report.check(group, "workbook pipeline: LLMs that fit", got["llmsFit"] == want["llmsFit"], got["llmsFit"])
    near("workbook pipeline: first LLM speed on its own", got["llms"][0]["tokAlone"], want["tokAlone0"])
    report.check(group, "workbook pipeline: pending voice weights are counted, not invented", got["voicePending"] == 1 and got["voice"][1]["status"] == "WEIGHTS PENDING")
    one = run_pipeline(dict(args, activity="sequential"))
    report.check(group, "pipeline: one-at-a-time needs only the largest LLM's bandwidth", close(one["bwRequiredGbps"], 15 + 675.68) and one["llmsFit"] == 4)
    over = run_pipeline(dict(args, llms=[{"model": dict(args["llms"][0]["model"], **{"Max context": 4096}), "bits": 4, "context": 8192, "target": 40}]))
    report.check(group, "pipeline: context above the model limit is flagged", over["llms"][0]["status"] == "REVIEW: CONTEXT > MODEL LIMIT")
    small = run_pipeline(dict(args, memoryGb=24))
    report.check(group, "pipeline: a small accelerator reports the memory shortfall", small["llms"][0]["status"] == "NO - GPU MEMORY")
    unused = run_pipeline(dict(args, llms=[args["llms"][0], {"model": None}]))
    report.check(group, "pipeline: unused slots are skipped", unused["llmsUsed"] == 1 and unused["llms"][1]["status"] == "UNUSED")

    # Time to first token (workbook Power sheet: F8 GFLOPs per token, F18 GPU prefill, F29 CPU prefill, G18 TTFT).
    near("prefill = 0.35 x TFLOPS x 1000 / (2 x active params)", F.prefill_tok_per_second(32.8, 100), 0.35 * 100 * 1000 / (2 * 32.8))
    near("prefill: each extra card adds 80%", F.prefill_tok_per_second(32.8, 100, 2), 1.8 * F.prefill_tok_per_second(32.8, 100))
    near("prefill: concurrent users share the compute", F.prefill_tok_per_second(32.8, 100, 1, 4), F.prefill_tok_per_second(32.8, 100) / 4)
    near("prefill on CPU uses the 0.6 factor", F.prefill_tok_per_second(32.8, 7.2, 1, 1, True), 0.35 * 0.6 * 7.2 * 1000 / (2 * 32.8))
    near("prefill MFU can be overridden", F.prefill_tok_per_second(10, 100, mfu=0.5), 0.5 * 100 * 1000 / 20)
    near("prefill is zero without a compute figure", F.prefill_tok_per_second(32.8, 0), 0)
    near("TTFT = prompt / prefill + 1 / decode", F.time_to_first_token(2048, 533.5365853658536, 11.0), 2048 / 533.5365853658536 + 1 / 11.0)
    report.check(group, "TTFT is unknown without a prefill or decode rate", F.time_to_first_token(2048, 0, 11) is None and F.time_to_first_token(2048, 500, 0) is None)
    near("prefill with CPU offload runs the two shares in series", F.split_prefill_tok_per_second(0.5, 400, 20), 1 / (0.5 / 400 + 0.5 / 20))

    # Concurrent sets of Stage 1-4: each set carries its own model copies (workbook Concurrent Model Selector B20, B22, B23).
    report.check(group, "sets: workbook example fits 2 sets by memory and 0 by bandwidth", (got["setsByMemory"], got["setsByBandwidth"], got["sets"], got["setsLimit"]) == (2, 0, 0, "bandwidth"),
                 (got["setsByMemory"], got["setsByBandwidth"], got["sets"], got["setsLimit"]))
    near("sets: memory per set is voice reserve plus every LLM", got["setMemoryGb"], 58.754411444)
    near("sets: bandwidth per set is voice plus every LLM", got["setBwGbps"], 1205.68)
    three = run_pipeline(dict(args, llms=args["llms"][:3], tflops=100, promptTokens=2048))
    report.check(group, "sets: three LLMs fit 3 sets by memory and 1 by bandwidth", (three["setsByMemory"], three["setsByBandwidth"], three["sets"], three["setsLimit"]) == (3, 1, 1, "bandwidth"),
                 (three["setsByMemory"], three["setsByBandwidth"], three["sets"]))
    half = run_pipeline(dict(args, llms=[dict(slot, target=10) for slot in args["llms"][:3]], tflops=100, promptTokens=2048))
    report.check(group, "sets: a lower speed target lets bandwidth carry more sets", half["setsByBandwidth"] == 5 and half["sets"] == 3 and half["setsLimit"] == "memory", (half["setsByMemory"], half["setsByBandwidth"]))
    one_at_a_time = run_pipeline(dict(args, activity="sequential", tflops=100, promptTokens=2048))
    report.check(group, "sets: one-at-a-time counts only the largest LLM's bandwidth", (one_at_a_time["setsByMemory"], one_at_a_time["setsByBandwidth"], one_at_a_time["sets"]) == (2, 1, 1))
    report.check(group, "sets: no answer while an input needs review", over["sets"] is None and over["setsLimit"] is None)
    first = three["llms"][0]
    near("pipeline TTFT alone = prompt / prefill + 1 / own speed", first["ttftAlone"], 2048 / (0.35 * 100 * 1000 / 8) + 1 / first["tokAlone"])
    near("pipeline TTFT with every set busy shares the prefill compute", first["ttftLoaded"], 2048 / (0.35 * 100 * 1000 / 8 / 3) + 1 / 40)
    near("pipeline TTFT reports the slowest LLM", three["ttftWorstLoaded"], max(row["ttftLoaded"] for row in three["llms"]))
    near("pipeline TTFT one-at-a-time does not share prefill inside a set", one_at_a_time["llms"][0]["ttftLoaded"], 2048 / (0.35 * 100 * 1000 / 8) + 1 / 40)
    report.check(group, "pipeline TTFT is unknown without a compute figure", got["ttftWorstLoaded"] is None and got["llms"][0]["ttftAlone"] is None)

    # Per-stage runtime and KV cache: each stage's bandwidth is counted at the accelerator, divided by its own runtime efficiency.
    one_llm = args["llms"][3:]
    shared = run_pipeline(dict(args, llms=[dict(one_llm[0], target=20)]))
    per_stage = run_pipeline(dict(args, utilization=None, voice=[dict(stage, utilization=0.5) for stage in args["voice"]], llms=[dict(one_llm[0], target=20, utilization=0.5, kvBits=16)]))
    report.check(group, "stage runtimes: the same runtime in every stage equals the shared setting",
                 (per_stage["sets"], per_stage["setsByMemory"], per_stage["setsByBandwidth"], per_stage["llms"][0]["status"]) == (shared["sets"], shared["setsByMemory"], shared["setsByBandwidth"], shared["llms"][0]["status"])
                 and close(per_stage["setRawBwGbps"], shared["setBwGbps"] / 0.5) and close(per_stage["llms"][0]["tokAlone"], shared["llms"][0]["tokAlone"]))
    mixed = run_pipeline(dict(args, utilization=None, voice=[dict(stage, utilization=u) for stage, u in zip(args["voice"], (0.6, 0.5, 0.55))], llms=[dict(one_llm[0], target=20, utilization=0.75, kvBits=8)]))
    near("stage runtimes: bandwidth per set = sum of stage bandwidth / stage runtime efficiency", mixed["setRawBwGbps"], 5 / 0.6 + 5 / 0.5 + 5 / 0.55 + 337.84 / 0.75)
    report.check(group, "stage runtimes: sets by bandwidth = accelerator bandwidth / bandwidth per set", mixed["setsByBandwidth"] == math.floor(1712 / mixed["setRawBwGbps"]) == 3, mixed["setsByBandwidth"])
    near("stage runtimes: the LLM alone decodes on the bandwidth the voice stages leave", mixed["llms"][0]["tokAlone"], (1712 - mixed["voiceRawBwGbps"]) * 0.75 / 16.892)
    near("stage KV cache: an LLM's own KV bits size its cache", mixed["llms"][0]["kvOverhead"], F.kv_per_token_gb(one_llm[0]["model"], 8) * 8192 + 16.892 * F.RUNTIME_SHARE)
    report.check(group, "stage KV cache: 8-bit is smaller than the shared 16-bit", mixed["llms"][0]["kvOverhead"] < shared["llms"][0]["kvOverhead"] and mixed["llms"][0]["kvBits"] == 8)
    no_llm_runtime = run_pipeline(dict(args, utilization=None, voice=[dict(stage, utilization=0.5) for stage in args["voice"]], llms=[dict(one_llm[0], utilization=0)]))
    report.check(group, "stage runtimes: an LLM runtime that does not run gives no answer", no_llm_runtime["llms"][0]["status"] == "REVIEW: RUNTIME" and no_llm_runtime["sets"] is None and no_llm_runtime["verdict"].startswith("REVIEW: pick a runtime"))
    no_voice_runtime = run_pipeline(dict(args, utilization=None, voice=[dict(stage, utilization=0 if index == 2 else 0.5) for index, stage in enumerate(args["voice"])], llms=[dict(one_llm[0], utilization=0.5)]))
    report.check(group, "stage runtimes: a voice runtime that does not run gives no answer", no_voice_runtime["voice"][2]["status"] == "REVIEW: RUNTIME" and no_voice_runtime["sets"] is None)
    on_cpu = run_pipeline(dict(args, utilization=None, voice=[dict(stage, utilization=0, placement="CPU") if index == 2 else dict(stage, utilization=0.5) for index, stage in enumerate(args["voice"])], llms=[dict(one_llm[0], target=20, utilization=0.5)]))
    report.check(group, "stage runtimes: a stage on the CPU needs no accelerator runtime", on_cpu["voice"][2]["status"] == "CPU - not sized" and on_cpu["sets"] is not None and close(on_cpu["voiceRawBwGbps"], 20))
    voice_kv = run_pipeline(dict(args, voice=[dict(stage, kvBits=bits) for stage, bits in zip(args["voice"], (16, 8, 4))], llms=[dict(one_llm[0], target=20)]))
    report.check(group, "stage KV cache: a voice stage's reserve scales with its own KV bits (FP16 = as entered)", [row["extraGb"] for row in voice_kv["voice"]] == [5, 2.5, 1.25] and [row["kvBits"] for row in voice_kv["voice"]] == [16, 8, 4])
    near("stage KV cache: memory per set drops by the reserve saved", shared["setMemoryGb"] - voice_kv["setMemoryGb"], 2.5 + 3.75)
    report.check(group, "stage KV cache: FP16 in every voice stage equals the workbook reserve", close(shared["voiceMemoryGb"], want["voiceMemoryGb"]) and all(row["kvBits"] == 16 for row in shared["voice"])
                 and close(run_pipeline(dict(args, voice=[dict(stage, kvBits=16) for stage in args["voice"]]))["voiceMemoryGb"], want["voiceMemoryGb"]))
    report.check(group, "stage KV cache: the shared LLM setting does not change a voice stage", close(run_pipeline(dict(args, kvBits=8))["voiceMemoryGb"], want["voiceMemoryGb"]))

    catalog = load_rows("models")
    sized = [m for m in catalog if F.has_sizing_specs(m)]
    bad = [m["Model"] for m in sized if not (0 < F.memory_need(m, 4.85, 8192)["total"] < 5000)]
    report.check(group, "every catalog model produces a finite memory need", not bad, sample(bad))
    smaller = [m["Model"] for m in sized if F.memory_need(m, 3, 8192)["total"] >= F.memory_need(m, 16, 8192)["total"]]
    report.check(group, "lower quantization always needs less memory", not smaller, sample(smaller))


# ----------------------------------------------------------- python = browser
def flag(value):
    return None if value is None else 1 if value else 0


def flatten_pipeline(result):
    out = [result["usableGb"], result["effectiveBwGbps"], result["voiceWeightsGb"], result["voiceExtraGb"], result["voiceMemoryGb"], result["voiceBwGbps"],
           result["voicePending"], result["memoryUsedGb"], result["memoryMarginGb"], result["bwRequiredGbps"], result["llmsUsed"], result["llmsFit"],
           flag(result["allFit"]), result["verdict"],
           result["setMemoryGb"], result["setBwGbps"], result["setsByMemory"], result["setsByBandwidth"], result["sets"], result["setsLimit"], result["ttftWorstLoaded"],
           result["rawBwGbps"], result["voiceRawBwGbps"], result["setRawBwGbps"]]
    for row in result["voice"]:
        out += [row["placement"], row["weightsGb"], row["extraGb"], row["bwGbps"], row["status"], row["utilization"], row["rawBwGbps"], row["kvBits"]]
    for row in result["llms"]:
        out += [row["status"], row["weights"], row["kvOverhead"], row["total"], row["bwNeeded"], row["cumulativeMemory"], row["cumulativeBw"], flag(row["fitsAlone"]), row["tokAlone"],
                row["target"], row["prefillAlone"], row["ttftAlone"], row["ttftLoaded"], row["utilization"], row["kvBits"], row["rawBwNeeded"], row["cumulativeRawBw"]]
    return out


def py_run(case, table, rules):
    fn, a = case["fn"], case["args"]
    if fn == "kvPerTokenGb":
        return [F.kv_per_token_gb(a["model"], a["kvBits"])]
    if fn in ("memoryNeed", "excelMemoryNeed"):
        r = (F.memory_need if fn == "memoryNeed" else F.excel_memory_need)(a["model"], a["bits"], a["context"], a["users"], a["kvBits"])
        return [r["weights"], r["kv"], r["runtime"], r["total"]]
    if fn == "quantityScale":
        return [F.quantity_scale(a["quantity"]), F.excel_quantity_scale(a["quantity"])]
    if fn == "decodeTokPerSecond":
        return [F.decode_tok_per_second(a["activeParamsB"], a["bits"], a["efficiency"], a["bandwidth"], a["users"], a["quantity"], a.get("utilization", 1))]
    if fn == "excelDecodeTokPerSecond":
        return [F.excel_decode_tok_per_second(a["model"], a["bits"], a["context"], a["efficiency"], a["bandwidth"], a["users"], a["quantity"], a["kvBits"])]
    if fn == "splitDecodeTokPerSecond":
        return [F.split_decode_tok_per_second(a["perTokenGb"], a["gpuShare"], a["gpuRate"], a["cpuRate"], a["users"])]
    if fn == "prefillTokPerSecond":
        return [F.prefill_tok_per_second(a["activeParamsB"], a["tflops"], a.get("quantity", 1), a.get("users", 1), a.get("cpu", False), a.get("mfu"))]
    if fn == "splitPrefillTokPerSecond":
        return [F.split_prefill_tok_per_second(a["gpuShare"], a["gpuRate"], a["cpuRate"])]
    if fn == "timeToFirstToken":
        return [F.time_to_first_token(a["promptTokens"], a["prefillTokPerSecond"], a["decodeTokPerSecond"])]
    if fn == "gpuUsableGb":
        return [F.gpu_usable_gb(a["memoryGb"], a["quantity"], a.get("headroomPct"))]
    if fn == "cpuUsableGb":
        return [F.cpu_usable_gb(a["ramGb"])]
    if fn == "hostOffloadGb":
        return [F.host_offload_gb(a["ramGb"], a.get("platformMaxGb"))]
    if fn == "offloadSupported":
        return [flag(F.offload_supported(a["runtime"]))]
    if fn == "isUnifiedMemory":
        return [flag(F.is_unified_memory(a["gpu"]))]
    if fn == "parseEfficiency":
        return [F.parse_efficiency(a["cell"])]
    if fn == "hardwareClass":
        return [F.hardware_class(a["row"], a["kind"])]
    if fn == "gpuRuntimeEfficiency":
        r = F.gpu_runtime_efficiency(table, a["runtime"], a["gpu"])
        return [r["efficiency"], r["backend"]]
    if fn == "cpuRuntimeEfficiency":
        return [F.cpu_runtime_efficiency(table, a["runtime"], a["cpu"])]
    if fn == "requiredBits":
        rule = F.task_rule(rules, a["task"])
        return [F.required_bits(a["model"], rule), flag(F.task_allows(a["model"], rule, a["bits"])), flag(F.is_vlm(a["model"]))]
    if fn == "hasSizingSpecs":
        return [flag(F.has_sizing_specs(a["model"]))]
    if fn == "voiceWeightsGb":
        return [F.voice_weights_gb(a["row"])]
    if fn == "pipelineFit":
        return flatten_pipeline(run_pipeline(a))
    raise ValueError(fn)


def build_cases():
    models, gpus, cpus = load_rows("models"), load_rows("gpus"), load_rows("cpus")
    libraries, task_rows, voice = load_rows("libraries"), load_rows("task_rules"), load_rows("voice_models")
    compute = {(row["Sheet"], row["Item"]): row["TFLOPS (est.)"] for row in load_rows("compute")}
    quants = [q["Bits/weight (eff.)"] for q in load_rows("quant")]
    cases = []
    add = lambda fn, **args: cases.append({"fn": fn, "args": args})
    for index, m in enumerate(models):
        gpu, cpu = gpus[index % len(gpus)], cpus[index % len(cpus)]
        gpu_tflops, cpu_tflops = compute.get(("GPUs", gpu["Accelerator"]), 0), compute.get(("CPUs", cpu["CPU / platform"]), 0)
        add("prefillTokPerSecond", activeParamsB=m["Active params (B)"], tflops=gpu_tflops, quantity=1 + index % 4, users=1 + index % 3)
        add("prefillTokPerSecond", activeParamsB=m["Active params (B)"], tflops=cpu_tflops, cpu=True, users=2, mfu=0.25)
        add("timeToFirstToken", promptTokens=2048, prefillTokPerSecond=F.prefill_tok_per_second(m["Active params (B)"], gpu_tflops), decodeTokPerSecond=3.5 + index)
        add("hasSizingSpecs", model=m)
        add("kvPerTokenGb", model=m, kvBits=16)
        for bits, context, users, kv in ((4.85, 8192, 1, 8), (8.5, 32768, 4, 16), (3, 262144, 2, 4)):
            add("memoryNeed", model=m, bits=bits, context=context, users=users, kvBits=kv)
            add("excelMemoryNeed", model=m, bits=bits, context=context, users=users, kvBits=kv)
        add("decodeTokPerSecond", activeParamsB=m["Active params (B)"], bits=4.85, efficiency=0.5, bandwidth=456, users=2, quantity=2, utilization=0.75)
        add("excelDecodeTokPerSecond", model=m, bits=5.7, context=16384, efficiency=0.6, bandwidth=960, users=2, quantity=4, kvBits=8)
        for row in task_rows:
            add("requiredBits", model=m, task=row["Task"], bits=5.7)
    for bits in quants:
        add("decodeTokPerSecond", activeParamsB=12, bits=bits, efficiency=0.65, bandwidth=1008, users=1, quantity=1)
    for library in libraries:
        add("offloadSupported", runtime=library["Library"])
        for hw in F.HW_CLASSES:
            add("parseEfficiency", cell=library[hw])
        for g in gpus:
            add("gpuRuntimeEfficiency", runtime=library["Library"], gpu=g)
        for c in cpus:
            add("cpuRuntimeEfficiency", runtime=library["Library"], cpu=c)
    for g in gpus:
        add("hardwareClass", row=g, kind="gpu")
        add("isUnifiedMemory", gpu=g)
        for quantity, headroom in ((1, None), (2, 10), (8, 0)):
            add("gpuUsableGb", memoryGb=g["Memory (GB)"], quantity=quantity, headroomPct=headroom)
    for c in cpus:
        add("hardwareClass", row=c, kind="cpu")
        add("cpuUsableGb", ramGb=c["Max RAM (GB)"])
        add("hostOffloadGb", ramGb=512, platformMaxGb=c["Max RAM (GB)"])
    for quantity in (1, 2, 4, 8, 12):
        add("quantityScale", quantity=quantity)
    for share in (0, 0.25, 0.5, 0.999, 1):
        add("splitDecodeTokPerSecond", perTokenGb=4.2, gpuShare=share, gpuRate=228, cpuRate=49.5, users=2)
    add("splitDecodeTokPerSecond", perTokenGb=4.2, gpuShare=0.5, gpuRate=228, cpuRate=0, users=1)
    for share in (0, 0.4, 1):
        add("splitPrefillTokPerSecond", gpuShare=share, gpuRate=533.5, cpuRate=23.05)
    add("timeToFirstToken", promptTokens=8192, prefillTokPerSecond=0, decodeTokPerSecond=20)
    add("timeToFirstToken", promptTokens=8192, prefillTokPerSecond=500, decodeTokPerSecond=None)
    add("gpuRuntimeEfficiency", runtime="not a runtime", gpu=gpus[0])
    add("requiredBits", model=models[0], task="not a task", bits=4.85)
    for row in voice:
        add("voiceWeightsGb", row=row)
    args, _ = workbook_pipeline_case()
    sized = [m for m in models if F.has_sizing_specs(m)]
    add("pipelineFit", **args)
    add("pipelineFit", **dict(args, tflops=200, promptTokens=2048))
    add("pipelineFit", **dict(args, llms=args["llms"][:3], tflops=98, promptTokens=8192, mfu=0.5))
    add("pipelineFit", **dict(args, activity="sequential", quantity=2, kvBits=8, headroomPct=5, tflops=117, promptTokens=512))
    add("pipelineFit", **dict(args, activity="sequential", quantity=2, kvBits=8, headroomPct=5))
    add("pipelineFit", **dict(args, memoryGb=24, bandwidthGbps=456, utilization=0.5))
    add("pipelineFit", **dict(args, utilization=0))
    staged = [dict(stage, utilization=u, kvBits=bits) for stage, u, bits in zip(args["voice"], (0.6, 0.5, 0.55), (16, 8, 4))]
    add("pipelineFit", **dict(args, utilization=None, voice=staged, llms=[dict(args["llms"][3], target=20, utilization=0.75, kvBits=8)], tflops=310, promptTokens=2048))
    add("pipelineFit", **dict(args, voice=staged, llms=[dict(slot, utilization=u, kvBits=kv) for slot, u, kv in zip(args["llms"], (0.5, 0.65, 0.8, 0.45), (4, 8, 16, 8))]))
    add("pipelineFit", **dict(args, utilization=None, voice=[dict(staged[0], utilization=0), *staged[1:]], llms=[dict(args["llms"][0], utilization=0.5)]))
    add("pipelineFit", **dict(args, utilization=None, voice=staged, llms=[dict(args["llms"][0], utilization=0), dict(args["llms"][1], utilization=0.6)], activity="sequential"))
    add("pipelineFit", **dict(args, voice=[dict(args["voice"][0], placement="CPU"), *args["voice"][1:]],
                              llms=[{"model": sized[0], "bits": 4.85, "context": 262144, "target": 20}, {"model": None, "bits": 4.85, "context": 8192, "target": 20},
                                    {"model": sized[-1], "bits": 0, "context": 8192, "target": 20}, {"model": sized[len(sized) // 2], "bits": 8.5, "context": 4096, "target": 5}]))
    for index in range(0, len(sized) - 3, 4):
        add("pipelineFit", **dict(args, tflops=40 + index, promptTokens=2048, llms=[{"model": m, "bits": 4.85, "context": 8192, "target": 20} for m in sized[index:index + 4]]))
    return {"libraries": libraries, "taskRules": task_rows, "cases": cases}


def same(a, b):
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return close(float(a), float(b))
    return a == b


def check_parity(report):
    group = "Python = browser"
    node = shutil.which("node")
    if not node:
        report.review(group, "Node.js not found", "install Node.js to compare scripts/formulas.py with docs/formulas.js")
        return 0
    payload = build_cases()
    case_file = OUT / "formula_cases.json"
    case_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    run = subprocess.run([node, str(SCRIPTS / "formula_harness.js"), str(case_file)], capture_output=True, text=True, encoding="utf-8")
    if not report.check(group, "docs/formulas.js runs under Node.js", run.returncode == 0, run.stderr.strip()[-400:]):
        return 0
    answer = json.loads(run.stdout)
    constants = answer["constants"]
    for name in ("WEIGHT_MARGIN", "RUNTIME_SHARE", "DEFAULT_HEADROOM_PCT", "EXTRA_CARD_SCALE", "HOST_RESERVE_GB", "VOICE_EXTRA_GB", "VOICE_BW_GBPS", "VOICE_RESERVE_BITS",
                 "PREFILL_MFU", "CPU_PREFILL_FACTOR", "PREFILL_GFLOPS_PER_B"):
        report.check(group, f"constant {name} is identical", constants[name] == getattr(F, name), f"js {constants[name]} vs py {getattr(F, name)}")
    report.check(group, "CPU-offload runtime list is identical", constants["OFFLOAD_RUNTIMES"] == list(F.OFFLOAD_RUNTIMES))
    report.check(group, "hardware class list is identical", constants["HW_CLASSES"] == list(F.HW_CLASSES))
    csv_rules = [[r["Task"], F.positive(r["Base min bits (4-30B)"]), F.positive(r["Small model <4B min bits"]), F.positive(r["Large model >=30B min bits"]), r["VLM required?"]] for r in payload["taskRules"]]
    report.check(group, "built-in fallback task rules equal data/task_rules.csv", constants["DEFAULT_TASK_RULES"] == csv_rules, constants["DEFAULT_TASK_RULES"])

    table, rules = F.build_efficiency_table(payload["libraries"]), F.build_task_rules(payload["taskRules"])
    round_trip = json.loads(json.dumps(payload["cases"]))
    by_fn = collections.defaultdict(lambda: [0, []])
    for case, js_result in zip(round_trip, answer["results"]):
        py_result = py_run(case, table, rules)
        by_fn[case["fn"]][0] += 1
        if len(py_result) != len(js_result) or not all(same(a, b) for a, b in zip(py_result, js_result)):
            label = case["args"].get("model", {}) or {}
            by_fn[case["fn"]][1].append(f"{label.get('Model', '')} py={py_result[:6]} js={js_result[:6]}")
    for fn, (count, mismatches) in sorted(by_fn.items()):
        report.check(group, f"{fn}: Python and browser agree on {count} cases", not mismatches, sample(mismatches, 2))
    return len(round_trip)


# --------------------------------------------------------------------- online
def http_get(url, timeout=25):
    request = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(2_000_000)
    except urllib.error.HTTPError as error:
        return error.code, b""
    except Exception as error:  # noqa: BLE001 - network failures are reported, not raised
        return 0, str(error).encode()


def check_online(report, tables):
    group = "Online"
    models = load_rows("models")
    repos = {m["Model"]: str(m["HF repo"]).strip() for m in models if m.get("HF repo") and "/" in str(m["HF repo"])}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(repos, pool.map(lambda repo: http_get(f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}"), repos.values())))
    reachable = sum(1 for status, _ in answers.values() if status == 200)
    if not reachable:
        report.review(group, "Hugging Face could not be reached", f"statuses {sample(sorted({status for status, _ in answers.values()}))}; online checks skipped")
        return {}
    gone = sorted(f"{name} ({repos[name]})" for name, (status, _) in answers.items() if status == 404)
    report.check(group, f"model repositories still exist on Hugging Face ({len(repos)} checked)", not gone, sample(gone, 10))
    blocked = sorted(f"{name}: HTTP {status}" for name, (status, _) in answers.items() if status not in (200, 404))
    if blocked:
        report.review(group, "model repositories that did not answer (gated, private or rate-limited)", sample(blocked, 10))
    by_name = {m["Model"]: m for m in models}
    card_totals = {row["Item"] for row in tables["evidence"] if row["Sheet"] == "Models" and row["Parameter"] == "Total params (B)"
                   and row["Status"] == "VERIFIED" and "safetensors" not in row["Source type"].lower()}
    drift, checked = [], 0
    for name, (status, body) in answers.items():
        if status != 200:
            continue
        info = json.loads(body)
        total = (info.get("safetensors") or {}).get("total")
        if total:
            checked += 1
            listed = F.positive(by_name[name].get("Total params (B)"))
            if listed and name not in card_totals and abs(total / 1e9 - listed) / listed > 0.05:
                drift.append(f"{name}: table {listed:g}B vs repo {total / 1e9:.2f}B")
    report.check(group, f"repository metadata was read for parameter counts ({checked} repos publish them)", checked > 0)
    if drift:
        report.review(group, "parameter counts more than 5% away from the repo's safetensors total and not sourced from the model card", sample(drift, 12))

    year = max(int(F.positive(m.get("Year"))) for m in models)
    orgs = sorted({repo.split("/")[0] for name, repo in repos.items() if int(F.positive(by_name[name].get("Year"))) >= year - 1})
    known = {repo.lower() for repo in repos.values()}
    known_stems = {re.sub(r"[^a-z0-9]", "", repo.split("/")[1].lower()) for repo in repos.values()}
    cutoff = (datetime.date.today() - datetime.timedelta(days=120)).isoformat()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        listings = dict(zip(orgs, pool.map(lambda org: http_get(f"https://huggingface.co/api/models?author={urllib.parse.quote(org)}&sort=createdAt&direction=-1&limit=12"), orgs)))
    candidates = []
    for org, (status, body) in listings.items():
        if status != 200:
            continue
        for item in json.loads(body):
            repo, created = item.get("id", ""), str(item.get("createdAt", ""))[:10]
            stem = re.sub(r"[^a-z0-9]", "", repo.split("/")[-1].lower())
            if created < cutoff or repo.lower() in known or QUANT_REPO.search(repo) or item.get("pipeline_tag") not in ("text-generation", "image-text-to-text", "any-to-any", None):
                continue
            if any(stem.startswith(k) or k.startswith(stem) for k in known_stems):
                continue
            candidates.append(f"{repo} ({created})")
    report.check(group, f"newest repositories were listed for {len(orgs)} publishers", any(status == 200 for status, _ in listings.values()))
    if candidates:
        report.review(group, f"{len(candidates)} recent repositories not in the catalog (review before adding; many are fine-tunes or variants)", sample(sorted(candidates, key=lambda c: c[-11:], reverse=True), 25))

    urls = collections.OrderedDict()
    for table, key in (("gpus", "Accelerator"), ("cpus", "CPU / platform")):
        for row in tables[table]:
            for url in (part.strip() for part in row.get("Source URL", "").split(";")):
                if url.startswith("http"):
                    urls.setdefault(url, row[key])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = dict(zip(urls, pool.map(lambda url: http_get(url)[0], urls)))
    ok = sum(1 for status in statuses.values() if status == 200)
    report.check(group, f"hardware source links were requested ({ok} of {len(urls)} answered 200)", ok > 0)
    dead = sorted(f"{urls[url]}: {url}" for url, status in statuses.items() if status in (404, 410))
    if dead:
        report.review(group, "hardware source links answering 404/410 (confirm in a browser; some vendor sites refuse scripts)", sample(dead, 10))
    refused = collections.Counter(status for status in statuses.values() if status not in (200, 404, 410))
    if refused:
        report.review(group, "hardware source links that refused or timed out for a script", ", ".join(f"HTTP {status or 'no answer'} x{count}" for status, count in sorted(refused.items())))
    return {"hf_repos_checked": len(repos), "hf_repos_ok": reachable, "new_release_candidates": candidates, "param_drift": drift, "dead_links": dead}


# ----------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--online", action="store_true", help="also check Hugging Face repos, source links and newer releases")
    parser.add_argument("--no-write", action="store_true", help="do not update docs/verification.json")
    options = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    report = Report()
    tables = {name: read_csv(name) for name in REQUIRED}
    tables["reference_models"] = read_csv("reference_models")
    counts = check_data(report, tables)
    check_references(report, tables)
    check_build(report)
    check_wiring(report)
    check_themes(report)
    check_formulas(report, tables)
    counts["formula_cases_python_vs_browser"] = check_parity(report)
    online = check_online(report, tables) if options.online else {}

    print(f"{'Group':<20}{'passed':>8}{'failed':>8}{'review':>8}")
    groups = []
    for name in GROUPS:
        tally = report.tally(name)
        if not sum(tally.values()):
            continue
        groups.append({"name": name, **tally})
        print(f"{name:<20}{tally['pass']:>8}{tally['fail']:>8}{tally['warn']:>8}")
    total = report.tally()
    print(f"{'TOTAL':<20}{total['pass']:>8}{total['fail']:>8}{total['warn']:>8}")
    failures = [row for row in report.rows if row["status"] == "fail"]
    reviews = [row for row in report.rows if row["status"] == "review"]
    if failures:
        print("\nFAILED")
        for row in failures:
            print(f"  [{row['group']}] {row['name']}: {row['detail']}")
    if reviews:
        print("\nTO REVIEW (does not fail the run)")
        for row in reviews:
            print(f"  [{row['group']}] {row['name']}: {row['detail']}")

    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (OUT / "reverify_report.json").write_text(json.dumps({"generated": generated, "summary": total, "counts": counts, "online": online, "checks": report.rows},
                                                         indent=1, ensure_ascii=False), encoding="utf-8")
    if not options.no_write:
        public = {"generated": generated, "mode": "online" if options.online else "offline", "summary": total, "groups": groups, "counts": counts,
                  "failures": [f"{row['group']}: {row['name']}" for row in failures][:20]}
        (DOCS / "verification.json").write_text(json.dumps(public, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {total['pass']} passed, {total['fail']} failed, {total['warn']} to review. Detail: out/reverify_report.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
