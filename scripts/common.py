"""Shared helpers: repo paths, CSV loading with numeric coercion, cached HTTP fetch."""
import csv, pathlib, re, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"
CACHE = OUT / "cache"
for p in (OUT, CACHE):
    p.mkdir(parents=True, exist_ok=True)

NUMERIC = {"Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim", "Max context", "Year",
           "Memory (GB)", "Bandwidth (GB/s)", "TDP / board power (W)", "CUDA compute capability",
           "Cores", "Base / default power (W)", "Max turbo / max power (W)", "Memory BW (GB/s)", "Max RAM (GB)", "CPU PCIe lanes",
           "Bits / weight", "Bits/weight (eff.)", "GB per 1B params", "TFLOPS (est.)"}

def _coerce(k, v):
    if v is None or v == "":
        return None
    if k in NUMERIC:
        try:
            f = float(v)
            return int(f) if f.is_integer() and k not in ("CUDA compute capability",) else f
        except ValueError:
            return v
    return v

def load_rows(name):
    """Load data/<name>.csv as a list of dicts (numbers coerced); adds _row = 1-based CSV data index."""
    path = DATA / f"{name}.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = []
        for i, r in enumerate(csv.DictReader(f), start=1):
            d = {k: _coerce(k, v) for k, v in r.items()}
            d["_row"] = i
            rows.append(d)
    return rows

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/json,application/pdf;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}

def fetch(url, timeout=60):
    """GET with on-disk cache. Returns (bytes|None, http_status)."""
    key = CACHE / (re.sub(r"[^A-Za-z0-9._-]", "_", url)[:170] + ".bin")
    if key.exists():
        d = key.read_bytes()
        return (None, int(d[7:10])) if d.startswith(b"__ERR__") else (d, 200)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            d = r.read()
        key.write_bytes(d)
        return d, 200
    except urllib.error.HTTPError as e:
        key.write_bytes(b"__ERR__%03d" % e.code)
        return None, e.code
    except Exception:
        return None, 0

def write_evidence(records, name):
    import json
    json.dump(records, open(OUT / f"{name}_evidence.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    cols = ["Sheet", "Item", "Parameter", "Sheet value", "Source value", "Status", "Source type", "Source URL", "Note", "Checked"]
    with open(OUT / f"{name}_evidence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in records:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
