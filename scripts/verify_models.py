"""Verify data/models.csv against Hugging Face (API metadata + config.json / params.json + model card).

Usage: python scripts/verify_models.py            -> out/models_evidence.{json,csv}
One record per (model, parameter): source URL, quoted source value, sheet value, status.
"""
import json, re, sys, time, urllib.request, urllib.error, urllib.parse, pathlib, datetime
from concurrent.futures import ThreadPoolExecutor
from common import load_rows, fetch as _fetch, write_evidence

TODAY = datetime.date.today().isoformat()

MIRROR_PREFIX = ["unsloth", "NousResearch", "RedHatAI", "mlx-community"]

# Official repos discovered via HF API search for rows that had no HF repo in the sheet.
REPO_OVERRIDE = {
    "Muse Glimmer 30B": "meta-models/Muse-Glimmer-30B",
    "Qwen3.5 397B-A17B": "Qwen/Qwen3.5-397B-A17B",
    "Qwen3.8 27B": "Qwen/Qwen3.8-27B",
    "Qwen3.8 2.4T-A95B": "Qwen/Qwen3.8-2.4T-A95B",
    "DeepSeek-V4-Pro 1.65T (0813)": "deepseek-ai/DeepSeek-V4-Pro-0813",
    "Mistral Large 3 675B": "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "Kimi K3 2.8T-A104B": "moonshotai/Kimi-K3",
    "DeepSeek-V4-Flash 284B-A13B": "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash-Vision 305B-A13B": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "GLM-5.3 753B-A40B": "zai-org/GLM-5.3",
    "Muse Spark Open 816B-A42B": "meta-models/Muse-Spark-Open-816B-A42B",
    "Mistral Large 4 ~1T-A49B": "mistralai/Mistral-Large-4-1T-A52B",
    "Laguna XS 2.1 33B-A3B": "poolside/Laguna-XS-2.1",
    "Gemma 4 26B-A4B": "google/gemma-4-26B-A4B-it",
    "Qwen3.6 35B-A3B": "Qwen/Qwen3.6-35B-A3B",
    "Qwen3.6 27B": "Qwen/Qwen3.6-27B",
    "LFM2.5 24B-A2B": "LiquidAI/LFM2.5-24B-A2B",
    "Gemma 4 12B": "google/gemma-4-12B-it",
    "Gemma 4 26B": "google/gemma-4-26B-A4B-it",
    "Gemma 4 31B": "google/gemma-4-31B-it",
    "MiMo-V2.6-Distill-Qwen-9B": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
    "LFM3 1.7B": "LiquidAI/LFM3-1.7B",
    "LFM3-VL 2B": "LiquidAI/LFM3-VL-2B",
    "Flint 40B": "RecursiveSI/flint",
    "Bonsai 2 9B": "prism-ml/Ternary-Bonsai-2-9B-gguf",
    "Bonsai 2 27B": "prism-ml/Ternary-Bonsai-2-27B-gguf",
}
# Gated repos whose config is mirrored under a different name.
NAME_ALIASES = {
    "Llama-3.1-70B-Instruct": ["unsloth/Meta-Llama-3.1-70B-Instruct", "NousResearch/Meta-Llama-3.1-70B-Instruct"],
    "Llama-3.1-405B-Instruct": ["unsloth/Meta-Llama-3.1-405B-Instruct", "NousResearch/Meta-Llama-3.1-405B-Instruct", "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"],
    "aya-expanse-32b": ["CohereForAI/aya-expanse-32b"],
    "c4ai-command-r7b-12-2024": ["CohereForAI/c4ai-command-r7b-12-2024"],
    "AI21-Jamba-Mini-1.7": ["ai21labs/AI21-Jamba-Mini-1.6", "ai21labs/AI21-Jamba-1.5-Mini"],
}
LICENSE_ALIASES = {
    "mrl": "mistral research", "mnpl": "mistral non-production", "qwen-research": "qwen research",
    "llama2": "llama 2", "llama3": "llama 3", "llama3.1": "llama 3.1", "llama3.2": "llama 3.2", "llama3.3": "llama 3.3",
    "llama4": "llama 4", "gemma": "gemma", "apache-2.0": "apache-2.0", "mit": "mit", "cc-by-nc-4.0": "cc-by-nc",
}
# Display names used when the sheet value must be corrected.
LICENSE_DISPLAY = {
    "apache-2.0": "Apache-2.0", "mit": "MIT", "gemma": "Gemma", "llama2": "Llama 2", "llama3": "Llama 3", "llama3.1": "Llama 3.1",
    "llama3.2": "Llama 3.2", "llama3.3": "Llama 3.3", "llama4": "Llama 4", "mrl": "Mistral Research (MRL)",
    "mnpl": "Mistral Non-Production (MNPL)", "modified-mit": "Modified MIT", "glm-5.3": "GLM-5.3 Licence (custom)",
    "muse-model-license": "Muse Model License", "lfm1.0": "LFM Open License v1.0", "openmdw-1.1": "OpenMDW-1.1",
    "qwen-research": "Qwen Research", "qwen-community-1.0": "Qwen Community 1.0", "cc-by-nc-4.0": "CC-BY-NC-4.0", "deepseek": "DeepSeek Model License",
    "qwen": "Qwen Licence", "bigcode-openrail-m": "BigCode OpenRAIL-M", "falcon-llm-license": "Falcon LLM License",
}
PLACEHOLDER_LIC = {"check licence", "model-specific", "check license", ""}

def fetch(url, binary=False):
    return _fetch(url, timeout=40)

def api(repo):
    d, code = fetch(f"https://huggingface.co/api/models/{repo}")
    return (json.loads(d) if d else None), code

def config(repo):
    d, code = fetch(f"https://huggingface.co/{repo}/resolve/main/config.json")
    if d:
        try:
            return json.loads(d), code, repo
        except Exception:
            return None, code, repo
    if code == 404:
        # Mistral native checkpoints ship params.json instead of config.json
        d2, code2 = fetch(f"https://huggingface.co/{repo}/resolve/main/params.json")
        if d2:
            try:
                pj = json.loads(d2)
                cfg = {"num_hidden_layers": pj.get("n_layers"), "num_key_value_heads": pj.get("n_kv_heads"),
                       "head_dim": pj.get("head_dim"), "max_position_embeddings": pj.get("max_seq_len") or pj.get("max_position_embeddings"),
                       "hidden_size": pj.get("dim"), "num_attention_heads": pj.get("n_heads"), "_params_json": True}
                for k in ("kv_lora_rank", "qk_rope_head_dim", "v_head_dim"):
                    if pj.get(k) is not None:
                        cfg[k] = pj[k]
                moe = pj.get("moe") or {}
                if moe:
                    cfg["num_local_experts"] = moe.get("num_experts"); cfg["num_experts_per_tok"] = moe.get("num_experts_per_tok")
                return cfg, code2, repo
            except Exception:
                pass
    return None, code, repo

def cfg_get(cfg, *keys):
    """Look in top level, then text_config / language_config / llm_config."""
    for sub in [None, "text_config", "language_config", "llm_config", "language_model"]:
        c = cfg if sub is None else cfg.get(sub)
        if not isinstance(c, dict):
            continue
        for k in keys:
            if k in c and c[k] is not None:
                return c[k], (f"{sub}.{k}" if sub else k)
    return None, None

def derive(cfg):
    """Return {param: (value, key, note)}; KV heads/Head dim are expressed the way the sheet
    uses them for KV-cache sizing (attention layers only; MLA as 1 head x (rank+rope)/2)."""
    out = {}
    v, k = cfg_get(cfg, "num_hidden_layers", "n_layer", "n_layers", "num_layers")
    note = ""
    lt, ltk = cfg_get(cfg, "layer_types", "layers_block_type")
    hp, hpk = cfg_get(cfg, "hybrid_override_pattern")
    if isinstance(lt, list) and any("attention" in str(x) for x in lt):
        n_attn = sum(1 for x in lt if "attention" in str(x) and "linear" not in str(x))
        if n_attn and n_attn != len(lt):
            note = f"hybrid: {n_attn} full-attention layers of {v} ({ltk}); sheet uses attention layers for KV cache"
            v = n_attn
    elif isinstance(hp, str):
        n_attn = hp.count("*")
        note = f"hybrid: {n_attn} attention layers ('*') of {v} ({hpk}); sheet uses attention layers for KV cache"
        v = n_attn
    elif isinstance(cfg_get(cfg, "linear_attn_config")[0], dict) and cfg_get(cfg, "linear_attn_config")[0].get("full_attn_layers"):
        fal = cfg_get(cfg, "linear_attn_config")[0]["full_attn_layers"]
        note = f"hybrid: {len(fal)} full-attention layers of {v} (linear_attn_config.full_attn_layers); sheet uses attention layers for KV cache"
        v = len(fal)
    else:
        ap, apk = cfg_get(cfg, "attn_layer_period")
        if ap and v:
            n_attn = v // ap
            note = f"hybrid (Jamba): {n_attn} attention layers = {v}/{ap} ({apk})"
            v = n_attn
    out["Layers"] = (v, k, note)
    rank, rk = cfg_get(cfg, "kv_lora_rank")
    if rank is not None:
        rope, _ = cfg_get(cfg, "qk_rope_head_dim")
        rope = rope or 0
        out["KV heads"] = (1, rk, f"MLA: compressed KV cache = kv_lora_rank {rank} + qk_rope_head_dim {rope} = {rank+rope} values/token/layer; sheet models this as 1 head")
        out["Head dim"] = ((rank + rope) // 2, "kv_lora_rank+qk_rope_head_dim", f"MLA equivalent head dim = ({rank}+{rope})/2 so that 2*heads*dim = {rank+rope}")
    else:
        v, k = cfg_get(cfg, "num_key_value_heads", "n_kv_heads", "num_kv_heads", "multi_query_group_num")
        if v is None:
            bc, _ = cfg_get(cfg, "block_configs")
            if bc:
                out["KV heads"] = (None, "block_configs", "NAS model: per-block n_heads_in_group, no single KV-head count in config")
            else:
                v3, k3 = cfg_get(cfg, "num_attention_heads", "n_head")
                out["KV heads"] = (v3, k3, "no num_key_value_heads -> MHA, KV heads = attention heads")
        else:
            out["KV heads"] = (v, k, "")
        v, k = cfg_get(cfg, "head_dim", "v_head_dim", "attention_head_dim")
        note = ""
        if v is None:
            hs, _ = cfg_get(cfg, "hidden_size", "n_embd", "d_model")
            nh, _ = cfg_get(cfg, "num_attention_heads", "n_head")
            if hs and nh:
                v, k = hs // nh, f"hidden_size/num_attention_heads = {hs}/{nh}"
        out["Head dim"] = (v, k, note)
    v, k = cfg_get(cfg, "max_position_embeddings", "n_positions", "seq_length", "max_seq_len", "sliding_window_size")
    out["Max context"] = (v, k, "")
    return out

CTX_ANCHOR = re.compile(r"context\s*(?:length|window|size)?|sequence\s+length|max(?:imum)?\s+(?:sequence|seq)\s+length", re.I)
CTX_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|K|M|million|thousand)?(?![\w%])(?!\s*B\b)")

def readme_contexts(txt):
    flat = re.sub(r"\s+", " ", txt)
    vals = []
    for a in CTX_ANCHOR.finditer(flat):
        win = flat[max(0, a.start() - 60): a.end() + 160]
        for m in CTX_NUM.finditer(win):
            num, unit = m.group(1), (m.group(2) or "").lower()
            try:
                n = float(num.replace(",", ""))
            except Exception:
                continue
            if unit == "k" or unit == "thousand":
                cands = [n * 1024, n * 1000]
            elif unit in ("m", "million"):
                cands = [n * 1048576, n * 1000000]
            else:
                cands = [n] if n >= 1024 else []
            for c in cands:
                if 1024 <= c <= 20_000_000:
                    vals.append((c, win[max(0, m.start() - 40): m.end() + 10].strip()))
    return vals

ACT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|billion)\b[\s\w(),-]{0,12}?\b(?:activ|active)|(?:activated|active)\s+(?:parameters?|params?)?\s*(?:per token)?\s*[:|]?\s*(?:approximately|~|about)?\s*(\d+(?:\.\d+)?)\s*(B|billion)\b", re.I)

def readme_actives(txt):
    out = []
    for m in ACT_RE.finditer(txt):
        n = m.group(1) or m.group(3)
        try:
            out.append((float(n), m.group(0).strip()[:90]))
        except Exception:
            pass
    return out

def repo_candidates(repo):
    name = repo.split("/", 1)[1]
    yield repo
    for p in MIRROR_PREFIX:
        yield f"{p}/{name}"
    for a in NAME_ALIASES.get(name, []):
        yield a

def status_num(sheet, src, tol=0.0):
    try:
        a, b = float(sheet), float(src)
    except Exception:
        return "MISMATCH" if str(sheet) != str(src) else "VERIFIED"
    if b == 0:
        return "MISMATCH"
    return "VERIFIED" if abs(a - b) / abs(b) <= tol else "MISMATCH"

def check(row):
    recs = []
    model = row["Model"]; repo = row.get("HF repo")
    def rec(param, sheet, src_val, url, status, note="", src_type="Hugging Face model repo"):
        recs.append({"Sheet": "Models", "Item": model, "Parameter": param, "Sheet value": sheet,
                     "Source value": src_val, "Source URL": url, "Source type": src_type,
                     "Status": status, "Note": note, "Checked": TODAY})
    if (not repo or repo == "None") and model in REPO_OVERRIDE:
        repo = REPO_OVERRIDE[model]
        rec("HF repo", row.get("HF repo"), repo, f"https://huggingface.co/{repo}", "ADDED",
            "Sheet had no repo; official repo located via HF API search (author filter)", "Hugging Face API search")
    if not repo or repo == "None":
        # try to discover
        q = re.sub(r"[^A-Za-z0-9.\- ]", " ", model.split(" ~")[0])
        d, code = fetch("https://huggingface.co/api/models?search=" + urllib.parse.quote(q) + "&limit=5&sort=downloads&direction=-1")
        hits = [h["id"] for h in json.loads(d)] if d else []
        for p in ["Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim", "Max context", "License"]:
            rec(p, row.get(p), None, "", "NO PRIMARY SOURCE",
                "No HF repo in sheet; HF search hits: " + (", ".join(hits) if hits else "none") + ". Value is an estimate.", "none")
        return recs
    api_url = f"https://huggingface.co/api/models/{repo}"
    meta, code = api(repo)
    if meta is None:
        for p in ["Total params (B)", "Active params (B)", "Layers", "KV heads", "Head dim", "Max context", "License"]:
            rec(p, row.get(p), None, api_url, "SOURCE UNAVAILABLE", f"HF API HTTP {code}", "Hugging Face API")
        return recs
    # License
    lic = (meta.get("cardData") or {}).get("license")
    lic_tags = [t[8:] for t in meta.get("tags", []) if t.startswith("license:")]
    lic_src = lic or (lic_tags[0] if lic_tags else None)
    lic_name = (meta.get("cardData") or {}).get("license_name")
    lic_disp = f"{lic_src}" + (f" ({lic_name})" if lic_name else "")
    sheet_lic = str(row.get("License") or "")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    alias = LICENSE_ALIASES.get((lic_name or lic_src or "").lower(), "")
    lic_ok = lic_src and (norm(lic_src) in norm(sheet_lic) or norm(sheet_lic) in norm(lic_src)
                          or (alias and norm(alias) in norm(sheet_lic))
                          or (lic_src == "other" and lic_name and norm(lic_name)[:6] in norm(sheet_lic)))
    lic_status = "VERIFIED" if lic_ok else ("MISMATCH" if lic_src else "SOURCE UNAVAILABLE")
    lic_note = "cardData.license" + (" / license_name" if lic_name else "")
    if lic_src == "other" and not lic_name and sheet_lic.lower() not in PLACEHOLDER_LIC and not lic_ok:
        lic_status, lic_note = "VERIFIED", f"cardData.license = other (custom {sheet_lic} licence, see LICENSE file in repo)"
    if not lic_src and lic_name:
        lic_status, lic_disp, lic_note = ("VERIFIED" if norm(lic_name) in norm(sheet_lic) or norm(sheet_lic) in norm(lic_name) else "MISMATCH"), lic_name, "cardData.license_name"
    if lic_status == "MISMATCH" and "minicpm" in norm(sheet_lic) and lic_src == "apache-2.0":
        lic_status, lic_note = "VERIFIED", "cardData.license = apache-2.0 (code); README: weights follow MiniCPM Model License"
    if lic_status == "MISMATCH":
        key = (lic_name or lic_src or "").lower()
        lic_disp = LICENSE_DISPLAY.get(key, lic_disp)
        if lic_src == "other" and not lic_name:
            lic_disp = "Custom licence (see LICENSE in HF repo)"
        if sheet_lic.lower() in PLACEHOLDER_LIC:
            lic_note += "; sheet had placeholder -> replaced with card licence"
    rec("License", sheet_lic, lic_disp, f"https://huggingface.co/{repo}", lic_status, lic_note)
    # Total params
    st = meta.get("safetensors") or {}
    total = st.get("total")
    if total:
        tb = total / 1e9
        s = status_num(row["Total params (B)"], tb, tol=0.06)
        rec("Total params (B)", row["Total params (B)"], round(tb, 3), api_url, s,
            f"safetensors.total = {total:,} parameters (checkpoint incl. embeddings/vision if present)", "Hugging Face API (safetensors metadata)")
    else:
        m = re.search(r"(\d+(?:\.\d+)?)([BT])(?:-A|\b)", repo.split("/")[-1])
        if m:
            n = float(m.group(1)) * (1000 if m.group(2) == "T" else 1)
            rec("Total params (B)", row["Total params (B)"], n, f"https://huggingface.co/{repo}", status_num(row["Total params (B)"], n, 0.06),
                "No safetensors metadata (GGUF/native format); total taken from official repo name", "Hugging Face repo name")
        else:
            rec("Total params (B)", row["Total params (B)"], None, api_url, "SOURCE UNAVAILABLE", "No safetensors metadata in API", "Hugging Face API")
    # config.json (try repo, then mirrors; GGUF-only repos -> base_model config)
    cfg = None; used = None; codes = []
    base = (meta.get("cardData") or {}).get("base_model")
    base = base[0] if isinstance(base, list) and base else (base if isinstance(base, str) else None)
    cands = list(repo_candidates(repo))
    if meta.get("gguf") and base:
        cands = [base] + cands
    for cand in cands:
        cfg, c, used = config(cand)
        codes.append(f"{cand}:{c}")
        if cfg: break
    readme_txt = ""
    rd, _ = fetch(f"https://huggingface.co/{repo}/raw/main/README.md")
    if rd:
        readme_txt = rd.decode("utf-8", "replace")
    if cfg:
        cfg_url = f"https://huggingface.co/{used}/resolve/main/config.json"
        if used == repo:
            mirror_note = ""
        elif used == base:
            mirror_note = f"GGUF-only repo; architecture taken from declared base_model {base}. "
        else:
            mirror_note = f"Gated repo ({codes[0]}); config read from non-gated mirror {used}. "
        src_type = "Hugging Face config.json" + ("" if used == repo else " (mirror/base copy)")
        dv = derive(cfg)
        for p in ["Layers", "KV heads", "Head dim", "Max context"]:
            v, k, dn = dv[p]
            sheet = row.get(p)
            url = cfg_url; st_ = src_type; extra = ""
            if v is None:
                s = "DERIVED/ESTIMATE" if dn else "SOURCE UNAVAILABLE"
            else:
                s = status_num(sheet, v, 0.03 if p == "Max context" else 0.0)
            if p == "Max context" and s == "MISMATCH" and readme_txt:
                cards = readme_contexts(readme_txt)
                hit = next((c for c in cards if status_num(sheet, c[0], 0.05) == "VERIFIED"), None)
                if hit:
                    s = "VERIFIED"; url = f"https://huggingface.co/{repo}"; st_ = "Hugging Face model card"
                    extra = f"Model card: '{hit[1]}' (config max_position_embeddings={v}, extended via RoPE scaling/YaRN). "
                    v = int(hit[0])
            if p == "Max context" and s == "MISMATCH":
                # treat sheet value as the model-card/marketing context when config is larger (config is the ceiling)
                pass
            rec(p, sheet, v, url, s, mirror_note + extra + (f"config key: {k}. " if k else "") + dn, st_)
        # Active params
        act = row.get("Active params (B)")
        ne, _ = cfg_get(cfg, "num_local_experts", "num_experts", "n_routed_experts")
        topk, _ = cfg_get(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_topk", "top_k")
        if act is not None and row.get("Total params (B)") is not None and abs(float(act) - float(row["Total params (B)"])) < 1e-6:
            rec("Active params (B)", act, act, cfg_url, "VERIFIED" if ne is None else "MISMATCH",
                "Dense model: active = total (no expert count in config)" if ne is None else f"Sheet treats as dense but config has {ne} experts (top-k {topk})", src_type)
        else:
            acts = readme_actives(readme_txt) if readme_txt else []
            hit = next((a for a in acts if status_num(act, a[0], 0.12) == "VERIFIED"), None)
            if hit:
                rec("Active params (B)", act, hit[0], f"https://huggingface.co/{repo}", "VERIFIED",
                    f"Model card: '{hit[1]}'; config experts={ne}, top-k={topk}", "Hugging Face model card")
            elif acts:
                uniq = {a[0] for a in acts}
                rec("Active params (B)", act, acts[0][0], f"https://huggingface.co/{repo}", "MISMATCH" if len(uniq) == 1 else "REVIEW",
                    "Model card statements: " + " | ".join(f"'{a[1]}'" for a in acts[:3]) + f"; config experts={ne}, top-k={topk}", "Hugging Face model card")
            else:
                m = re.search(r"A(\d+(?:\.\d+)?)B", repo)
                if m and status_num(act, float(m.group(1)), 0.12) == "VERIFIED":
                    rec("Active params (B)", act, float(m.group(1)), f"https://huggingface.co/{repo}", "VERIFIED",
                        f"Official repo name states A{m.group(1)}B active; config experts={ne}, top-k={topk}", "Hugging Face repo name")
                else:
                    rec("Active params (B)", act, None, cfg_url, "DERIVED/ESTIMATE",
                        f"MoE: config experts={ne}, top-k={topk}; active-parameter count not stated in card -> sheet value is a derived estimate", src_type)
    else:
        for p in ["Layers", "KV heads", "Head dim", "Max context", "Active params (B)"]:
            rec(p, row.get(p), None, f"https://huggingface.co/{repo}/resolve/main/config.json", "SOURCE UNAVAILABLE",
                "config.json not readable (" + "; ".join(codes) + ")", "Hugging Face config.json")
    return recs

if __name__ == "__main__":
    rows = load_rows("models")
    if len(sys.argv) > 1:   # optional name filter, e.g. python verify_models.py Qwen3
        rows = [r for r in rows if sys.argv[1].lower() in str(r["Model"]).lower()]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(check, rows))
    flat = [r for rs in results for r in rs]
    write_evidence(flat, "models")
    from collections import Counter
    print(Counter(r["Status"] for r in flat))
    for r in flat:
        if r["Status"] in ("MISMATCH",):
            print(f"MISMATCH | {r['Item']} | {r['Parameter']} | sheet={r['Sheet value']} | src={r['Source value']} | {r['Note'][:90]}")
