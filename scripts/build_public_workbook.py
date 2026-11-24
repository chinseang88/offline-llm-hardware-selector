"""Build the public Excel edition of the selector from the published tables in data/ only.

Usage: python scripts/build_public_workbook.py [--out excel/Offline_LLM_Hardware_Selector_public.xlsx]

It reads nothing but data/*.csv, so the workbook cannot carry anything that is not already in this repository.
The cell formulas are the ones in scripts/formulas.py and take their constants from that module.
"""
import argparse
import csv
import pathlib
import re

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import formulas as F
from common import DATA, NUMERIC, ROOT

DEFAULT_OUT = ROOT / "excel" / "Offline_LLM_Hardware_Selector_public.xlsx"
TABLES = [("Models", "models"), ("GPUs", "gpus"), ("CPUs", "cpus"), ("Quant", "quant"), ("Libraries", "libraries"),
          ("Voice Models", "voice_models"), ("Task Rules", "task_rules"), ("Arch Support", "arch_support"),
          ("Compute", "compute"), ("Reference Models", "reference_models"), ("Evidence", "evidence")]
TASK_BITS = ("Base min bits (4-30B)", "Small model <4B min bits", "Large model >=30B min bits")
NUMERIC_COLUMNS = NUMERIC | set(TASK_BITS)
PRIVATE_TEXT = re.compile(r"Intel Confidential|playbook|Gold deck|\bEDS \d+|917114|914397", re.I)
PRIVATE_ROW = re.compile(r"^(PTL|RZL)", re.I)

MH, HM, ML = "Model to Hardware", "Hardware to Models", "Multi-LLM"
GPU_EFF, CPU_EFF = "GPU Runtime Eff", "CPU Runtime Eff"
MH_IN = {"model": "B4", "quant": "B5", "context": "B6", "users": "B7", "kv": "B8", "runtime": "B9", "task": "B10", "prompt": "B11", "quantity": "B12", "headroom": "B13"}
MH_OUT = {"weights": "H4", "kv": "H5", "runtime": "H6", "total": "H7", "blocker": "E16"}
MH_GPU_FIRST = 19
HM_IN = {"gpu": "B4", "quantity": "B5", "headroom": "B6", "runtime": "B7", "quant": "B8", "context": "B9", "users": "B10", "kv": "B11", "task": "B12", "prompt": "B13"}
HM_FIRST = 19
ML_IN = {"gpu": "B4", "quantity": "B5", "headroom": "B6", "prompt": "B7", "util": "B8"}
ML_VOICE_ROWS = {"VAD": 12, "Diarization": 13, "ASR": 14}
ML_LLM_ROW = 18
ML_OUT = {"setMemoryGb": "B22", "setRawBwGbps": "B23", "verdict": "B24", "setsByMemory": "B25", "setsByBandwidth": "B26", "sets": "B27",
          "setsLimit": "B28", "tokAlone": "B29", "prefillAlone": "B30", "ttftAlone": "B31", "ttftWorstLoaded": "B32"}
DEFAULTS = {"model": "Qwen3 32B", "quant": "Q4_K_M / INT4-AWQ", "runtime": "llama.cpp", "task": "General chat / Q&A",
            "pipeline_gpu": "Intel CRI XT 32Xe 160GB 300W (CCG)", "hardware_gpu": "Intel Arc Pro B60",
            "voice": {"VAD": "Silero VAD", "Diarization": "pyannote community-1", "ASR": "Qwen3-ASR-1.7B"}}

TITLE = Font(bold=True, size=14)
BOLD = Font(bold=True)
HEAD = PatternFill("solid", fgColor="DDEBF7")
INPUT = PatternFill("solid", fgColor="FFF2CC")
RESULT = PatternFill("solid", fgColor="E2EFDA")
k = lambda value: f"{value:g}"
MARGIN, SHARE, CARD, RESERVE = k(F.WEIGHT_MARGIN), k(F.RUNTIME_SHARE), k(F.EXTRA_CARD_SCALE), k(F.HOST_RESERVE_GB)
MFU, CPU_FACTOR, GFLOPS, VOICE_BITS = k(F.PREFILL_MFU), k(F.CPU_PREFILL_FACTOR), k(F.PREFILL_GFLOPS_PER_B), k(F.VOICE_RESERVE_BITS)


def read_table(name):
    with open(DATA / f"{name}.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    body = []
    for row in rows[1:]:
        values = []
        for column, value in zip(header, row):
            if value == "":
                values.append(None)
            elif column in NUMERIC_COLUMNS:
                try:
                    number = float(value)
                    values.append(int(number) if number.is_integer() and column != "CUDA compute capability" else number)
                except ValueError:
                    values.append(value)
            else:
                values.append(ILLEGAL_CHARACTERS_RE.sub("", value)[:32000])
        body.append(values)
    return header, body


def write_rows(ws, header, body):
    ws.append(header)
    for cell in ws[1]:
        cell.font, cell.fill = BOLD, HEAD
    for r, values in enumerate(body, start=2):
        for c, value in enumerate(values, start=1):
            cell = ws.cell(r, c, value)
            if isinstance(value, str) and value.startswith("="):
                cell.data_type = "s"  # a quoted source value, not a formula
    ws.freeze_panes = "B2"
    for c, name in enumerate(header, start=1):
        ws.column_dimensions[get_column_letter(c)].width = min(48, max(12, len(str(name)) + 2))


class Book:
    """Column letters and absolute ranges of the data sheets."""

    def __init__(self):
        self.wb = Workbook()
        self.wb.remove(self.wb.active)
        self.tables = {}

    def add(self, sheet, header, body):
        self.tables[sheet] = (header, body)
        write_rows(self.wb.create_sheet(sheet), header, body)

    def dicts(self, sheet):
        header, body = self.tables[sheet]
        return [dict(zip(header, row)) for row in body]

    def count(self, sheet):
        return len(self.tables[sheet][1])

    def letter(self, sheet, column):
        return get_column_letter(self.tables[sheet][0].index(column) + 1)

    def rng(self, sheet, column):
        letter = self.letter(sheet, column)
        return f"'{sheet}'!${letter}$2:${letter}${self.count(sheet) + 1}"

    def at(self, sheet, column, row):
        return f"'{sheet}'!${self.letter(sheet, column)}${row}"

    def pick(self, sheet, column, row_expr):
        return f"INDEX({self.rng(sheet, column)},{row_expr})"

    def find(self, sheet, column, value_expr):
        return f"MATCH({value_expr},{self.rng(sheet, column)},0)"


def a(address):
    column, row = re.match(r"([A-Z]+)(\d+)", address).groups()
    return f"${column}${row}"


def escaped(expr):
    """A lookup value with Excel's wildcard escape character doubled (one model name contains ~)."""
    return 'SUBSTITUTE(' + expr + ',"~","~~")'


def put(ws, address, value, fill=None, bold=False, fmt=None):
    cell = ws[address]
    cell.value = value
    if fill:
        cell.fill = fill
    if bold:
        cell.font = BOLD
    if fmt:
        cell.number_format = fmt
    return cell


def dropdown(ws, address, source, blank=False):
    rule = DataValidation(type="list", formula1=source, allow_blank=blank)
    ws.add_data_validation(rule)
    rule.add(address)


def heading(ws, title, note):
    put(ws, "A1", title).font = TITLE
    put(ws, "A2", note)
    for column, width in zip("ABCDEFGHIJKLMNOPQRST", (34, 30, 14, 26, 16, 22, 30, 18, 14, 14, 14, 14, 30, 14, 20, 14, 14, 28, 10, 12)):
        ws.column_dimensions[column].width = width


def inputs(ws, rows, lists):
    put(ws, "A3", "Inputs (yellow cells)", bold=True)
    for address, label, value, source, fmt in rows:
        put(ws, "A" + address[1:], label)
        put(ws, address, value, fill=INPUT, fmt=fmt)
        if source:
            dropdown(ws, address, lists[source])


def table_header(ws, row, names):
    for c, name in enumerate(names, start=1):
        cell = ws.cell(row, c, name)
        cell.font, cell.fill = BOLD, HEAD
        cell.alignment = Alignment(wrap_text=True, vertical="top")


def model_to_hardware(book, lists):
    ws = book.wb.create_sheet(MH, 0)
    i = {name: a(address) for name, address in MH_IN.items()}
    heading(ws, "Model → Hardware", "Pick a model and workload. Every accelerator and CPU in the tables is checked for fit, decode speed and time to first token.")
    inputs(ws, [(MH_IN["model"], "Model", DEFAULTS["model"], "models", None), (MH_IN["quant"], "Quantization", DEFAULTS["quant"], "quants", None),
                (MH_IN["context"], "Context tokens", 8192, "contexts", "#,##0"), (MH_IN["users"], "Concurrent users", 1, "users", None),
                (MH_IN["kv"], "KV cache bits", 8, "kv", None), (MH_IN["runtime"], "Runtime", DEFAULTS["runtime"], "runtimes", None),
                (MH_IN["task"], "Task", DEFAULTS["task"], "tasks", None), (MH_IN["prompt"], "Prompt tokens", 2048, "prompts", "#,##0"),
                (MH_IN["quantity"], "Accelerator quantity", 1, "quantities", None), (MH_IN["headroom"], "GPU memory headroom", F.DEFAULT_HEADROOM_PCT / 100, "headrooms", "0%")], lists)
    row, task = "$E$15", "$E$17"
    model = lambda column: book.pick("Models", column, row)
    put(ws, "D3", "Selected model", bold=True)
    derived = [("E4", "Bits per weight", f"={book.pick('Quant', 'Bits/weight (eff.)', book.find('Quant', 'Quantization', i['quant']))}"),
               ("E5", "Total parameters (B)", f"={model('Total params (B)')}"), ("E6", "Active parameters (B)", f"={model('Active params (B)')}"),
               ("E7", "Layers", f"={model('Layers')}"), ("E8", "KV heads", f"={model('KV heads')}"), ("E9", "Head dimension", f"={model('Head dim')}"),
               ("E10", "Max context", f"={model('Max context')}"), ("E11", "Type", f"={model('Type')}"), ("E12", "Runtime architecture", f"={model('Runtime arch')}"),
               ("E13", "Runtime loads it (Y / P / N)", "=" + arch_flag(book, "$E$12", i["runtime"])),
               ("E14", "Minimum bits for the task", f"=IF($E$5<4,{book.pick('Task Rules', TASK_BITS[1], task)},IF($E$5>=30,{book.pick('Task Rules', TASK_BITS[2], task)},{book.pick('Task Rules', TASK_BITS[0], task)}))"),
               ("E15", "Row in Models", "=" + book.find("Models", "Model", escaped(i["model"]))),
               ("E16", "Model or task blocker", f"=IF($E$13=\"N\",\"Runtime cannot load model\",IF(AND(UPPER({book.pick('Task Rules', 'VLM required?', task)})=\"Y\",NOT({is_vlm('$E$11')})),\"Needs a vision model\",IF($E$4<$E$14,\"Below task quant floor\",\"\")))"),
               ("E17", "Row in Task Rules", f"={book.find('Task Rules', 'Task', i['task'])}")]
    for address, label, formula in derived:
        put(ws, "D" + address[1:], label)
        put(ws, address, formula)
    gpus, cpus = book.count("GPUs"), book.count("CPUs")
    g0, g1 = MH_GPU_FIRST, MH_GPU_FIRST + gpus - 1
    c0 = g1 + 4
    c1 = c0 + cpus - 1
    put(ws, "G3", "Result", bold=True)
    results = [("H4", "Weights (GB)", f"=$E$5*$E$4/8*{MARGIN}", "0.00"), ("H5", "KV cache (GB)", f"=2*$E$7*$E$8*$E$9*{i['kv']}/8/1E9*{i['context']}*{i['users']}", "0.00"),
               ("H6", "Runtime overhead (GB)", f"=$H$4*{SHARE}", "0.00"), ("H7", "Memory needed (GB)", "=$H$4+$H$5+$H$6", "0.00"),
               ("H8", "GB read per token", "=$E$6*$E$4/8", "0.00"), ("H9", "Card scale", f"=1+{CARD}*({i['quantity']}-1)", "0.0"),
               ("H10", "Accelerators that fit", f"=COUNTIF($G${g0}:$G${g1},\"Fits\")", "0"), ("H11", "CPUs that fit", f"=COUNTIF($G${c0}:$G${c1},\"Fits\")", "0"),
               ("H12", "Fastest first token, accelerators that fit (s)", f"=IF(COUNT($L${g0}:$L${g1})=0,\"\",MIN($L${g0}:$L${g1}))", "0.00"),
               ("H13", "On", f"=IF($H$12=\"\",\"\",INDEX($A${g0}:$A${g1},MATCH($H$12,$L${g0}:$L${g1},0)))", None)]
    for address, label, formula, fmt in results:
        put(ws, "G" + address[1:], label)
        put(ws, address, formula, fill=RESULT, fmt=fmt)

    names = ["Memory (GB)", "Bandwidth (GB/s)", "Usable memory (GB)", "Runtime efficiency", "Result", "Decode tok/s", "Compute (TFLOPS est.)", "Prefill tok/s",
             "Time to first token (s)", "First token if it fits (s)", "Evidence status"]
    put(ws, f"A{g0 - 2}", "Accelerators", bold=True)
    table_header(ws, g0 - 1, ["Accelerator", "Vendor"] + names)
    last = get_column_letter(book.count("Libraries") + 1)
    for n in range(gpus):
        r, s = g0 + n, n + 2
        hardware_row(ws, r, [f"={book.at('GPUs', 'Accelerator', s)}", f"={book.at('GPUs', 'Vendor', s)}", f"={book.at('GPUs', 'Memory (GB)', s)}", f"={book.at('GPUs', 'Bandwidth (GB/s)', s)}",
                             f"=$C{r}*(1-{i['headroom']})*{i['quantity']}", efficiency(GPU_EFF, s, last, i["runtime"]),
                             fit(r, i["runtime"]), f"=IF($F{r}=0,0,$F{r}*$D{r}*$H$9/$H$8/{i['users']})", compute(book, "GPUs", f"$A{r}"),
                             f"={MFU}*$I{r}*$H$9*1000/({GFLOPS}*$E$6)/{i['users']}", ttft(r, i["prompt"]), f"=IF(AND($G{r}=\"Fits\",$K{r}<>\"\"),$K{r},\"\")",
                             f"={book.at('GPUs', 'Evidence status', s)}"])
    put(ws, f"A{c0 - 2}", "CPUs and SoCs (model held in system memory)", bold=True)
    table_header(ws, c0 - 1, ["CPU / platform", "Class", "Max RAM (GB)", "Memory BW (GB/s)"] + names[2:])
    for n in range(cpus):
        r, s = c0 + n, n + 2
        hardware_row(ws, r, [f"={book.at('CPUs', 'CPU / platform', s)}", f"={book.at('CPUs', 'Class', s)}", f"={book.at('CPUs', 'Max RAM (GB)', s)}", f"={book.at('CPUs', 'Memory BW (GB/s)', s)}",
                             f"=$C{r}-MIN({RESERVE},$C{r}*0.25)", efficiency(CPU_EFF, s, last, i["runtime"]),
                             fit(r, i["runtime"]), f"=IF($F{r}=0,0,$F{r}*$D{r}/$H$8/{i['users']})", compute(book, "CPUs", f"$A{r}"),
                             f"={MFU}*$I{r}*{CPU_FACTOR}*1000/({GFLOPS}*$E$6)/{i['users']}", ttft(r, i["prompt"]), f"=IF(AND($G{r}=\"Fits\",$K{r}<>\"\"),$K{r},\"\")",
                             f"={book.at('CPUs', 'Evidence status', s)}"])
    ws.freeze_panes = "B4"
    return {"gpu_first": g0, "cpu_first": c0}


def hardware_row(ws, r, formulas):
    for c, formula in enumerate(formulas, start=1):
        cell = ws.cell(r, c, formula)
        if c in (3, 4, 5, 8, 9, 10, 11, 12):
            cell.number_format = "0.00"


def arch_flag(book, arch_expr, runtime_expr):
    header, _ = book.tables["Arch Support"]
    first, last = get_column_letter(1), get_column_letter(len(header))
    rows = book.count("Arch Support") + 1
    return (f"IFERROR(INDEX('Arch Support'!${first}$2:${last}${rows},{book.find('Arch Support', 'Arch key', arch_expr)},"
            f"MATCH({runtime_expr},'Arch Support'!${first}$1:${last}$1,0)),\"\")")


def is_vlm(type_expr):
    return f"OR(ISNUMBER(SEARCH(\"VLM\",{type_expr})),ISNUMBER(SEARCH(\"Vision\",{type_expr})))"


def efficiency(sheet, source_row, last, runtime_expr):
    return f"=IFERROR(INDEX('{sheet}'!$B${source_row}:${last}${source_row},MATCH({runtime_expr},'{sheet}'!$B$1:${last}$1,0)),0)"


def fit(r, runtime_expr):
    return f"=IF($F{r}=0,{runtime_expr}&\" not supported\",IF($E$16<>\"\",$E$16,IF($E{r}>=$H$7,\"Fits\",\"Memory shortfall\")))"


def compute(book, kind, name_expr):
    return f"=SUMIFS({book.rng('Compute', 'TFLOPS (est.)')},{book.rng('Compute', 'Sheet')},\"{kind}\",{book.rng('Compute', 'Item')},{name_expr})"


def ttft(r, prompt_expr):
    return f"=IF(OR($J{r}=0,$H{r}=0),\"\",{prompt_expr}/$J{r}+1/$H{r})"


def hardware_to_models(book, lists):
    ws = book.wb.create_sheet(HM, 1)
    i = {name: a(address) for name, address in HM_IN.items()}
    gpu = DEFAULTS["hardware_gpu"] if any(row["Accelerator"] == DEFAULTS["hardware_gpu"] for row in book.dicts("GPUs")) else book.dicts("GPUs")[0]["Accelerator"]
    heading(ws, "Hardware → Models", "Pick an accelerator and workload. Every model in the catalog is checked for fit, decode speed and time to first token.")
    inputs(ws, [(HM_IN["gpu"], "Accelerator", gpu, "gpus", None), (HM_IN["quantity"], "Accelerator quantity", 1, "quantities", None),
                (HM_IN["headroom"], "GPU memory headroom", F.DEFAULT_HEADROOM_PCT / 100, "headrooms", "0%"), (HM_IN["runtime"], "Runtime", DEFAULTS["runtime"], "runtimes", None),
                (HM_IN["quant"], "Quantization", DEFAULTS["quant"], "quants", None), (HM_IN["context"], "Context tokens", 8192, "contexts", "#,##0"),
                (HM_IN["users"], "Concurrent users", 1, "users", None), (HM_IN["kv"], "KV cache bits", 8, "kv", None),
                (HM_IN["task"], "Task", DEFAULTS["task"], "tasks", None), (HM_IN["prompt"], "Prompt tokens", 2048, "prompts", "#,##0")], lists)
    row, task = "$E$15", "$E$16"
    last = get_column_letter(book.count("Libraries") + 1)
    put(ws, "D3", "Selected accelerator", bold=True)
    derived = [("E4", "Memory per card (GB)", f"={book.pick('GPUs', 'Memory (GB)', row)}"), ("E5", "Bandwidth per card (GB/s)", f"={book.pick('GPUs', 'Bandwidth (GB/s)', row)}"),
               ("E6", "Usable memory (GB)", f"=$E$4*(1-{i['headroom']})*{i['quantity']}"),
               ("E7", "Runtime efficiency", f"=IFERROR(INDEX('{GPU_EFF}'!$B$2:${last}${book.count('GPUs') + 1},{row},MATCH({i['runtime']},'{GPU_EFF}'!$B$1:${last}$1,0)),0)"),
               ("E8", "Compute (TFLOPS est.)", compute(book, "GPUs", i["gpu"])),
               ("E9", "Bits per weight", f"={book.pick('Quant', 'Bits/weight (eff.)', book.find('Quant', 'Quantization', i['quant']))}"),
               ("E10", "Card scale", f"=1+{CARD}*({i['quantity']}-1)"), ("E11", "Task needs a vision model", f"=UPPER({book.pick('Task Rules', 'VLM required?', task)})"),
               ("E12", "Minimum bits, 4-30B models", f"={book.pick('Task Rules', TASK_BITS[0], task)}"), ("E13", "Minimum bits, models under 4B", f"={book.pick('Task Rules', TASK_BITS[1], task)}"),
               ("E14", "Minimum bits, models of 30B or more", f"={book.pick('Task Rules', TASK_BITS[2], task)}"), ("E15", "Row in GPUs", f"={book.find('GPUs', 'Accelerator', i['gpu'])}"),
               ("E16", "Row in Task Rules", f"={book.find('Task Rules', 'Task', i['task'])}")]
    for address, label, formula in derived:
        put(ws, "D" + address[1:], label)
        put(ws, address, formula)
    models = book.count("Models")
    m0, m1 = HM_FIRST, HM_FIRST + models - 1
    put(ws, "G3", "Result", bold=True)
    for address, label, formula, fmt in [("H4", "Models that fit", f"=COUNTIF($F${m0}:$F${m1},\"Fits\")", "0"),
                                         ("H5", "Fastest first token, models that fit (s)", f"=IF(COUNT($J${m0}:$J${m1})=0,\"\",MIN($J${m0}:$J${m1}))", "0.00"),
                                         ("H6", "On", f"=IF($H$5=\"\",\"\",INDEX($A${m0}:$A${m1},MATCH($H$5,$J${m0}:$J${m1},0)))", None)]:
        put(ws, "G" + address[1:], label)
        put(ws, address, formula, fill=RESULT, fmt=fmt)
    put(ws, f"A{m0 - 2}", "Models", bold=True)
    table_header(ws, m0 - 1, ["Model", "Type", "Total parameters (B)", "Active parameters (B)", "Memory needed (GB)", "Result", "Decode tok/s", "Prefill tok/s",
                              "Time to first token (s)", "First token if it fits (s)", "Runtime loads it (Y / P / N)"])
    for n in range(models):
        r, s = m0 + n, n + 2
        kv = f"2*{book.at('Models', 'Layers', s)}*{book.at('Models', 'KV heads', s)}*{book.at('Models', 'Head dim', s)}*{i['kv']}/8/1E9*{i['context']}*{i['users']}"
        floor = f"IF($C{r}<4,$E$13,IF($C{r}>=30,$E$14,$E$12))"
        formulas = [f"={book.at('Models', 'Model', s)}", f"={book.at('Models', 'Type', s)}", f"={book.at('Models', 'Total params (B)', s)}", f"={book.at('Models', 'Active params (B)', s)}",
                    f"=$C{r}*$E$9/8*{MARGIN}*(1+{SHARE})+{kv}",
                    f"=IF($E$7=0,{i['runtime']}&\" not supported\",IF($K{r}=\"N\",\"Runtime cannot load model\",IF(AND($E$11=\"Y\",NOT({is_vlm(f'$B{r}')})),\"Needs a vision model\","
                    f"IF($E$9<{floor},\"Below task quant floor\",IF($E$6>=$E{r},\"Fits\",\"Memory shortfall\")))))",
                    f"=IF($E$7=0,0,$E$7*$E$5*$E$10/($D{r}*$E$9/8)/{i['users']})", f"={MFU}*$E$8*$E$10*1000/({GFLOPS}*$D{r})/{i['users']}",
                    f"=IF(OR($H{r}=0,$G{r}=0),\"\",{i['prompt']}/$H{r}+1/$G{r})", f"=IF(AND($F{r}=\"Fits\",$I{r}<>\"\"),$I{r},\"\")",
                    "=" + arch_flag(book, book.at("Models", "Runtime arch", s), i["runtime"])]
        for c, formula in enumerate(formulas, start=1):
            cell = ws.cell(r, c, formula)
            if c in (3, 4, 5, 7, 8, 9, 10):
                cell.number_format = "0.00"
    ws.freeze_panes = "B4"


def multi_llm(book, lists):
    ws = book.wb.create_sheet(ML, 2)
    i = {name: a(address) for name, address in ML_IN.items()}
    heading(ws, "Multi-LLM", "Stage 1 VAD, Stage 2 diarization, Stage 3 speech recognition and Stage 4 one LLM on one accelerator. Stage 5 counts the complete sets that run at once.")
    inputs(ws, [(ML_IN["gpu"], "Accelerator", DEFAULTS["pipeline_gpu"], "gpus", None), (ML_IN["quantity"], "Accelerator quantity", 1, "quantities", None),
                (ML_IN["headroom"], "GPU memory headroom", 0.1, "headrooms", "0%"), (ML_IN["prompt"], "Prompt tokens", 2048, "prompts", "#,##0"),
                (ML_IN["util"], "Bandwidth utilization (Auto = each stage's runtime)", "Auto", "utils", None)], lists)
    gpu_row = "$E$9"
    last = get_column_letter(book.count("Libraries") + 1)
    stage_use = lambda runtime: f"IF({i['util']}=\"Auto\",IFERROR(INDEX('{GPU_EFF}'!$B$2:${last}${book.count('GPUs') + 1},{gpu_row},MATCH({runtime},'{GPU_EFF}'!$B$1:${last}$1,0)),0),{i['util']})"
    put(ws, "D3", "Selected accelerator", bold=True)
    for address, label, formula in [("E4", "Memory per card (GB)", f"={book.pick('GPUs', 'Memory (GB)', gpu_row)}"), ("E5", "Bandwidth per card (GB/s)", f"={book.pick('GPUs', 'Bandwidth (GB/s)', gpu_row)}"),
                                    ("E6", "Usable memory (GB)", f"=$E$4*{i['quantity']}*(1-{i['headroom']})"), ("E7", "Accelerator bandwidth (GB/s)", f"=$E$5*(1+{CARD}*({i['quantity']}-1))"),
                                    ("E8", "Compute (TFLOPS est.)", compute(book, "GPUs", i["gpu"])), ("E9", "Row in GPUs", f"={book.find('GPUs', 'Accelerator', i['gpu'])}")]:
        put(ws, "D" + address[1:], label)
        put(ws, address, formula)

    put(ws, "A10", "Stage 1 to 3: voice", bold=True)
    table_header(ws, 11, ["Stage", "Model (blank = not used)", "Runs on", "Runtime", "KV cache bits", "Measured weights (GB, optional)", "Activation reserve (GB at FP16)", "Bandwidth (GB/s)",
                          "Catalog weights (GB)", "On the accelerator", "Resident weights (GB)", "Reserve counted (GB)", "Runtime efficiency", "Bandwidth at accelerator (GB/s)", "Status"])
    params, bits = book.rng("Voice Models", "Total params (B)"), book.rng("Voice Models", "Bits / weight")
    for number, (stage, r) in enumerate(ML_VOICE_ROWS.items(), start=1):
        where = f"{book.rng('Voice Models', 'Stage')},\"{stage}\",{book.rng('Voice Models', 'Model')},$B{r}"
        put(ws, f"A{r}", f"{number} {stage}")
        for column, value, source in (("B", DEFAULTS["voice"][stage], f"voice_{stage}"), ("C", "GPU", "placements"), ("D", DEFAULTS["runtime"], "runtimes"), ("E", F.VOICE_RESERVE_BITS, "voice_kv"),
                                      ("F", None, None), ("G", F.VOICE_EXTRA_GB, None), ("H", F.VOICE_BW_GBPS, None)):
            put(ws, f"{column}{r}", value, fill=INPUT)
            if source:
                dropdown(ws, f"{column}{r}", lists[source], blank=True)
        put(ws, f"I{r}", f"=IF($B{r}=\"\",\"\",IF(AND(SUMIFS({params},{where})>0,SUMIFS({bits},{where})>0),SUMIFS({params},{where})*SUMIFS({bits},{where})/8,\"\"))", fmt="0.000")
        put(ws, f"J{r}", f"=AND($B{r}<>\"\",$C{r}=\"GPU\")")
        put(ws, f"K{r}", f"=IF(NOT($J{r}),0,IF(ISNUMBER($F{r}),$F{r},IF(ISNUMBER($I{r}),$I{r},\"\")))", fmt="0.000")
        put(ws, f"L{r}", f"=IF($J{r},$G{r}*$E{r}/{VOICE_BITS},0)", fmt="0.000")
        put(ws, f"M{r}", f"=IF(NOT($J{r}),0,{stage_use(f'$D{r}')})", fmt="0.00")
        put(ws, f"N{r}", f"=IF(AND($J{r},$M{r}>0),$H{r}/$M{r},0)", fmt="0.00")
        put(ws, f"O{r}", f"=IF($B{r}=\"\",\"NOT USED\",IF($C{r}<>\"GPU\",\"CPU - not sized\",IF($M{r}<=0,\"REVIEW: RUNTIME\",IF($K{r}=\"\",\"WEIGHTS PENDING\",\"INPUTS SET\"))))")

    r = ML_LLM_ROW
    put(ws, f"A{r - 2}", "Stage 4: LLM", bold=True)
    table_header(ws, r - 1, ["Stage", "Model (blank = not used)", "Quantization", "Runtime", "KV cache bits", "Context tokens", "Target tok/s", "Bits per weight", "Total parameters (B)",
                             "Active parameters (B)", "KV cache per token (GB)", "Weights (GB)", "KV cache and overhead (GB)", "Total (GB)", "Bandwidth needed (GB/s)", "Runtime efficiency",
                             "Bandwidth at accelerator (GB/s)", "Status", "Row in Models", "Max context"])
    put(ws, f"A{r}", "4 LLM")
    for column, value, source in (("B", DEFAULTS["model"], "models"), ("C", DEFAULTS["quant"], "quants"), ("D", DEFAULTS["runtime"], "runtimes"), ("E", 16, "kv"), ("F", 8192, "contexts"), ("G", 20, None)):
        put(ws, f"{column}{r}", value, fill=INPUT)
        if source:
            dropdown(ws, f"{column}{r}", lists[source], blank=True)
    model = lambda column: f"IF($B{r}=\"\",0,{book.pick('Models', column, f'$S{r}')})"
    put(ws, f"H{r}", f"=IF($B{r}=\"\",0,{book.pick('Quant', 'Bits/weight (eff.)', book.find('Quant', 'Quantization', f'$C{r}'))})")
    put(ws, f"I{r}", "=" + model("Total params (B)"))
    put(ws, f"J{r}", "=" + model("Active params (B)"))
    put(ws, f"K{r}", f"=2*{model('Layers')}*{model('KV heads')}*{model('Head dim')}*$E{r}/8/1E9")
    put(ws, f"L{r}", f"=$I{r}*$H{r}/8*{MARGIN}", fmt="0.000")
    put(ws, f"M{r}", f"=$K{r}*$F{r}+$L{r}*{SHARE}", fmt="0.000")
    put(ws, f"N{r}", f"=$L{r}+$M{r}", fmt="0.000")
    put(ws, f"O{r}", f"=$J{r}*$H{r}/8*{MARGIN}*$G{r}", fmt="0.00")
    put(ws, f"P{r}", f"=IF($B{r}=\"\",0,{stage_use(f'$D{r}')})", fmt="0.00")
    put(ws, f"Q{r}", f"=IF($P{r}>0,$O{r}/$P{r},0)", fmt="0.00")
    put(ws, f"R{r}", f"=IF($B{r}=\"\",\"NOT USED\",IF(OR($H{r}<=0,$F{r}<=0,$G{r}<=0),\"REVIEW: CHECK INPUTS\",IF(AND($T{r}>0,$F{r}>$T{r}),\"REVIEW: CONTEXT > MODEL LIMIT\","
                     f"IF($P{r}<=0,\"REVIEW: RUNTIME\",IF($B$22>$E$6,\"NO - GPU MEMORY\",IF($B$23>$E$7,\"NO - BANDWIDTH\",\"FITS\"))))))")
    put(ws, f"S{r}", f"=IF($B{r}=\"\",\"\"," + book.find("Models", "Model", escaped(f"$B{r}")) + ")")
    put(ws, f"T{r}", "=" + model("Max context"))

    first, final = min(ML_VOICE_ROWS.values()), max(ML_VOICE_ROWS.values())
    review = f"LEFT($R${r},6)=\"REVIEW\""
    put(ws, "A21", "Stage 5: concurrent sets", bold=True)
    stage5 = [("B22", "Memory per set (GB)", f"=SUM($K${first}:$K${final})+SUM($L${first}:$L${final})+$N${r}", "0.00"),
              ("B23", "Bandwidth per set at the accelerator (GB/s)", f"=SUM($N${first}:$N${final})+$Q${r}", "0.00"),
              ("B24", "Input status", f"=IF(OR($E$6<=0,$E$7<=0),\"REVIEW: select accelerator, quantity and headroom\",IF(COUNTIF($O${first}:$O${final},\"REVIEW*\")+IF($R${r}=\"REVIEW: RUNTIME\",1,0)>0,"
                                      f"\"REVIEW: pick a runtime that runs on this accelerator\",IF({review},\"REVIEW: fix LLM inputs\",\"INPUTS SET\")))", None),
              ("B25", "Sets by memory", "=IF(AND($B$24=\"INPUTS SET\",$B$22>0),INT($E$6/$B$22+1E-9),\"\")", "0"),
              ("B26", "Sets by bandwidth", "=IF(AND($B$24=\"INPUTS SET\",$B$22>0,$B$23>0),INT($E$7/$B$23+1E-9),\"\")", "0"),
              ("B27", "Sets of Stage 1 to 4 that run concurrently", "=IF($B$25=\"\",\"\",IF($B$26=\"\",$B$25,MIN($B$25,$B$26)))", "0"),
              ("B28", "Limit", "=IF($B$27=\"\",\"\",IF($B$26=\"\",\"memory\",IF($B$25<$B$26,\"memory\",IF($B$26<$B$25,\"bandwidth\",\"memory and bandwidth\"))))", None),
              ("B29", "LLM decode tok/s, one set alone", f"=IF(OR($B${r}=\"\",{review}),\"\",MAX(0,$E$7-SUM($N${first}:$N${final}))*$P${r}/($J${r}*$H${r}/8*{MARGIN}))", "0.00"),
              ("B30", "LLM prefill tok/s, one set alone", f"=IF(OR($B${r}=\"\",{review},$E$8<=0),\"\",{MFU}*$E$8*(1+{CARD}*({i['quantity']}-1))*1000/({GFLOPS}*$J${r}))", "0.00"),
              ("B31", "Time to first token, one set alone (s)", f"=IF(OR(N($B$29)=0,N($B$30)=0),\"\",{i['prompt']}/$B$30+1/$B$29)", "0.00"),
              ("B32", "Time to first token, every set busy (s)", f"=IF(OR(N($B$27)=0,N($B$30)=0,$G${r}<=0),\"\",{i['prompt']}/($B$30/$B$27)+1/$G${r})", "0.00")]
    for address, label, formula, fmt in stage5:
        put(ws, "A" + address[1:], label)
        put(ws, address, formula, fill=RESULT, fmt=fmt)
    ws["B27"].font = TITLE
    ws.freeze_panes = "B4"


def lists_sheet(book):
    ws = book.wb.create_sheet("Lists")
    voice = book.dicts("Voice Models")
    columns = {"voice_VAD": [row["Model"] for row in voice if row["Stage"] == "VAD"], "voice_Diarization": [row["Model"] for row in voice if row["Stage"] == "Diarization"],
               "voice_ASR": [row["Model"] for row in voice if row["Stage"] == "ASR"], "contexts": [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144],
               "headrooms": [0, 0.05, 0.08, 0.1, 0.15, 0.2], "users": [1, 2, 4, 8, 16, 32], "quantities": [1, 2, 4, 8], "prompts": [128, 512, 1024, 2048, 4096, 8192, 16384, 32768],
               "utils": ["Auto", 0.4, 0.5, 0.55, 0.6, 0.7, 0.8], "kv": [4, 8, 16], "voice_kv": [4, 8, 16, 32], "placements": ["GPU", "CPU"]}
    lists = {}
    for c, (name, values) in enumerate(columns.items(), start=1):
        letter = get_column_letter(c)
        ws.cell(1, c, name).font = BOLD
        for r, value in enumerate(values, start=2):
            ws.cell(r, c, value)
        lists[name] = f"'Lists'!${letter}$2:${letter}${len(values) + 1}"
    for name, sheet, column in (("models", "Models", "Model"), ("gpus", "GPUs", "Accelerator"), ("quants", "Quant", "Quantization"), ("runtimes", "Libraries", "Library"), ("tasks", "Task Rules", "Task")):
        lists[name] = book.rng(sheet, column)
    return lists


def efficiency_sheets(book):
    table = F.build_efficiency_table(book.dicts("Libraries"))
    runtimes = [row["Library"] for row in book.dicts("Libraries")]
    gpus, cpus = book.dicts("GPUs"), book.dicts("CPUs")
    write_rows(book.wb.create_sheet(GPU_EFF), ["Accelerator"] + runtimes,
               [[gpu["Accelerator"]] + [F.gpu_runtime_efficiency(table, name, gpu)["efficiency"] for name in runtimes] for gpu in gpus])
    write_rows(book.wb.create_sheet(CPU_EFF), ["CPU / platform"] + runtimes,
               [[cpu["CPU / platform"]] + [F.cpu_runtime_efficiency(table, name, cpu) for name in runtimes] for cpu in cpus])


def read_me(book):
    ws = book.wb.create_sheet("Read me", 0)
    ws.column_dimensions["A"].width = 150
    lines = [("Offline LLM Hardware Selector: public Excel edition", TITLE),
             ("Built by scripts/build_public_workbook.py from the tables in data/ of https://github.com/chinseang88/offline-llm-hardware-selector. It contains nothing that is not in that repository.", None),
             ("Live site: https://chinseang88.github.io/offline-llm-hardware-selector/", None), ("", None),
             ("Sheets", BOLD),
             ("Model to Hardware: pick a model and workload, see which accelerators and CPUs fit, the decode speed and the time to first token.", None),
             ("Hardware to Models: pick an accelerator and workload, see which models fit.", None),
             ("Multi-LLM: Stage 1 VAD, Stage 2 diarization, Stage 3 speech recognition, Stage 4 one LLM; Stage 5 counts the sets that run concurrently. Every stage has its own runtime and KV cache bits.", None),
             ("Models, GPUs, CPUs, Quant, Libraries, Voice Models, Task Rules, Arch Support, Compute, Reference Models, Evidence: the published data tables.", None),
             ("GPU Runtime Eff, CPU Runtime Eff: bandwidth efficiency of every runtime on every row, worked out from Libraries by the build script. Rebuild the workbook after changing Libraries, GPUs or CPUs.", None),
             ("", None), ("How to use", BOLD),
             ("Change the yellow cells only; they have drop-down lists. Green cells are results. Filter a table with Data > Filter; do not sort it, because each row reads its own row of the data sheet.", None),
             ("", None), ("Formulas", BOLD),
             (f"Weights GB = total parameters (B) x bits per weight / 8 x {MARGIN}. KV cache GB = 2 x layers x KV heads x head dimension x KV bits / 8 x context x users / 1e9. Runtime overhead = {SHARE} x weights.", None),
             (f"Decode tok/s = runtime efficiency x bandwidth x card scale / (active parameters x bits / 8) / users. Card scale = 1 + {CARD} x (cards - 1).", None),
             (f"Prefill tok/s = {MFU} x TFLOPS x 1000 / ({GFLOPS} x active parameters) x card scale / users; a CPU is counted at {CPU_FACTOR} of it. Time to first token = prompt tokens / prefill tok/s + 1 / decode tok/s.", None),
             (f"Usable memory: accelerator = memory x (1 - headroom) x cards; CPU = RAM - min({RESERVE} GB, 25% of RAM).", None),
             ("Concurrent sets = min(usable memory / memory per set, accelerator bandwidth / sum of (stage bandwidth / stage runtime efficiency)).", None),
             (f"A voice stage has no KV geometry in the catalog: its activation reserve is entered at FP16 and counted as reserve x KV bits / {VOICE_BITS}. This is a planning assumption.", None),
             ("", None), ("Limits", BOLD),
             ("These are planning estimates, not benchmarks. Compute (TFLOPS) figures are estimates with no vendor source and are used only for time to first token.", None),
             ("Rows marked Pre-release planning are not confirmed by a public vendor source; their compute figure is derived from published columns (see the Compute sheet).", None),
             ("Not in this edition, compared with the site: CPU offload, per-device memory overrides, the Intel-only switch and the evidence drawer.", None),
             ("Prices are not tracked.", None)]
    for r, (text, font) in enumerate(lines, start=1):
        cell = ws.cell(r, 1, text)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if font:
            cell.font = font


def scan(path):
    """Fail if the saved workbook carries a marker of non-public material."""
    wb = load_workbook(path)
    problems = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and PRIVATE_TEXT.search(cell.value):
                    problems.append(f"{ws.title}!{cell.coordinate}: {cell.value[:60]}")
    for sheet in ("GPUs", "CPUs", "Compute"):
        column = 2 if sheet == "Compute" else 1
        problems += [f"{sheet} row {cell.row}: {cell.value}" for (cell,) in wb[sheet].iter_rows(min_row=2, min_col=column, max_col=column) if PRIVATE_ROW.search(str(cell.value or ""))]
    hidden = [ws.title for ws in wb.worksheets if ws.sheet_state != "visible"]
    if hidden:
        problems.append(f"hidden sheets: {hidden}")
    return problems


def build(out_path):
    book = Book()
    for sheet, name in TABLES:
        book.add(sheet, *read_table(name))
    keys = [row[key] for sheet, key in (("GPUs", "Accelerator"), ("CPUs", "CPU / platform"), ("Libraries", "Library"), ("Task Rules", "Task"), ("Quant", "Quantization"),
                                         ("Voice Models", "Model"), ("Arch Support", "Arch key")) for row in book.dicts(sheet)]
    wild = [key for key in keys if re.search(r"[*?~]", str(key))]
    if wild:
        raise SystemExit(f"lookup keys with Excel wildcard characters need SUBSTITUTE handling: {wild}")
    efficiency_sheets(book)
    lists = lists_sheet(book)
    model_to_hardware(book, lists)
    hardware_to_models(book, lists)
    multi_llm(book, lists)
    read_me(book)
    book.wb.properties.creator = book.wb.properties.lastModifiedBy = "Offline LLM Hardware Selector"
    book.wb.properties.title = "Offline LLM Hardware Selector (public edition)"
    book.wb.calculation.fullCalcOnLoad = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    book.wb.save(out_path)
    return book


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    out_path = pathlib.Path(parser.parse_args().out)
    book = build(out_path)
    problems = scan(out_path)
    if problems:
        out_path.unlink()
        raise SystemExit("not written, non-public markers found:\n  " + "\n  ".join(problems[:20]))
    counts = ", ".join(f"{book.count(sheet)} {sheet}" for sheet, _ in TABLES)
    print(f"{out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}: {counts}")


if __name__ == "__main__":
    main()
