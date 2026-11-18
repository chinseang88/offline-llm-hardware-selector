# Applies out/apply_plan.json to the OPEN selector workbook via Excel COM (running instance, Windows only).
# Usage: .\apply_to_excel.ps1 [-Workbook Offline_LLM_Hardware_Selector_v27.xlsx] [-Plan ..\..\out\apply_plan.json] [-SkipOps]
param([string]$Workbook = 'Offline_LLM_Hardware_Selector_v27.xlsx', [string]$Plan = (Join-Path $PSScriptRoot '..\..\out\apply_plan.json'), [switch]$SkipOps)
$ErrorActionPreference = 'Stop'
if (-not ('RunningCom' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class RunningCom {
    [DllImport("oleaut32.dll", PreserveSig=false)] static extern void GetActiveObject(ref Guid rclsid, IntPtr pvReserved, [MarshalAs(UnmanagedType.IUnknown)] out object ppunk);
    [DllImport("ole32.dll")] static extern int CLSIDFromProgID([MarshalAs(UnmanagedType.LPWStr)] string lpszProgID, out Guid pclsid);
    public static object Get(string progId) { Guid g; CLSIDFromProgID(progId, out g); object o; GetActiveObject(ref g, IntPtr.Zero, out o); return o; }
}
'@
}
$app = [RunningCom]::Get('Excel.Application')
$book = $null
foreach ($b in $app.Workbooks) { if ($b.Name -eq $Workbook) { $book = $b } }
if (-not $book) { throw "Workbook $Workbook is not open in Excel" }
$plan = Get-Content $Plan -Raw -Encoding UTF8 | ConvertFrom-Json

function ToExcelColor([int]$rgb) { $r = ($rgb -shr 16) -band 0xFF; $g = ($rgb -shr 8) -band 0xFF; $b = $rgb -band 0xFF; return $r + ($g * 256) + ($b * 65536) }

$app.ScreenUpdating = $false
$app.EnableEvents = $false
$prevCalc = $app.Calculation
$app.Calculation = -4135   # xlCalculationManual
$app.DisplayAlerts = $false
$sheets = @{}
foreach ($ws in $book.Worksheets) { $sheets[$ws.Name] = $ws }

$n = 0; $corr = 0
$opsToRun = if ($SkipOps) { @() } else { $plan.ops }
foreach ($op in $opsToRun) {
    $ws = $sheets[$op.sheet]
    $rng = $ws.Range($op.cell)
    if ($null -ne $op.new) {
        if ($rng.HasFormula) { continue }
        if ($op.new -is [string]) { $rng.Value2 = [string]$op.new } else { $rng.Value2 = [double]$op.new }
        $corr++
    }
    if ($null -ne $op.fill) { $rng.Interior.Color = ToExcelColor([int]$op.fill) }
    if ($null -ne $op.comment -and $op.comment -ne '') {
        if ($null -ne $rng.Comment) { $rng.Comment.Delete() }
        $c = $rng.AddComment([string]$op.comment)
        try { $c.Shape.Width = 320; $c.Shape.Height = 110 } catch { }
        $c.Visible = $false
    }
    $n++
}
"cell ops applied: $n (values written: $corr)"

# --- Evidence sheet ---
foreach ($ws in $book.Worksheets) { if ($ws.Name -eq 'Evidence') { $ws.Delete() } }
$ev = $book.Worksheets.Add([Type]::Missing, $book.Worksheets.Item($book.Worksheets.Count))
$ev.Name = 'Evidence'
$hdr = @('Sheet','Item','Cell','Parameter','Value in sheet (before)','Corrected to','Source value / quote','Status','Source type','Source URL','Note','Checked')
$rows = $plan.evidence
$rowCount = $rows.Count
$arr = New-Object 'object[,]' $rowCount, $hdr.Count
for ($i = 0; $i -lt $rowCount; $i++) {
    $r = $rows[$i]
    for ($j = 0; $j -lt $hdr.Count; $j++) {
        $v = $r[$j]
        if ($null -eq $v) { $v = '' }
        elseif ($v -is [System.Management.Automation.PSCustomObject]) { $v = [string]$v }
        $arr[$i, $j] = $v
    }
}
$startRow = 9
$ev.Range($ev.Cells.Item($startRow, 1), $ev.Cells.Item($startRow, $hdr.Count)).Value2 = [object[]]$hdr
$ev.Range($ev.Cells.Item($startRow + 1, 1), $ev.Cells.Item($startRow + $rowCount, $hdr.Count)).Value2 = $arr
# counts
$counts = @{}
foreach ($r in $rows) { $s = [string]$r[7]; if ($counts.ContainsKey($s)) { $counts[$s]++ } else { $counts[$s] = 1 } }
$today = Get-Date -Format 'yyyy-MM-dd'
$ev.Cells.Item(1,1).Value2 = "EVIDENCE LOG – every data parameter in GPUs / CPUs / Models / Voice Models / Quant / NVMe constants checked against its primary source on $today"
$ev.Cells.Item(1,1).Font.Bold = $true
$ev.Cells.Item(2,1).Value2 = 'Legend (cell fill on the data sheets):'
$ev.Cells.Item(2,2).Value2 = 'VERIFIED = value matches primary source'; $ev.Cells.Item(2,2).Interior.Color = ToExcelColor(0xC6EFCE)
$ev.Cells.Item(2,4).Value2 = 'MISMATCH = value corrected to source (old value in comment + this log)'; $ev.Cells.Item(2,4).Interior.Color = ToExcelColor(0xFFC7CE)
$ev.Cells.Item(2,7).Value2 = 'Amber = estimate / derived / internal Intel doc / not stated on vendor page / source unavailable'; $ev.Cells.Item(2,7).Interior.Color = ToExcelColor(0xFFEB9C)
$ev.Cells.Item(2,10).Value2 = 'Blue = HF repo added (was blank)'; $ev.Cells.Item(2,10).Interior.Color = ToExcelColor(0xDDEBF7)
$ev.Cells.Item(3,1).Value2 = 'Status counts:'
$col = 2
foreach ($k in ($counts.Keys | Sort-Object)) { $ev.Cells.Item(3,$col).Value2 = "$k = $($counts[$k])"; $col++ }
$ev.Cells.Item(4,1).Value2 = 'Indicative prices (GPUs col G, CPUs col K) are planning placeholders with NO source – kept as-is per reviewer decision; cost outputs on sheet 1 / Design Checklist remain indicative only.'
$ev.Cells.Item(5,1).Value2 = 'Sources: Hugging Face config.json / safetensors metadata / model cards; Intel ARK; AMD product pages; NVIDIA product pages, whitepapers and datasheets; Apple tech specs; Raspberry Pi / Rockchip / NVIDIA Jetson pages; llama.cpp quantize.cpp; OCP MX spec; PCI-SIG / SATA-IO / USB-IF / JEDEC. Pre-production rows (if present in a private copy) cite non-public documents and are not externally verifiable.'
$ev.Cells.Item(6,1).Value2 = 'Each checked data cell also carries a cell note with the quote, source type and URL. Column AG/AH (GPUs), AE/AF (CPUs), P/Q (Models) summarise the source URL and status per row.'
$hdrRng = $ev.Range($ev.Cells.Item($startRow, 1), $ev.Cells.Item($startRow, $hdr.Count))
$hdrRng.Font.Bold = $true; $hdrRng.Interior.Color = ToExcelColor(0xD9E1F2)
$dataRng = $ev.Range($ev.Cells.Item($startRow, 1), $ev.Cells.Item($startRow + $rowCount, $hdr.Count))
$dataRng.AutoFilter() | Out-Null
# status colours in Evidence
for ($i = 0; $i -lt $rowCount; $i++) {
    $s = [string]$rows[$i][7]
    $c = switch -Regex ($s) { '^VERIFIED$' { 0xC6EFCE } '^MISMATCH$' { 0xFFC7CE } '^ADDED$' { 0xDDEBF7 } default { 0xFFEB9C } }
    $ev.Cells.Item($startRow + 1 + $i, 8).Interior.Color = ToExcelColor($c)
    $u = [string]$rows[$i][9]
    if ($u -and $u.StartsWith('http')) { $ev.Hyperlinks.Add($ev.Cells.Item($startRow + 1 + $i, 10), $u, [Type]::Missing, [Type]::Missing, $u) | Out-Null }
}
$widths = @(14, 34, 8, 24, 18, 14, 40, 20, 34, 60, 70, 11)
for ($j = 0; $j -lt $hdr.Count; $j++) { $ev.Columns.Item($j + 1).ColumnWidth = $widths[$j] }
try { $ev.Activate(); $app.ActiveWindow.FreezePanes = $false; $ev.Range("A$($startRow + 1)").Select() | Out-Null; $app.ActiveWindow.FreezePanes = $true } catch { "freeze panes skipped: $($_.Exception.Message)" }
# price placeholders header marks
foreach ($pc in $plan.price_cols) {
    $ws = $sheets[$pc[0]]; $h = $ws.Range("$($pc[1])1")
    $h.Interior.Color = ToExcelColor(0xFFEB9C)
    if ($null -ne $h.Comment) { $h.Comment.Delete() }
    $h.AddComment("[PLACEHOLDER – NO SOURCE] Indicative prices are planning placeholders; no vendor/distributor price source was used or verified ($today).") | Out-Null
}
# header styling for new URL/status columns
foreach ($pair in @(@('GPUs','AG','AH'), @('CPUs','AE','AF'), @('Models','P','Q'))) {
    $ws = $sheets[$pair[0]]
    foreach ($c in $pair[1..2]) { $h = $ws.Range("$($c)1"); $h.Font.Bold = $true; $h.Interior.Color = ToExcelColor(0xD9E1F2); $ws.Columns.Item($c).ColumnWidth = 48 }
}
$app.Calculation = $prevCalc
$app.Calculate()
$app.EnableEvents = $true
$app.ScreenUpdating = $true
$app.DisplayAlerts = $true
$book.Save()
"Saved=$($book.Saved) path=$($book.FullName) evidence rows=$rowCount"
