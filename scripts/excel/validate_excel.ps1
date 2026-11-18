param([string]$Workbook = 'Offline_LLM_Hardware_Selector_v27.xlsx')
$ErrorActionPreference = 'Stop'
$app = [RunningCom]::Get('Excel.Application')
$book = $null
foreach ($b in $app.Workbooks) { if ($b.Name -eq $Workbook) { $book = $b } }
$app.Calculate()
$errs = @()
foreach ($ws in $book.Worksheets) {
    try { $e = $ws.UsedRange.SpecialCells(-4123, 16); $errs += "$($ws.Name): $($e.Count) error cells e.g. $($e.Address($false,$false))" } catch { }
}
"Formula errors: " + $(if ($errs) { $errs -join ' | ' } else { 'none' })
$ev = $book.Worksheets.Item('Evidence')
"Evidence rows: " + ($ev.UsedRange.Rows.Count - 9) + "  sheets: " + $book.Worksheets.Count
$chk = @(@('Models','C30'), @('Models','E158'), @('Models','K41'), @('Models','J151'), @('GPUs','D40'), @('GPUs','E56'), @('GPUs','D101'), @('CPUs','H10'), @('CPUs','E62'), @('CPUs','H63'), @('Models','P2'), @('GPUs','AG5'), @('CPUs','AF25'))
foreach ($c in $chk) { $r = $book.Worksheets.Item($c[0]).Range($c[1]); $cm = if ($r.Comment) { $r.Comment.Text().Substring(0, [Math]::Min(90, $r.Comment.Text().Length)) } else { '' }; "{0}!{1} = {2} | fill={3:X6} | {4}" -f $c[0], $c[1], $r.Value2, [int]$r.Interior.Color, $cm }
$cnt = 0; foreach ($n in @('Models','GPUs','CPUs','Voice Models','Quant')) { $cnt += $book.Worksheets.Item($n).Comments.Count }
"Total cell notes on data sheets: $cnt"
"Sheet1 cost text: " + $book.Worksheets.Item('1 Plan a Build').Range('G15').Text
"Sheet1 F6 (model GB): " + $book.Worksheets.Item('1 Plan a Build').Range('C6').Text + ' / H? ' + $book.Worksheets.Item('Engine Selector').Range('B19').Text
"Saved=" + $book.Saved
