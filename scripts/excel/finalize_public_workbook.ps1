# Recalculates the public workbook in Excel and saves it with calculated values and without personal information.
# Run after: python scripts/build_public_workbook.py     (Windows with Excel only)
param([string]$Workbook = (Join-Path $PSScriptRoot "..\..\excel\Offline_LLM_Hardware_Selector_public.xlsx"))

$path = (Resolve-Path $Workbook).Path
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
try {
    $book = $excel.Workbooks.Open($path)
    $excel.CalculateFull()
    $book.RemovePersonalInformation = $true
    $book.Worksheets.Item(1).Activate()
    $book.Save()
    $book.Close($false)
    "recalculated and saved $path"
} finally {
    $excel.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
}
