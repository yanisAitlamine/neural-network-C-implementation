param(
    [string]$arg1,
    [string]$arg2
)
if (!$PSBoundParameters.ContainsKey('arg1')){
    echo "Please provide the name of the resulting executable."
    echo "Usage: ./test.ps1 <executable_name> <main_path>"
    exit 1
}
if (!$PSBoundParameters.ContainsKey('arg2')){
    echo "Please provide the name of the resulting executable."
    echo "Usage: ./test.ps1 <executable_name> <main_path>"
    exit 1
}
rm NNtest.nn
rm log
rm errors.txt
./build.ps1 $arg1 $arg2
$startTime = Get-Date

Invoke-Expression ".\$arg1 > log 2>&1"

$endTime = Get-Date
$executionTime = $endTime - $startTime

Write-Host "Execution time: $executionTime"
"Execution time: $executionTime" >> log
