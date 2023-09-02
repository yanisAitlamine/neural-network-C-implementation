param(
    [string]$arg1,
    [string]$arg2
)

$c_files=Get-ChildItem -Path ".\source\" 

if (!$PSBoundParameters.ContainsKey('arg1')){
    echo "Please provide the name of the resulting executable."
    echo "Usage: ./build.ps1 <executable_name> <main_path>"
    exit 1
}
if (!$PSBoundParameters.ContainsKey('arg2')){
    echo "Please provide the path to the main."
    echo "Usage: ./build.ps1 <executable_name> <main_path>"
    exit 1
}
$gccArgs = "-O3 -o $arg1 $arg2 $($c_files.FullName -join ' ') -lm"

echo "gcc $gccArgs"
echo "==================================================================="

Start-Process -Wait -FilePath "gcc" -ArgumentList $gccArgs -RedirectStandardError "errors.txt"
