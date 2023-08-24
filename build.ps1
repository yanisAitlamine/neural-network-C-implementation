param(
    [string]$arg1
)

$c_files=Get-ChildItem -Path ".\source\" 

if (!$PSBoundParameters.ContainsKey('arg1')){
     echo "Please provide the name of the resulting executable."
    echo "Usage: ./build.ps1 <executable_name>"
    exit 1
}
$gccArgs = "-o $arg1 $($c_files.FullName -join ' ') -lm"

echo "gcc $gccArgs"
echo "==================================================================="

Start-Process -Wait -FilePath "gcc" -ArgumentList $gccArgs