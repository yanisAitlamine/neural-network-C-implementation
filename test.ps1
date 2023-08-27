param(
    [string]$arg1
)
if (!$PSBoundParameters.ContainsKey('arg1')){
    echo "Please provide the name of the resulting executable."
    echo "Usage: ./test.ps1 = <main_path>"
    exit 1
}
rm NNtest.nn
rm log
./build.ps1 test.exe $arg1
./test.exe >> log
