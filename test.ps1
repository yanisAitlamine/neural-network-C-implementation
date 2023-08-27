#rm NNtest.nn
rm log
./build.ps1 test.exe .\applications\mnist.c
./test.exe >> log
