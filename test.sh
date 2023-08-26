rm NNtest.nn
rm log
./build.sh test.exe
./test.exe >> log
nvim log
