rm NNtest.nn
rm log
./build.sh test.exe

# Number of iterations to run the program
iterations=10

# Run the program for the specified number of iterations and measure execution time
total_time=0


start_time=$(date +%s.%N)
"./test.exe >> log" "$@"
end_time=$(date +%s.%N)
elapsed_time=$(echo "$end_time - $start_time" | bc)


echo "Benchmark complete."
echo "execution time: $elapsed_time seconds"

"execution time: $elapsed_time seconds" >> log



nvim log
