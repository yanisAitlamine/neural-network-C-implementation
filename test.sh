rm NNtest.nn
rm log

# Check if the user provided the name of the resulting executable
if [ $# -eq 0 ]; then
    echo "Please provide the name of the resulting executable."
    echo "Usage: ./build.sh <executable_name> <path to main>"
    exit 1
fi

executable_name=$1
path_to_main=$2

./build.sh $executable_name $path_to_main

# Number of iterations to run the program
iterations=10

# Run the program for the specified number of iterations and measure execution time
total_time=0


start_time=$(date +%s.%N)
./$executable_name >> log 
end_time=$(date +%s.%N)
elapsed_time=$(echo "$end_time - $start_time" | bc)


echo "Benchmark complete."
echo "execution time: $elapsed_time seconds"
echo "execution time: $elapsed_time seconds" >> log



