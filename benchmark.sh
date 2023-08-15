
#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <program> [arguments...]"
    exit 1
fi

program="$1"
shift

if ! command -v "$program" >/dev/null 2>&1; then
    echo "Error: $program not found in PATH."
    exit 1
fi

echo "Benchmarking $program..."

# Number of iterations to run the program
iterations=10

# Run the program for the specified number of iterations and measure execution time
total_time=0

for i in $(seq 1 $iterations); do
    echo "Iteration $i..."
    start_time=$(date +%s.%N)
    "$program" "$@"
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    total_time=$(echo "$total_time + $elapsed_time" | bc)
done

average_time=$(echo "scale=6; $total_time / $iterations" | bc)

echo "Benchmark complete."
echo "Average execution time: $average_time seconds"
