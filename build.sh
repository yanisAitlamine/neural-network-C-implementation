#!/bin/bash

# Get a list of all .c files in the current directory
c_files=$(find ./source -maxdepth 1 -type f -name "*.c")

# Check if the user provided the name of the resulting executable
if [ $# -eq 0 ]; then
    echo "Please provide the name of the resulting executable."
    echo "Usage: ./build.sh <executable_name> <path to main>"
    exit 1
fi

executable_name=$1
path_to_main=$2

# Compile each .c file with gcc
echo "gcc -O3 -o $executable_name $path_to_main $c_files -lm"
echo "==================================================================="

gcc -O3 -o $executable_name $path_to_main $c_files -lm

