#!/bin/bash

# Get a list of all .c files in the current directory
c_files=$(find . -maxdepth 1 -type f -name "*.c")
echo $c_files
# Check if the user provided the name of the resulting executable
if [ $# -eq 0 ]; then
    echo "Please provide the name of the resulting executable."
    echo "Usage: ./compile.sh <executable_name>"
    exit 1
fi

executable_name=$1

# Compile each .c file with gcc

gcc -o $1 $c_files 

