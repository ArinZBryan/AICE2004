#!/bin/bash
set -euo pipefail

# Check that we are in correct directory
if ! [ -f "CMakeLists.txt" ] || ! grep -q "project(aice-fashion-network)" CMakeLists.txt; then
    echo "error: in the wrong directory. Please go to root of project and run: scripts/bench.sh"
    exit 1
fi

# Check if the executable has been produced.
if ! [ -x build/bench/accuracy ]; then
    echo "error: executable build/bench/accuracy not produced. Please run scripts/build.sh bench and make sure it produces an executable build/bench/accuracy"
    exit 1
fi

##### INSERT RUN INSTRUCTIONS BELOW THIS LINE #####
mpirun -np 1 build/bench/accuracy
##### INSERT RUN INSTRUCTIONS ABOVE THIS LINE #####
