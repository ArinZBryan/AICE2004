#!/bin/bash
set -euo pipefail

# Check that we are in correct directory
if ! [ -f "CMakeLists.txt" ] || ! grep -q "project(aice-fashion-network)" CMakeLists.txt; then
    echo "error: in the wrong directory. Please go to root of project and run: scripts/bench.sh"
    exit 1
fi

# Check if the executable has been produced.
if ! [ -x build/bench/main ]; then
    echo "error: executable build/bench/main not produced. Please run scripts/build.sh bench and make sure it produces an executable build/bench/main"
    exit 1
fi

##### INSERT RUN INSTRUCTIONS BELOW THIS LINE #####
mpirun -np 1 build/bench/main
##### INSERT RUN INSTRUCTIONS ABOVE THIS LINE #####
