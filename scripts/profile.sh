#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 -s SEED -e EPOCHS -l LEARNING_RATE -b BATCH -z HIDDEN_SIZE --threads THREADS"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s) SEED=$2; shift 2;;
        -e) EPOCHS=$2; shift 2;;
        -l) LR=$2; shift 2;;
        -b) BATCH=$2; shift 2;;
        -z) HIDDEN=$2; shift 2;;
        --threads) THREADS=$2; shift 2;;
        *) usage;;
    esac
done

# Check required arguments
: "${SEED:?Missing -s (seed)}"
: "${EPOCHS:?Missing -e (epochs)}"
: "${LR:?Missing -l (learning rate)}"
: "${BATCH:?Missing -b (batch size)}"
: "${HIDDEN:?Missing -z (hidden layer size)}"
: "${THREADS:?Missing --threads}"



# Check that we are in correct directory
if ! [ -f "CMakeLists.txt" ] || ! grep -q "project(aice-fashion-network)" CMakeLists.txt; then
    echo "error: in the wrong directory. Please go to root of project and run: scripts/profile.sh"
    exit 1
fi

# Check if the executable has been produced.
if ! [ -x build/main/profile/main ]; then
    echo "error: executable build/main/profile/main not produced. Please run scripts/build.sh main --profile  and make sure it produces an executable build/main/profile/main"
    exit 1
fi

##### INSERT RUN INSTRUCTIONS BELOW THIS LINE #####
mpirun -np 1 build/main/profile/main -s "$SEED" -e "$EPOCHS" -l "$LR" -b "$BATCH" -z "$HIDDEN" --threads "$THREADS" --tasks 1
##### INSERT RUN INSTRUCTIONS ABOVE THIS LINE #####
