#!/bin/bash

# Check that we are in correct directory
if ! [ -f "CMakeLists.txt" ] && grep -q "project(aice-fashion-network)" CMakeLists.txt; then
    echo "error: in the wrong directory. Please go to root of project and run: scripts/build.sh"
    exit 1
fi

# MUST be run from root of project.
if [ ! -d build ]; then 
    mkdir build
    mkdir build/main;
    mkdir build/test;
    mkdir build/bench;
fi

##### INSERT BUILD INSTRUCTIONS BELOW THIS LINE #####
if [ "$1" == "main" ]; then
    if [ "$2" == "--release" ]; then
        if [ ! -d build/main/release ]; then mkdir build/main/release; fi
        cmake -S . -B build/main/release -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" -DCMAKE_CXX_FLAGS="-Wall -Wextra -O3 -DUSE_MPI";
        cmake --build build/main/release --target main -j12;
    elif [ "$2" == "--debug" ]; then
        if [ ! -d build/main/debug ]; then mkdir build/main/debug; fi
        cmake -S . -B build/main/debug -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" -DCMAKE_CXX_FLAGS="-Wall -Wextra -O0 -g -fno-omit-frame-pointer -fsanitize=address,undefined";
        cmake --build build/main/debug --target main -j12;
    elif [ "$2" == "--profile" ]; then
        if [ ! -d build/main/profile ]; then mkdir build/main/profile; fi
        cmake -S . -B build/main/profile -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" -DCMAKE_CXX_FLAGS="-Wall -Wextra -O3 -pg -DUSE_MPI";
        cmake --build build/main/profile --target main -j12;
    fi
elif [ "$1" == "test" ]; then
    if [ ! -d build/test ]; then mkdir build/test; fi
    cmake -S . -B build/test -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" -DCMAKE_CXX_FLAGS="-O0";
    cmake --build build/test --target tests -j12;
elif [ "$1" == "compare" ]; then
    if [ ! -d build/compare ]; then mkdir build/compare; fi
    cmake -S . -B build/compare -DCMAKE_CXX_FLAGS="-O3";
    cmake --build build/compare --target compare -j12;
else
    if [ ! -d build/main/release ]; then mkdir build/main/release; fi
    cmake -S . -B build/main/release -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" -DCMAKE_CXX_FLAGS="-Wall -Wextra -O3 -DUSE_MPI";
    cmake --build build/main/release --target main -j12;
fi
##### INSERT BUILD INSTRUCTIONS ABOVE THIS LINE #####

