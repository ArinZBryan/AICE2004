cmake_minimum_required(VERSION 3.5)
project(aice-fashion-network)

# Enforce use of GNU g++ and require a minimum compiler version (4.8.5)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	message(FATAL_ERROR "This project is configured to build with GNU g++ (GCC). Detected compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.5")
	message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} detected. This project requires GCC >= 4.8.5.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find MPI
find_package(MPI REQUIRED C CXX)

# Find TBB
find_package(TBB REQUIRED)

# Find Catch2
find_package(Catch2 3 REQUIRED)

# Use gnu++11 (GCC 4.8.5 supports C++11 sufficiently for this project)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native -ffast-math -lm ${OpenMP_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")

include_directories(${CMAKE_SOURCE_DIR}/include)

# Build Main Executable
file(GLOB MAIN_SOURCES src/*.cpp)
if (MAIN_SOURCES)
	add_executable(main main.cpp ${MAIN_SOURCES})
	target_include_directories(main PRIVATE ${MPI_CXX_INCLUDE_DIRS})
	target_link_libraries(main OpenMP::OpenMP_CXX MPI::MPI_CXX TBB::tbb)
else()
	message(FATAL_ERROR "No source files found in the src/ directory. Exiting.")
endif()

# Build Unit Test Executable
file(GLOB TEST_SOURCES tests/*.cpp)
if (TEST_SOURCES)
	add_executable(tests ${TEST_SOURCES})
	target_sources(tests PRIVATE ${MAIN_SOURCES})
	target_link_libraries(tests MPI::MPI_CXX TBB::tbb Catch2::Catch2WithMain)
else()
	message(WARNING "No test files found in the tests/ directory. Skipping test target creation.")
endif()

# Build Accuracy Test Executable
if (MAIN_SOURCES)
	add_executable(accuracy accuracy-test-main.cpp)
	target_sources(accuracy PRIVATE ${MAIN_SOURCES})
	target_link_libraries(accuracy MPI::MPI_CXX TBB::tbb)
else()
	message(WARNING "No source files found in the src/ directory. Skipping accuracy test target creation.")
endif()

include(CTest)
include(Catch)
catch_discover_tests(tests)
