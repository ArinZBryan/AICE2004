#pragma once

// Configure what features this program supports during compilation
#define USE_FP64                                // Use double precision floats rather than single precision
//#define USE_MPI                                 // Use MPI acceleration where available
#define USE_AVX_MAT_TIMES_VEC                   // Use AVX intrinsic version of this function
#define USE_AVX_MAT_TRANSPOSE_TIMES_VEC         // Use AVX intrinsic version of this function
#define USE_AVX_MAT_PLUS_MAT                    // Use AVX intrinsic version of this function
#define USE_AVX_VEC_PLUS_VEC                    // Use AVX intrinsic version of this function
#define USE_AVX_VEC_MINUS_VEC                   // Use AVX intrinsic version of this function
#define USE_AVX_MULTIPLY_ELEMENTWISE_VEC        // Use AVX intrinsic version of this function
#define USE_AVX_SIGMOID_VEC                     // Use AVX intrinsic version of this function
#define USE_AVX_SIGMOID_DERIVATIVE              // Use AVX intrinsic version of this function
#define USE_AVX_PRECOMPUTED_SIGMOID_DERIVATIVE  // Use AVX intrinsic version of this function
#define USE_AVX_SOFTMAX_VEC                     // Use AVX intrinsic version of this function
#define USE_AVX_CROSS_ENTROPY_LOSS              // Use AVX intrinsic version of this function
#define USE_AVX_OUTER_PRODUCT                   // Use AVX intrinsic version of this function

// Do Not Modify!
#ifndef USE_FP64
typedef float number;
#define MPI_NUMBER MPI_FLOAT
#else
typedef double number;
#define MPI_NUMBER MPI_DOUBLE
#endif