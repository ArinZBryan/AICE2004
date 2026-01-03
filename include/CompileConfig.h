#pragma once

// Configure what features this program supports during compilation

// Use double precision floats rather than single precision
//#define USE_FP64              

// Use MPI acceleration where available - leave undefined here to allow for defintion in compiler flags
//#define USE_MPI      

// Use AVX intrinsic version of these functions
#define USE_AVX_MAT_TIMES_VEC                  
#define USE_AVX_MAT_TRANSPOSE_TIMES_VEC        
#define USE_AVX_MAT_PLUS_MAT                   
#define USE_AVX_VEC_PLUS_VEC                   
#define USE_AVX_VEC_MINUS_VEC                  
#define USE_AVX_MULTIPLY_ELEMENTWISE_VEC       
#define USE_AVX_DIVIDE_ELEMENTWISE_VEC         
#define USE_AVX_SIGMOID_VEC                    
#define USE_AVX_SIGMOID_DERIVATIVE             
#define USE_AVX_PRECOMPUTED_SIGMOID_DERIVATIVE 
#define USE_AVX_SOFTMAX_VEC                    
#define USE_AVX_CROSS_ENTROPY_LOSS             
#define USE_AVX_OUTER_PRODUCT                  

// Do Not Modify!
#ifndef USE_FP64
typedef float number;
#define MPI_NUMBER MPI_FLOAT
#else
typedef double number;
#define MPI_NUMBER MPI_DOUBLE
#endif