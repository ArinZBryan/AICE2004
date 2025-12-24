#pragma once
//#define USE_FP64

#ifndef USE_FP64
typedef float number;
#define MPI_NUMBER MPI_FLOAT
#else
typedef double number;
#define MPI_NUMBER MPI_DOUBLE
#endif