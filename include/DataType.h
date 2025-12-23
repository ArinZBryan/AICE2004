#pragma once

//#define USE_FP64

#ifndef USE_FP64
typedef float number;
#else
typedef double number;
#endif