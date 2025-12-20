#pragma once
#include "Vector.h"
#include "Matrix.h"
#include <vector>
#include <cstddef>
#include <limits>

Vector mat_times_vec(const Matrix &mat, const Vector &vec);
Vector vec_plus_vec(const Vector &vec1, const Vector &vec2);
Vector vec_minus_vec(const Vector &vec1, const Vector &vec2);

Vector sigmoid_vec(const Vector &vec);
Vector sigmoid_derivative(const Vector &vec);

Vector softmax_vec(const Vector &vec);
Vector cross_entropy_loss(const Vector &predicted, const Vector &actual);

Matrix transpose(const Matrix &x);
Vector multiply_elementwise_vec(const Vector &vec1, const Vector &vec2);

// Allocate-and-return outer product: returns matrix MxN with out[i][j] = a[i] * b[j]
Matrix outer_product(const Vector &a, const Vector &b);