#pragma once
#include "Vector.h"
#include "Matrix.h"
#include <vector>
#include <cstddef>
#include <limits>

Vector mat_times_vec(const Matrix &mat, const Vector &vec);
void mat_times_vec(const Matrix& mat, const Vector& vec, Vector& out);
Vector mat_transpose_times_vec(const Matrix& mat, const Vector &vec);
void mat_transpose_times_vec(const Matrix& mat, const Vector& vec, Vector& out);
Matrix mat_plus_mat(const Matrix& mat1, const Matrix& mat2, bool useAVX);
void mat_plus_mat(Matrix& mat1, Matrix& mat2, Matrix& out, bool useAVX);

Vector vec_plus_vec(const Vector &vec1, const Vector &vec2, bool useAVX);
void vec_plus_vec(Vector& vec1, Vector& vec2, Vector& out, bool useAVX);
Vector vec_minus_vec(const Vector &vec1, const Vector &vec2);
void vec_minus_vec(Vector& vec1, Vector& vec2, Vector& out);
Vector multiply_elementwise_vec(const Vector &vec1, const Vector &vec2);
void multiply_elementwise_vec(Vector& vec1, Vector& vec2, Vector& out);
Vector divide_elementwise_vec(const Vector &vec1, const Vector &vec2);
void divide_elementwise_vec(Vector &vec1, Vector &vec2, Vector& out);

Vector sigmoid_vec(const Vector &vec);
void sigmoid_vec(Vector& vec, Vector& out);
Vector sigmoid_derivative(const Vector &vec);
void sigmoid_derivative(Vector& vec, Vector& out);
Vector precomputed_sigmoid_derivative(const Vector& vec);
void precomputed_sigmoid_derivative(Vector& vec, Vector& out);

Vector softmax_vec(const Vector &vec);
void softmax_vec(Vector& vec, Vector& out);
Vector cross_entropy_loss(const Vector &predicted, const Vector &actual);
void cross_entropy_loss(Vector& predicted, Vector& actual, Vector& out);

Matrix transpose(const Matrix &x);
void transpose(const Matrix& x, Matrix& out);

Matrix outer_product(const Vector &a, const Vector &b);
void outer_product(const Vector& a, const Vector& b, Matrix& out);