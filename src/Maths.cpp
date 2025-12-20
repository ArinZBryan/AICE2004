
#include "Maths.h"
#include "Vector.h"
#include "Matrix.h"
#include <algorithm>
#include <cmath>
#include <cassert>

Vector mat_times_vec(const Matrix &mat, const Vector &vec) {
	assert((mat.cols() == vec.size()));
	Vector result(mat.rows(), 0.0f);
	for (size_t row = 0; row < mat.rows(); ++row) {
		for (size_t col = 0; col < mat.cols(); ++col) {
			float val1 = mat(row, col);
			float val2 = vec(col);
			result(row) += val1 * val2;
		}
	}
	return result;
}

Vector vec_plus_vec(const Vector &vec1, const Vector &vec2) {
	Vector result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result(i) = vec1(i) + vec2(i);
	}
	return result;
}

Vector sigmoid_vec(const Vector& vec) {
	Vector result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result(i) = 1.0f / (1.0f + exp(-vec(i)));
	}
	return result;
}

Vector softmax_vec(const Vector &vec) {
	Vector result(vec.size());
	float max_val = -std::numeric_limits<float>::infinity();
	for (size_t i = 0; i < vec.size(); i++) { max_val = (vec(i) > max_val) ? vec(i) : max_val; }
	float sum = 0.0f;
	for (size_t i = 0; i < vec.size(); ++i) {
		result(i) = exp(vec(i) - max_val);
		sum += result(i);
	}
	for (size_t i = 0; i < vec.size(); ++i) {
		result(i) /= sum;
	}
	return result;
}

Vector cross_entropy_loss(const Vector &predicted, const Vector &actual) {
	Vector result(predicted.size());
	for (size_t i = 0; i < predicted.size(); ++i) {
		result(i) = -actual(i) * log(predicted(i) + 1e-15f); // add small value to avoid log(0)
	}
	return result;
}

Vector vec_minus_vec(const Vector &vec1, const Vector &vec2) {
	Vector result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result(i) = vec1(i) - vec2(i);
	}
	return result;
}

Matrix transpose(const Matrix &x) {
	// x can be mat or vec
	assert((x.cols() != 0 && x.rows() != 0));
	size_t rows = x.rows();
	size_t cols = x.cols();
	Matrix result(cols, rows);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(j,i) = x(i,j);
		}
	}
	return result;
}

Vector sigmoid_derivative(const Vector &vec) {
	Vector sig = sigmoid_vec(vec);
	Vector result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result(i) = sig(i) * (1.0f - sig(i));
	}
	return result;
}

Vector multiply_elementwise_vec(const Vector &vec1, const Vector &vec2) {
	Vector result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result(i) = vec1(i) * vec2(i);
	}
	return result;
}

Matrix outer_product(const Vector &a, const Vector &b) {
	size_t m = a.size();
	size_t n = b.size();
	Matrix out(m, n, 0.0f);
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			out(i,j) = a(i) * b(j);
		}
	}
	return out;
}