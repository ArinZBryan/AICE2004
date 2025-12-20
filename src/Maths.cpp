
#include "Maths.h"
#include <algorithm>
#include <cmath>

vector<float> mat_times_vec(const vector<vector<float>>& mat, const vector<float>& vec) {
	vector<float> result(mat.size(), 0.0f);
	for (size_t i = 0; i < mat.size(); ++i) {
		for (size_t j = 0; j < vec.size(); ++j) {
			result[i] += mat[i][j] * vec[j];
		}
	}
	return result;
}

vector<float> vec_plus_vec(const vector<float>& vec1, const vector<float>& vec2) {
	vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] + vec2[i];
	}
	return result;
}

vector<float> sigmoid_vec(const vector<float>& vec) {
	vector<float> result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = 1.0f / (1.0f + exp(-vec[i]));
	}
	return result;
}

vector<float> softmax_vec(const vector<float>& vec) {
	vector<float> result(vec.size());
	float max_val = *max_element(vec.begin(), vec.end());
	float sum = 0.0f;
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = exp(vec[i] - max_val);
		sum += result[i];
	}
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] /= sum;
	}
	return result;
}

vector<float> cross_entropy_loss(const vector<float>& predicted, const vector<float>& actual) {
	vector<float> result(predicted.size());
	for (size_t i = 0; i < predicted.size(); ++i) {
		result[i] = -actual[i] * log(predicted[i] + 1e-15f); // add small value to avoid log(0)
	}
	return result;
}

vector<float> vec_minus_vec(const vector<float>& vec1, const vector<float>& vec2) {
	vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] - vec2[i];
	}
	return result;
}

vector<vector<float>> transpose(const vector<vector<float>>& x) {
	// x can be mat or vec
	if (x.empty())
		return {};
	size_t rows = x.size();
	size_t cols = x[0].size();
	vector<vector<float>> result(cols, vector<float>(rows));
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result[j][i] = x[i][j];
		}
	}
	return result;
}

vector<float> sigmoid_derivative(const vector<float>& vec) {
	vector<float> sig = sigmoid_vec(vec);
	vector<float> result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = sig[i] * (1.0f - sig[i]);
	}
	return result;
}

vector<float> multiply_elementwise_vec(const vector<float>& vec1, const vector<float>& vec2) {
	vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] * vec2[i];
	}
	return result;
}

void outer_product_into(const vector<float>& a, const vector<float>& b, vector<vector<float>>& out) {
	size_t m = a.size();
	size_t n = b.size();
	out.assign(m, vector<float>(n, 0.0f));
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			out[i][j] = a[i] * b[j];
		}
	}
}

vector<vector<float>> outer_product(const vector<float>& a, const vector<float>& b) {
	size_t m = a.size();
	size_t n = b.size();
	vector<vector<float>> out(m, vector<float>(n, 0.0f));
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			out[i][j] = a[i] * b[j];
		}
	}
	return out;
}