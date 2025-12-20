#pragma once
#include <vector>
using namespace std;

vector<float> mat_times_vec(const vector<vector<float>>& mat, const vector<float>& vec);
vector<float> vec_plus_vec(const vector<float>& vec1, const vector<float>& vec2);
vector<float> vec_minus_vec(const vector<float>& vec1, const vector<float>& vec2);

vector<float> sigmoid_vec(const vector<float>& vec);
vector<float> sigmoid_derivative(const vector<float>& vec);

vector<float> softmax_vec(const vector<float>& vec);
vector<float> cross_entropy_loss(const vector<float>& predicted, const vector<float>& actual);

vector<vector<float>> transpose(const vector<vector<float>>& x);
vector<float> multiply_elementwise_vec(const vector<float>& vec1, const vector<float>& vec2);

// Compute outer product of a and b into `out` (non-allocating). Resizes `out` to a.size() x b.size().
// out[i][j] = a[i] * b[j]
void outer_product_into(const vector<float>& a, const vector<float>& b, vector<vector<float>>& out);

// Allocate-and-return outer product: returns matrix MxN with out[i][j] = a[i] * b[j]
vector<vector<float>> outer_product(const vector<float>& a, const vector<float>& b);