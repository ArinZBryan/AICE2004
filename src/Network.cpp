#include "Network.h"
#include "Matrix.h"
#include "DataType.h"
#include <cstdint>
#include <fstream>
#include <iomanip>

Network::Network(unsigned int hidden_size, unsigned int seed) : hidden_size(hidden_size), random_seed(seed) {
	weights = { 
		Matrix(hidden_size, INPUT_SIZE),
		Matrix(OUTPUT_SIZE, hidden_size)
	};

	bias = {
	    Vector(hidden_size, 0.0f),
	    Vector(OUTPUT_SIZE, 0.0f)
	};

	xavier_initialization(weights[0], INPUT_SIZE, hidden_size);
	xavier_initialization(weights[1], hidden_size, OUTPUT_SIZE);
}

void Network::save_weights(const std::string &path) const {
	std::ofstream ofs(path, std::ios::binary);
	if (!ofs) {
		throw std::runtime_error("Unable to open file for writing: " + path);
	}

	// Header: three uint32_t values: INPUT_SIZE, hidden_size, OUTPUT_SIZE
	uint32_t in     = static_cast<uint32_t>(INPUT_SIZE);
	uint32_t hidden = static_cast<uint32_t>(hidden_size);
	uint32_t out    = static_cast<uint32_t>(OUTPUT_SIZE);
	ofs.write(reinterpret_cast<const char *>(&in), sizeof(in));
	ofs.write(reinterpret_cast<const char *>(&hidden), sizeof(hidden));
	ofs.write(reinterpret_cast<const char *>(&out), sizeof(out));

	// Write W1 (hidden x INPUT_SIZE)
	for (uint32_t i = 0; i < hidden; ++i) {
		for (uint32_t j = 0; j < static_cast<uint32_t>(INPUT_SIZE); ++j) {
			float v = static_cast<float>(weights[0](i,j));
			ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
		}
	}

	// Write W2 (OUTPUT_SIZE x hidden)
	for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
		for (uint32_t j = 0; j < hidden; ++j) {
			float v = static_cast<float>(weights[1](i,j));
			ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
		}
	}

	// Write b1 (hidden)
	for (uint32_t i = 0; i < hidden; ++i) {
		float v = static_cast<float>(bias[0](i));
		ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
	}

	// Write b2 (OUTPUT_SIZE)
	for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
		float v = static_cast<float>(bias[1](i));
		ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
	}

	ofs.close();
}

Network::~Network() {}

void Network::xavier_initialization(Matrix &W, int in_dim, int out_dim) {
	std::mt19937 gen(random_seed);

	//I don't know why, but if I change the random number generation to use `number`, and thus
	//allow the use of FP64, I get completely different numbers out. So, to ensure compatibility
	//we generate the numbers as floats and cast them to doubles only if we need to to preserve
	//values.
	float limit = sqrt(6.0f / (in_dim + out_dim));
	std::uniform_real_distribution<float> dist(-limit, limit);

	for (int i = 0; i < out_dim; i++) {
		for (int j = 0; j < in_dim; j++) {
			W(i,j) = static_cast<number>(dist(gen));
		}
	}
}

Vector Network::forward(const Vector &input) {
	// Input to hidden: weights * input + bias
	Vector v = mat_times_vec(weights[0], input);
	vec_plus_vec(v, bias[0], v);

	// Activation function: Sigmoid
	sigmoid_vec(v, v);

	// Hidden to output: weights * hidden + bias
	Vector result = mat_times_vec(weights[1], v);
	vec_plus_vec(result, bias[1], result);

	// Apply softmax to get probabilities
	softmax_vec(result, result);

	return result;
}

Network::TrainResult Network::train(const Vector& input, const Vector& target, const std::vector<Matrix>& weights, const std::vector<Vector>& bias) {
	
	// ======================= Forward Step ========================
	// Input->Hidden: weights * input + bias
	Vector sigmoid = mat_times_vec(weights[0], input);
	vec_plus_vec(sigmoid, const_cast<Vector&>(bias[0]), sigmoid);

	// Input->Hidden activation function
	sigmoid_vec(sigmoid, sigmoid);

	// Hidden->Output: weights * hidden + bias
	Vector forward_output = mat_times_vec(weights[1], sigmoid);
	vec_plus_vec(forward_output, const_cast<Vector&>(bias[1]), forward_output);

	// Output probability function
	softmax_vec(forward_output, forward_output);

	// ==================== Calculate CSE Delta ====================
	Vector cse_delta = cross_entropy_loss(forward_output, target);

	// ====================== Backpropogation ======================
	Vector& bias2_gradient = forward_output;
	vec_minus_vec(forward_output, const_cast<Vector&>(target), bias2_gradient);

	// Gradient for hidden -> output weights and biases
	Matrix weight2_gradient = outer_product(bias2_gradient, sigmoid);

	// Gradient for input -> hidden weights and biases
	Vector bias1_gradient = mat_transpose_times_vec(weights[1], bias2_gradient);
	precomputed_sigmoid_derivative(sigmoid, sigmoid);
	multiply_elementwise_vec(bias1_gradient, sigmoid, bias1_gradient);
	Matrix weight1_gradient = outer_product(bias1_gradient, input);

	return { weight1_gradient, weight2_gradient, bias1_gradient, bias2_gradient, cse_delta };
}

void Network::update(float learning_rate, const Network::TrainResult& result) {
	for (size_t i = 0; i < hidden_size; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			weights[0](i,j) -= learning_rate * result.weight1_grad(i,j);
		}
		bias[0](i) -= learning_rate * result.bias1_grad(i);
	}

	for (int i = 0; i < Network::OUTPUT_SIZE; i++) {
		for (size_t j = 0; j < this->hidden_size; j++) {
			weights[1](i,j) -= learning_rate * result.weight2_grad(i,j);
		}
		bias[1](i) -= learning_rate * result.bias2_grad(i);
	}
}

int Network::predict(const Vector &input) {
	Vector output = forward(input);
	return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

int Network::predict(const std::vector<float> &input) {
	Vector output = forward(Vector(input));
	return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}