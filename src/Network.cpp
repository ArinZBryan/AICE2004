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
	last_input = input;
	// Input to hidden: weights * input + bias
	last_x1 = mat_times_vec(weights[0], input);
	last_x2 = vec_plus_vec(last_x1, bias[0]);

	// Activation function: Sigmoid
	last_x3 = sigmoid_vec(last_x2);

	// Hidden to output: weights * hidden + bias
	last_x4 = mat_times_vec(weights[1], last_x3);
	last_x5 = vec_plus_vec(last_x4, bias[1]);

	// Apply softmax to get probabilities
	last_output = softmax_vec(last_x5);

	return last_output;
}

void Network::backpropagate(const Vector& target, double learning_rate) {
	// Derivative of loss w.r.t. output (cross-entropy loss with softmax)
	Vector delta_out = vec_minus_vec(last_output, target);

	// Gradient for hidden -> output weights and biases
	Vector z4 = delta_out;
	Matrix gradients_W2 = outer_product(z4, last_x3);

	// Gradient for input -> hidden weights and biases
	Matrix W2T = transpose(weights[1]);
	Vector z3 = mat_times_vec(W2T, z4);
	Vector sig_deriv = sigmoid_derivative(last_x2);
	Vector z2 = multiply_elementwise_vec(z3, sig_deriv);
	Matrix gradients_W1 = outer_product(z2, last_input);

	// Update the biases
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		for (size_t j = 0; j < hidden_size; j++) {
			weights[1](i,j) -= learning_rate * gradients_W2(i,j);
		}
		bias[1](i) -= learning_rate * z4(i);
	}

	// Update the weights
	for (size_t i = 0; i < hidden_size; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			weights[0](i,j) -= learning_rate * gradients_W1(i,j);
		}
		bias[0](i) -= learning_rate * z2(i);
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