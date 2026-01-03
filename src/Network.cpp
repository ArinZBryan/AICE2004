#include "Network.h"
#include "Matrix.h"
#include "CompileConfig.h"
#include <cstdint>
#include <fstream>
#include <iomanip>

Network::Network(unsigned int hidden_size, unsigned int seed) 
: hidden_size(hidden_size), random_seed(seed), weight1(hidden_size, INPUT_SIZE), weight2(OUTPUT_SIZE, hidden_size), bias1(hidden_size, 0.0), bias2(OUTPUT_SIZE, 0.0) {
	xavier_initialization(weight1, INPUT_SIZE, hidden_size);
	xavier_initialization(weight2, hidden_size, OUTPUT_SIZE);
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
			float v = static_cast<float>(weight1(i,j));
			ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
		}
	}

	// Write W2 (OUTPUT_SIZE x hidden)
	for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
		for (uint32_t j = 0; j < hidden; ++j) {
			float v = static_cast<float>(weight2(i,j));
			ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
		}
	}

	// Write b1 (hidden)
	for (uint32_t i = 0; i < hidden; ++i) {
		float v = static_cast<float>(bias1(i));
		ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
	}

	// Write b2 (OUTPUT_SIZE)
	for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
		float v = static_cast<float>(bias2(i));
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
	Vector v = mat_times_vec(weight1, input);
	vec_plus_vec(v, bias1, v);

	// Activation function: Sigmoid
	sigmoid_vec(v, v);

	// Hidden to output: weights * hidden + bias
	Vector result = mat_times_vec(weight2, v);
	vec_plus_vec(result, bias2, result);

	// Apply softmax to get probabilities
	softmax_vec(result, result);

	return result;
}

void Network::train(const Vector& input, const Vector& target, const Matrix* weight1, const Matrix* weight2, const Vector* bias1, const Vector* bias2, Network::TrainResult& out) {
	
	// ======================= Forward Step ========================
	// Input->Hidden: weights * input + bias
	Vector sigmoid = mat_times_vec(*weight1, input);
	vec_plus_vec(sigmoid, const_cast<Vector&>(*bias1), sigmoid);

	// Input->Hidden activation function
	sigmoid_vec(sigmoid, sigmoid);

	// Hidden->Output: weights * hidden + bias
	mat_times_vec(*weight2, sigmoid, out.bias2_grad);
	vec_plus_vec(out.bias2_grad, const_cast<Vector&>(*bias2), out.bias2_grad);

	// Output probability function
	softmax_vec(out.bias2_grad, out.bias2_grad);

	// ==================== Calculate CSE Delta ====================
	cross_entropy_loss(out.bias2_grad, const_cast<Vector&>(target), out.cse_delta);

	// ====================== Backpropogation ======================
	vec_minus_vec(out.bias2_grad, const_cast<Vector&>(target), out.bias2_grad);

	// Gradient for hidden -> output weights and biases
	outer_product(out.bias2_grad, sigmoid, out.weight2_grad);

	// Gradient for input -> hidden weights and biases
	mat_transpose_times_vec(*weight2, out.bias2_grad, out.bias1_grad);
	precomputed_sigmoid_derivative(sigmoid, sigmoid);
	multiply_elementwise_vec(out.bias1_grad, sigmoid, out.bias1_grad);
	outer_product(out.bias1_grad, input, out.weight1_grad);
}

void Network::update(float learning_rate, const Network::TrainResult& result) {
	for (size_t i = 0; i < hidden_size; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			weight1(i,j) -= learning_rate * result.weight1_grad(i,j);
		}
		bias1(i) -= learning_rate * result.bias1_grad(i);
	}

	for (int i = 0; i < Network::OUTPUT_SIZE; i++) {
		for (size_t j = 0; j < this->hidden_size; j++) {
			weight2(i,j) -= learning_rate * result.weight2_grad(i,j);
		}
		bias2(i) -= learning_rate * result.bias2_grad(i);
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