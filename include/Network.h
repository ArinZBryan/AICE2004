#pragma once

#include "DataLoader.h"
#include "Maths.h"
#include "Vector.h"
#include "Matrix.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <cstddef>

class Network {
  public:
	// Fashion-MNIST dimensions
	static constexpr const int INPUT_SIZE  = 784;
	static constexpr const int OUTPUT_SIZE = 10;

	Network(unsigned int hidden_size, unsigned int seed);
	~Network();

	// Inference
	int predict(const Vector &input);
	int predict(const std::vector<float> &input);

	// Training
	struct TrainResult {
		Matrix weight1_grad;
		Matrix weight2_grad;
		Vector bias1_grad;
		Vector bias2_grad;
		Vector cse_delta;
	};
	Vector forward(const Vector &input);
	static TrainResult train(const Vector& input, const Vector& target, const std::vector<Matrix>& weights, const std::vector<Vector>& bias);
	void update(float learning_rate, const TrainResult& result);


	// Getters
	const std::vector<Matrix> &get_weights() const {
		return weights;
	}
	const std::vector<Vector> &get_bias() const {
		return bias;
	}

	void save_weights(const std::string &path) const;

	// DO NOT USE OUTSIDE OF THE TESTING ENVIRONMENT
	struct state { 
		Network& network;
		void (Network::* func_xavier)(Matrix&, int, int);
		const unsigned int& hidden_size;
		const unsigned int& random_seed;
		const std::vector<Matrix>& weights;
		const std::vector<Vector>& bias;
	};
	// DO NOT IMPLEMENT OR USE OUTSIDE OF THE TESTING ENVIRONMENT
	friend state smuggle(Network& net);
	// the above ^^^^^ function is not implemented in the main project. This means
	// that any attempt to use them will result in a linker error. THIS IS CORRECT.
	// using this function and its associated struct means you accept that the risks
	// of completely breaking public/private encapsulation. For this, reason, it is
	// only acceptable to use this in tests, where private member variables and
	// functions are needed to test the class.


  private:
	void xavier_initialization(Matrix &W, int in_dim, int out_dim);
	unsigned int hidden_size;
	unsigned int random_seed;
	std::vector<Matrix> weights;
	std::vector<Vector> bias;
};
