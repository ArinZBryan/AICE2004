#pragma once

#include "DataLoader.h"
#include "Maths.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace std;

class Network {
  public:
	// Fashion-MNIST dimensions
	static constexpr const int INPUT_SIZE = 784;
	static constexpr const int OUTPUT_SIZE = 10;

	Network(int hidden_size, unsigned int seed);
	~Network();

	// inference
	vector<float> forward(const std::vector<float>& input);
	int predict(const std::vector<float>& input);

	// training
	void backpropagate(const vector<float>& target, double learning_rate);

	// Getters
	const vector<vector<vector<float>>>& get_weights() const {
		return weights;
	}
	const vector<vector<float>>& get_bias() const {
		return bias;
	}

	void save_weights(const std::string& path) const;

  private:
	void xavier_initialization(vector<vector<float>>& W, int in_dim, int out_dim);
	int hidden_size;
	unsigned int random_seed;
	vector<vector<vector<float>>> weights;
	vector<vector<float>> bias;

	// Intermediary variables for backpropagation
	vector<float> last_input;
	vector<float> last_x1;
	vector<float> last_x2;
	vector<float> last_x3;
	vector<float> last_x4;
	vector<float> last_x5;
	vector<float> last_output;
};
