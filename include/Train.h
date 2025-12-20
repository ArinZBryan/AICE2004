#pragma once

#include <iomanip>
#include <sstream>
#include <vector>

#include "DataLoader.h"
#include "Maths.h"
#include "Network.h"

using namespace std;

// Training configuration structure
struct TrainConfig {
	unsigned int random_seed;
	unsigned int epochs;
	double learning_rate;
	unsigned int batch_size;
	unsigned int hidden_size;
	unsigned int threads = 1;
	unsigned int tasks = 1;

	string str() const {
		std::ostringstream oss;
		oss << "seed" << random_seed
		    << "-epochs" << epochs
		    << "-lr" << std::fixed << std::setprecision(6) << learning_rate
		    << "-batch" << batch_size
		    << "-hidden" << hidden_size
		    << "-threads" << threads
		    << "-tasks" << tasks;
		return oss.str();
	}
};

// Function to train the model
void train_model(Network& model, const vector<Sample> data, const TrainConfig& config);

// Function to evaluate the model
void evaluate_model(Network& model, const vector<Sample>& data, std::vector<int>* predictions = nullptr);
