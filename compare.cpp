#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

#include <cmath>
#include <cstdint>

#include "Maths.h"
#include "Matrix.h"
#include "Network.h"
#include "Train.h"
#include "Vector.h"
#include "Arguments.h"

struct WeightsFile {
	uint32_t INPUT_SIZE;
	uint32_t hidden_size;
	uint32_t OUTPUT_SIZE;
	Matrix weight0;
	Matrix weight1;
	Vector bias0;
	Vector bias1;
};

WeightsFile loadWeights(std::string path) {
	std::ifstream ifile = std::ifstream(path, std::ios_base::binary);
	WeightsFile wf;
	ifile.read(reinterpret_cast<char*>(&wf), sizeof(uint32_t) * 3);
	wf.weight0 = Matrix(wf.hidden_size, wf.INPUT_SIZE);
	wf.weight1 = Matrix(wf.OUTPUT_SIZE, wf.hidden_size);
	wf.bias0 = Vector(wf.hidden_size);
	wf.bias1 = Vector(wf.OUTPUT_SIZE);
	ifile.read(reinterpret_cast<char*>(wf.weight0.data()), sizeof(float) * wf.hidden_size * wf.INPUT_SIZE);
	ifile.read(reinterpret_cast<char*>(wf.weight1.data()), sizeof(float) * wf.OUTPUT_SIZE * wf.hidden_size);
	ifile.read(reinterpret_cast<char*>(wf.bias0.data()), sizeof(float) * wf.hidden_size);
	ifile.read(reinterpret_cast<char*>(wf.bias1.data()), sizeof(float) * wf.OUTPUT_SIZE);
	ifile.close();
	return wf;
}

std::vector<uint8_t> loadPredictions(std::string path) {
	std::ifstream ifile = std::ifstream(path, std::ios_base::binary);
	std::vector<uint8_t> preds;
	while (!ifile.eof()) {
		preds.emplace_back(static_cast<uint8_t>(ifile.get()));
	}
	return preds;
}

std::vector<float> loadLossCurve(std::string path) {
	std::ifstream fs = std::ifstream(path);
	std::vector<float> out;
	std::string line;
	while (std::getline(fs, line)) {
		out.push_back(std::stof(line));
	}
	return out;
}

float predictionsAccuracy(std::vector<uint8_t> goal, std::vector<uint8_t> actual) {
	int correct = 0;
	int incorrect = 0;
	// if (goal.size() != actual.size()) { return -1.0f; }
	for (size_t i = 0; i < actual.size(); i++) {
		if (goal[i] == actual[i]) {
			correct++;
		} else {
			incorrect++;
			// std::cout << "Error: Sample " << i << " is " << actual[i] << " but is " << goal[i] << std::endl;
		}
	}
	return static_cast<float>(correct) / static_cast<float>(incorrect + correct);
}

std::tuple<float, float, float> vecAccuracy(std::vector<float> a, std::vector<float> b) {
	float min = std::numeric_limits<float>::infinity();
	float max = -std::numeric_limits<float>::infinity();
	double acc = 0;
	for (size_t i = 0; i < a.size(); i++) {
		number diff = std::abs((a[i] - b[i]) / a[i]);
		if (diff < min) {
			min = diff;
		}
		if (diff > max) {
			max = diff;
		}
		acc += diff;
	}
	return {min, acc / a.size(), max};
}

int main(int argc, char** argv) {
    TrainConfig config = parse_arguments(argc, argv);

	std::stringstream knowngoodpathbuilder;
	knowngoodpathbuilder << "output/knowngood";
	knowngoodpathbuilder << "-s" << config.random_seed;
	knowngoodpathbuilder << "-e" << config.epochs;
	knowngoodpathbuilder << "-l" << config.learning_rate;
	knowngoodpathbuilder << "-b1"; // knowngood output is always generated with batch size of 1
	knowngoodpathbuilder << "-z" << config.hidden_size;
	knowngoodpathbuilder << "/";

    std::stringstream actualpathbuilder;
    actualpathbuilder << "output/";
    actualpathbuilder << "seed" << config.random_seed;
    actualpathbuilder << "-epochs" << config.epochs;
    actualpathbuilder << "-lr" << std::fixed << std::setprecision(6) <<config.learning_rate;
    actualpathbuilder << "-batch" << config.batch_size;
    actualpathbuilder << "-hidden" << config.hidden_size;
    actualpathbuilder << "-threads" << config.threads;
    actualpathbuilder << "-tasks" << config.tasks;
    actualpathbuilder << "/";

	std::cout << "Attempting to load known good output from folder: " << knowngoodpathbuilder.str() << std::endl;
    std::cout << "Attempting to load actual output from folder: " << actualpathbuilder.str() << std::endl;

    if (!std::filesystem::is_directory(knowngoodpathbuilder.str())) {
        std::cout << knowngoodpathbuilder.str() << " is not a directory. Exiting." << std::endl;
        return 1;
    }
    if (!std::filesystem::is_directory(actualpathbuilder.str())) {
        std::cout << actualpathbuilder.str() << " is not a directory. Exiting." << std::endl;
        return 1;
    }

	WeightsFile goalWeightsFile = loadWeights(knowngoodpathbuilder.str() + "weights.bin");
	auto goalPredictions = loadPredictions(knowngoodpathbuilder.str() + "predictions.bin");
	auto goalLossCurve = loadLossCurve(knowngoodpathbuilder.str() + "losses.txt");
	

	WeightsFile actualWeightsFile = loadWeights(actualpathbuilder.str() + "weights.bin");
    auto actualPredictions = loadPredictions(actualpathbuilder.str() + "predictions.bin");
    auto actualLossCurve = loadLossCurve(actualpathbuilder.str() + "losses.txt");

	float predictionAccuracy = predictionsAccuracy(goalPredictions, actualPredictions);
	auto [lossaccuracymin, lossaccuracyavg, lossaccuracymax] = vecAccuracy(goalLossCurve, actualLossCurve);

	std::stringstream msg;

	msg << "=== Output Accuracies ===" << "\n";
	msg << "Prediction Accuracy: " << predictionAccuracy * 100 << "%\n";
	msg << "Loss Curve Accuracy:\n";
	msg << "\tMinimum Error: " << lossaccuracymin * 100 << "%\n";
	msg << "\tMaximum Error: " << lossaccuracymax * 100 << "%\n";
	msg << "\tAverage Error: " << lossaccuracyavg * 100 << "%\n";
	std::cout << msg.str();
	return 0;
}