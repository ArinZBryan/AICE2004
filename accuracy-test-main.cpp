#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <string>

#include <cstdint>
#include <cmath>

#include "Matrix.h"
#include "Vector.h"
#include "Maths.h"
#include "Network.h"
#include "Train.h"

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
    //if (goal.size() != actual.size()) { return -1.0f; }
    for (size_t i = 0; i < actual.size(); i++) {
        if (goal[i] == actual[i]) { correct++; }
        else { 
            incorrect++; 
            //std::cout << "Error: Sample " << i << " is " << actual[i] << " but is " << goal[i] << std::endl;
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
        if (diff < min) { min = diff; }
        if (diff > max) { max = diff; }
        acc += diff;
    }
    return {min, acc / a.size(), max};
}

int main() {
    constexpr unsigned int INPUT_SIZE = 784;
    constexpr unsigned int OUTPUT_SIZE = 10;
    constexpr TrainConfig config {
        1234,   // random_seed      [DO NOT MODIFY]
        10,     // epochs           [DO NOT MODIFY]
        0.001,  // learning_rate    [DO NOT MODIFY]
        1,     // batch_size          
        10,     // hidden_size      [DO NOT MODIFY]
        1,      // threads    (known good output only supports threads=1)
        1       // tasks      (known good output only supports tasks=1)
    };

    std::stringstream knowngoodpathbuilder;
    knowngoodpathbuilder << "output/knowngood";
    knowngoodpathbuilder << "-s" << config.random_seed;
    knowngoodpathbuilder << "-e" << config.epochs;
    knowngoodpathbuilder << "-l" << config.learning_rate;
    knowngoodpathbuilder << "-b" << config.batch_size;
    knowngoodpathbuilder << "-z" << config.hidden_size;
    knowngoodpathbuilder << "/";

    std::cout << "Loading known good output from folder: " << knowngoodpathbuilder.str() << std::endl;

    WeightsFile goalWeightsFile = loadWeights(knowngoodpathbuilder.str() + "weights.bin");
    std::vector<uint8_t> goalPredictions = loadPredictions(knowngoodpathbuilder.str() + "predictions.bin");
    std::vector<float> goalLossCurve = loadLossCurve(knowngoodpathbuilder.str() + "losses.txt");

    // Create DataLoader instance
	DataLoader loader;
    const std::vector<Sample> &train_data = loader.get_train_data();
	const std::vector<Sample> &test_data = loader.get_test_data();
    

	// Initialise the network with parsed seed and hidden size
	Network net = Network(config.hidden_size, config.random_seed);

    // Train.
    std::vector<float> actualLossCurve = train_model(net, train_data, config);
    // Print the trained prediction of data[eval_data_index].
    // Store predictions to save as file later.
	std::vector<int> predictions;
	evaluate_model(net, test_data, &predictions);

    std::vector<uint8_t> actualPredictions = std::vector<uint8_t>();
    actualPredictions.reserve(predictions.size());
    for (int p : predictions) {
        actualPredictions.emplace_back(static_cast<uint8_t>(p));
    }

    std::vector<Matrix> actualWeights = net.get_weights();
    std::vector<Vector> actualBiases = net.get_bias();
    WeightsFile actualWeightsFile = {
        INPUT_SIZE,
        config.hidden_size,
        OUTPUT_SIZE,
        actualWeights[0],
        actualWeights[1],
        actualBiases[0],
        actualBiases[1]
    };
    
    float predictionAccuracy = predictionsAccuracy(goalPredictions, actualPredictions);
    auto [ lossaccuracymin, lossaccuracyavg, lossaccuracymax ] = vecAccuracy(goalLossCurve, actualLossCurve);

    std::stringstream msg;
    
    msg << "=== Output Accuracies ===" << "\n";
    msg << "Prediction Accuracy: " << predictionAccuracy * 100 << "%\n";
    msg << "Loss Curve Accuracy:\n";
    msg <<      "\tMinimum Error: " << lossaccuracymin * 100 << "%\n";
    msg <<      "\tMaximum Error: " << lossaccuracymax * 100 << "%\n";
    msg <<      "\tAverage Error: " << lossaccuracyavg * 100 << "%\n";
    std::cout << msg.str();
    return 0;
}