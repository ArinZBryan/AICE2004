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

std::tuple<float, float, float> weightsAccuracy(WeightsFile goal, WeightsFile actual, float threshold_absolute) {
    int correct = 0;
    int incorrect = 0;
    std::tuple<float, float, float> res = { 0.0f, 0.0f, 0.0f };
    if (goal.weight0.cols() != actual.weight0.cols() || goal.weight0.rows() != actual.weight0.rows()) {
        std::get<0>(res) = -1.0f;
    } else {
        int w0correct = 0;
        int w0incorrect = 0;
        for (size_t i = 0; i < actual.weight0.rows(); i++) for (size_t j = 0; j < actual.weight0.cols(); j++) {
            if (std::fabs(actual.weight0(i, j) - goal.weight0(i, j)) < threshold_absolute) { w0correct++; }
            else { w0incorrect++; }
        }
        std::get<0>(res) = static_cast<float>(w0correct) / static_cast<float>(w0correct + w0incorrect);
        correct += w0correct;
        incorrect += w0incorrect;
    }
    if (goal.weight1.cols() != actual.weight1.cols() || goal.weight1.rows() != actual.weight1.rows()) {
        std::get<1>(res) = -1.0f;
    } else {
        int w1correct = 0;
        int w1incorrect = 0;
        for (size_t i = 0; i < actual.weight1.rows(); i++) for (size_t j = 0; j < actual.weight1.cols(); j++) {
            if (std::fabs(actual.weight1(i, j) - goal.weight1(i, j)) < threshold_absolute) { w1correct++; }
            else { w1incorrect++; }
        }
        std::get<1>(res) = static_cast<float>(w1correct) / static_cast<float>(w1correct + w1incorrect);
        correct += w1correct;
        incorrect += w1incorrect;
    }
    if (std::get<0>(res) == -1.0f || std::get<1>(res) == -1.0f) {
        std::get<2>(res) = -1.0f;
    } else {
        std::get<2>(res) = static_cast<float>(correct) / static_cast<float>(correct + incorrect);
    }
    return res;
}

std::tuple<float, float, float> biasesAccuracy(WeightsFile goal, WeightsFile actual, float threshold_absolute) {
    int correct = 0;
    int incorrect = 0;
    std::tuple<float, float, float> res = { 0.0f, 0.0f, 0.0f };
    if (goal.bias0.size() != actual.bias0.size()) {
        std::get<0>(res) = -1.0f;
    } else {
        int w0correct = 0;
        int w0incorrect = 0;
        for (size_t i = 0; i < actual.bias0.size(); i++){
            if (std::fabs(actual.bias0(i) - goal.bias0(i)) < threshold_absolute) { w0correct++; }
            else { w0incorrect++; }
        }
        std::get<0>(res) = static_cast<float>(w0correct) / static_cast<float>(w0correct + w0incorrect);
        correct += w0correct;
        incorrect += w0incorrect;
    }
    if (goal.bias1.size() != actual.bias1.size()) {
        std::get<1>(res) = -1.0f;
    } else {
        int w1correct = 0;
        int w1incorrect = 0;
        for (size_t i = 0; i < actual.bias1.size(); i++){
            if (std::fabs(actual.bias1(i) - goal.bias1(i)) < threshold_absolute) { w1correct++; }
            else { w1incorrect++; }
        }
        std::get<1>(res) = static_cast<float>(w1correct) / static_cast<float>(w1correct + w1incorrect);
        correct += w1correct;
        incorrect += w1incorrect;
    }
    if (std::get<0>(res) == -1.0f || std::get<1>(res) == -1.0f) {
        std::get<2>(res) = -1.0f;
    } else {
        std::get<2>(res) = static_cast<float>(correct) / static_cast<float>(correct + incorrect);
    }
    return res;    
}

std::tuple<float, float, float> weightsAccuracyRelative(WeightsFile goal, WeightsFile actual, float threshold_relative) {
    int correct = 0;
    int incorrect = 0;
    float threshold = threshold_relative / 100;
    std::tuple<float, float, float> res = { 0.0f, 0.0f, 0.0f };
    if (goal.weight0.cols() != actual.weight0.cols() || goal.weight0.rows() != actual.weight0.rows()) {
        std::get<0>(res) = -1.0f;
    } else {
        int w0correct = 0;
        int w0incorrect = 0;
        for (size_t i = 0; i < actual.weight0.rows(); i++) for (size_t j = 0; j < actual.weight0.cols(); j++) {
            if (std::fabs(actual.weight0(i, j) - goal.weight0(i, j)) < threshold * std::fabs(goal.weight0(i, j))) { w0correct++; }
            else { w0incorrect++; }
        }
        std::get<0>(res) = static_cast<float>(w0correct) / static_cast<float>(w0correct + w0incorrect);
        correct += w0correct;
        incorrect += w0incorrect;
    }
    if (goal.weight1.cols() != actual.weight1.cols() || goal.weight1.rows() != actual.weight1.rows()) {
        std::get<1>(res) = -1.0f;
    } else {
        int w1correct = 0;
        int w1incorrect = 0;
        for (size_t i = 0; i < actual.weight1.rows(); i++) for (size_t j = 0; j < actual.weight1.cols(); j++) {
            if (std::fabs(actual.weight1(i, j) - goal.weight1(i, j)) < threshold * std::fabs(goal.weight0(i, j))) { w1correct++; }
            else { w1incorrect++; }
        }
        std::get<1>(res) = static_cast<float>(w1correct) / static_cast<float>(w1correct + w1incorrect);
        correct += w1correct;
        incorrect += w1incorrect;
    }
    if (std::get<0>(res) == -1.0f || std::get<1>(res) == -1.0f) {
        std::get<2>(res) = -1.0f;
    } else {
        std::get<2>(res) = static_cast<float>(correct) / static_cast<float>(correct + incorrect);
    }
    return res;
}

std::tuple<float, float, float> biasesAccuracyRelative(WeightsFile goal, WeightsFile actual, float threshold_relative) {
    int correct = 0;
    int incorrect = 0;
    float threshold = threshold_relative / 100;
    std::tuple<float, float, float> res = { 0.0f, 0.0f, 0.0f };
    if (goal.bias0.size() != actual.bias0.size()) {
        std::get<0>(res) = -1.0f;
    } else {
        int w0correct = 0;
        int w0incorrect = 0;
        for (size_t i = 0; i < actual.bias0.size(); i++){
            if (std::fabs(actual.bias0(i) - goal.bias0(i)) < threshold * std::fabs(goal.bias0(i))) { w0correct++; }
            else { w0incorrect++; }
        }
        std::get<0>(res) = static_cast<float>(w0correct) / static_cast<float>(w0correct + w0incorrect);
        correct += w0correct;
        incorrect += w0incorrect;
    }
    if (goal.bias1.size() != actual.bias1.size()) {
        std::get<1>(res) = -1.0f;
    } else {
        int w1correct = 0;
        int w1incorrect = 0;
        for (size_t i = 0; i < actual.bias1.size(); i++){
            if (std::fabs(actual.bias1(i) - goal.bias1(i)) < threshold * std::fabs(goal.bias1(i))) { w1correct++; }
            else { w1incorrect++; }
        }
        std::get<1>(res) = static_cast<float>(w1correct) / static_cast<float>(w1correct + w1incorrect);
        correct += w1correct;
        incorrect += w1incorrect;
    }
    if (std::get<0>(res) == -1.0f || std::get<1>(res) == -1.0f) {
        std::get<2>(res) = -1.0f;
    } else {
        std::get<2>(res) = static_cast<float>(correct) / static_cast<float>(correct + incorrect);
    }
    return res;    
}


int main() {
    constexpr float WEIGHT_EQUAL_THRESHOLD_PERCENT = 5;  //Specification requires 100% accuracy within an error margin of 5%
    constexpr float BIAS_EQUAL_THRESHOLD_PERCENT = 5;    //Specification requires 100% accuracy within an error margin of 5%
    constexpr unsigned int INPUT_SIZE = 784;
    constexpr unsigned int OUTPUT_SIZE = 10;
    constexpr TrainConfig config {
        1234,   // random_seed      [DO NOT MODIFY]
        10,     // epochs           [DO NOT MODIFY]
        0.001,  // learning_rate    [DO NOT MODIFY]
        32,     // batch_size          
        10,     // hidden_size      [DO NOT MODIFY]
        1,      // threads    (known good output only supports threads=1)
        1       // tasks      (known good output only supports tasks=1)
    };

    WeightsFile goalWeightsFile = loadWeights("output/knowngood/weights.bin");
    std::vector<uint8_t> goalPredictions = loadPredictions("output/knowngood/predictions.bin");

    // Create DataLoader instance
	DataLoader loader;
    const std::vector<Sample> &train_data = loader.get_train_data();
	const std::vector<Sample> &test_data = loader.get_test_data();

	// Initialise the network with parsed seed and hidden size
	Network net = Network(config.hidden_size, config.random_seed);

    // Train.
    train_model(net, train_data, config);
    // Print the trained prediction of data[eval_data_index].
    // Store predictions to save as file later.
	std::vector<int> predictions;
	evaluate_model_real(net, test_data, &predictions);

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
    std::tuple<float, float, float> weightAccuracy = weightsAccuracyRelative(goalWeightsFile, actualWeightsFile, WEIGHT_EQUAL_THRESHOLD_PERCENT);
    std::tuple<float, float, float> biasAccuracy = biasesAccuracyRelative(goalWeightsFile, actualWeightsFile, BIAS_EQUAL_THRESHOLD_PERCENT);

    std::stringstream msg;
    
    msg << "=== Output Accuracies ===" << "\n";
    msg << "Prediction Accuracy: " << predictionAccuracy * 100 << "%\n";
    msg << "Weight Accuracy: \n";
    msg <<      "\tHidden Layer: " << std::get<0>(weightAccuracy) * 100 << "% (within +/- " << WEIGHT_EQUAL_THRESHOLD_PERCENT << "%)\n";
    msg <<      "\tOutput Layer: " << std::get<1>(weightAccuracy) * 100 << "% (within +/- " << WEIGHT_EQUAL_THRESHOLD_PERCENT << "%)\n";
    msg <<      "\tOverall: " << std::get<2>(weightAccuracy) * 100 << "% (within +/- " << WEIGHT_EQUAL_THRESHOLD_PERCENT << "%)\n";
    msg << "Bias Accuracy: \n";
    msg <<      "\tHidden Layer: " << std::get<0>(biasAccuracy) * 100 << "% (within +/- " << BIAS_EQUAL_THRESHOLD_PERCENT << "%)\n";
    msg <<      "\tOutput Layer: " << std::get<1>(biasAccuracy) * 100 << "% (within +/- " << BIAS_EQUAL_THRESHOLD_PERCENT << "%)\n";
    msg <<      "\tOverall: " << std::get<2>(biasAccuracy) * 100 << "% (within +/- " << BIAS_EQUAL_THRESHOLD_PERCENT << "%)\n";

    std::cout << msg.str();
    return 0;
}