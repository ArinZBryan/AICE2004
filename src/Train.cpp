#include "Train.h"

void train_model(Network& model, vector<Sample> data, const TrainConfig& config) {
	for (unsigned int epoch = 0; epoch < config.epochs; ++epoch) {
		vector<float> cse_epoch(10, 0.0f); // cross-entropy loss for the epoch
		for (unsigned int i = 0; i < data.size(); i++) {
			vector<float> sample_data = data[i].pixels;
			int sample_label = data[i].label;
			vector<float> label_one_hot(10, 0.0f);
			label_one_hot[sample_label] = 1.0;

			vector<float> output = model.forward(sample_data);

			vector<float> cse = cross_entropy_loss(output, label_one_hot);
			for (unsigned j = 0; j < cse.size(); j++) {
				cse_epoch[j] += cse[j];
			}

			// Backpropagate using the model's method
			model.backpropagate(label_one_hot, config.learning_rate);
		}
		for (unsigned j = 0; j < cse_epoch.size(); j++) {
			cse_epoch[j] /= data.size();
		}
		// Print average cross-entropy loss for the epoch
		std::cout << "Epoch " << epoch + 1 << "/" << config.epochs
		          << " - Cross-Entropy Loss: ";
		float total_loss = 0.0;
		for (const auto& loss : cse_epoch) {
			total_loss += loss;
		}

		std::cout << total_loss << std::endl;
	}
}

void evaluate_model(Network& model, const vector<Sample>& data, std::vector<int>* predictions) {
	int correct = 0;
	if (predictions)
		predictions->clear();
	for (const auto& sample : data) {
		int predicted = model.predict(sample.pixels);
		bool is_correct = (predicted == sample.label);
		if (is_correct)
			correct++;
		if (predictions)
			predictions->push_back(is_correct ? 1 : 0);
	}
	float accuracy = static_cast<float>(correct) / data.size();
	std::cout << "Evaluation Accuracy: " << accuracy * 100.0f << "%" << std::endl;
}
