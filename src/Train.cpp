#include "Train.h"
#include "Vector.h"
#include "Matrix.h"

std::vector<float> train_model(Network &model, std::vector<Sample> data, const TrainConfig &config) {
	std::vector<float> loss_curve;
	for (unsigned int epoch = 0; epoch < config.epochs; ++epoch) {
		Vector cse_epoch(10, 0.0f); // cross-entropy loss for the epoch
		for (unsigned int i = 0; i < data.size(); i++) {
			Vector sample_data = Vector(data[i].pixels);
			int sample_label = data[i].label;
			Vector label_one_hot(10, 0.0f);
			label_one_hot(sample_label) = 1.0;

			Network::TrainResult tr = Network::train(sample_data, label_one_hot, model.get_weights(), model.get_bias());
			model.update(config.learning_rate, tr);

			vec_plus_vec(cse_epoch, tr.cse_delta, cse_epoch);
		}
		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);
		
		// Print average cross-entropy loss for the epoch
		std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
        float total_loss = 0.0;
		for (const auto &loss : cse_epoch) { total_loss += loss; }
		loss_curve.push_back(total_loss);
		std::cout << total_loss << std::endl;
	}
	return loss_curve;
}

void evaluate_model(Network &model, const std::vector<Sample> &data, std::vector<int> *predictions) {
	int correct = 0;
	if (predictions)
		predictions->clear();
	for (const auto &sample : data) {
		int predicted = model.predict(Vector(sample.pixels));
		bool is_correct = (predicted == sample.label);
		if (is_correct)
			correct++;
		if (predictions)
			predictions->push_back(is_correct ? 1 : 0);
	}
	float accuracy = static_cast<float>(correct) / data.size();
	std::cout << "Evaluation Accuracy: " << accuracy * 100.0f << "%" << std::endl;
}

void evaluate_model_real(Network &model, const std::vector<Sample> &data, std::vector<int> *predictions) {
	if (predictions)
		predictions->clear();
	for (const auto &sample : data) {
		int predicted = model.predict(Vector(sample.pixels));
		if (predictions)
			predictions->push_back(predicted);
	}
}