#include "Train.h"
#include "Matrix.h"
#include "Vector.h"

#include <cstddef>
#include <oneapi/tbb.h>

class ParallelTrainBody {
	const Sample* samples;
	const Matrix* weight1;
	const Matrix* weight2;
	const Vector* bias1;
	const Vector* bias2;
	const size_t _hidden_size;

	Network::TrainResult tr;

  public:
	Network::TrainResult result;

	void operator()(const tbb::blocked_range<size_t>& r) {
		for (auto i = r.begin(); i != r.end(); i++) {
			Vector sample_data = Vector(samples[i].pixels);
			int sample_label = samples[i].label;
			Vector label_one_hot(10, 0.0f);
			label_one_hot(sample_label) = 1.0;

			Network::train(sample_data, label_one_hot, weight1, weight2, bias1, bias2, tr);
			mat_plus_mat(result.weight1_grad, tr.weight1_grad, result.weight1_grad);
			mat_plus_mat(result.weight2_grad, tr.weight2_grad, result.weight2_grad);
			vec_plus_vec(result.bias1_grad, tr.bias1_grad, result.bias1_grad);
			vec_plus_vec(result.bias2_grad, tr.bias2_grad, result.bias2_grad);
			vec_plus_vec(result.cse_delta, tr.cse_delta, result.cse_delta);
		}
	}

	ParallelTrainBody(ParallelTrainBody& other, tbb::split)
	    : samples(other.samples),
	      weight1(other.weight1),
	      weight2(other.weight2),
	      bias1(other.bias1),
	      bias2(other.bias2),
	      _hidden_size(other._hidden_size),
	      tr{
	          Matrix(other._hidden_size, Network::INPUT_SIZE, 0.0f),
	          Matrix(Network::OUTPUT_SIZE, other._hidden_size, 0.0f),
	          Vector(other._hidden_size, 0.0f),
	          Vector(Network::OUTPUT_SIZE, 0.0f),
	          Vector(10)},
	      result{
	          Matrix(other._hidden_size, Network::INPUT_SIZE, 0.0f),
	          Matrix(Network::OUTPUT_SIZE, other._hidden_size, 0.0f),
	          Vector(other._hidden_size, 0.0f),
	          Vector(Network::OUTPUT_SIZE, 0.0f),
	          Vector(10)} {}

	void join(const ParallelTrainBody& other) {
		mat_plus_mat(result.weight1_grad, const_cast<Matrix&>(other.result.weight1_grad), result.weight1_grad);
		mat_plus_mat(result.weight2_grad, const_cast<Matrix&>(other.result.weight2_grad), result.weight2_grad);
		vec_plus_vec(result.bias1_grad, const_cast<Vector&>(other.result.bias1_grad), result.bias1_grad);
		vec_plus_vec(result.bias2_grad, const_cast<Vector&>(other.result.bias2_grad), result.bias2_grad);
		vec_plus_vec(result.cse_delta, const_cast<Vector&>(other.result.cse_delta), result.cse_delta);
	}

	ParallelTrainBody(const std::vector<Sample> samples, const Matrix* weight1, const Matrix* weight2, const Vector* bias1, const Vector* bias2, const size_t hidden_size)
	    : samples(samples.data()),
	      weight1(weight1),
	      weight2(weight2),
	      bias1(bias1),
	      bias2(bias2),
	      _hidden_size(hidden_size),
	      tr{
	          Matrix(hidden_size, Network::INPUT_SIZE, 0.0f),
	          Matrix(Network::OUTPUT_SIZE, hidden_size, 0.0f),
	          Vector(hidden_size, 0.0f),
	          Vector(Network::OUTPUT_SIZE, 0.0f),
	          Vector(10)},
	      result{
	          Matrix(hidden_size, Network::INPUT_SIZE, 0.0f),
	          Matrix(Network::OUTPUT_SIZE, hidden_size, 0.0f),
	          Vector(hidden_size, 0.0f),
	          Vector(Network::OUTPUT_SIZE, 0.0f),
	          Vector(10)} {}
};
std::vector<float> train_model_mini_batch_parallel(Network& model, const std::vector<Sample>& data, const TrainConfig& config) {
	std::vector<float> loss_curve;
	for (unsigned int epoch = 0; epoch < config.epochs; ++epoch) {
		Vector cse_epoch(10, 0.0f); // cross-entropy loss for the epoch

		size_t batches = std::ceil(static_cast<float>(data.size()) / static_cast<float>(config.batch_size));

		for (size_t batch = 0; batch < batches; batch++) {
			size_t batch_begin = batch * config.batch_size;
			size_t batch_end = std::min(batch_begin + config.batch_size, data.size());

			ParallelTrainBody ptb(data, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), config.hidden_size);
			tbb::parallel_reduce(tbb::blocked_range<size_t>(batch_begin, batch_end), ptb);

			model.update(config.learning_rate, ptb.result);
			vec_plus_vec(cse_epoch, ptb.result.cse_delta, cse_epoch);
		}

		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);

		// Print average cross-entropy loss for the epoch
		std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
		float total_loss = 0.0;
		for (const auto& loss : cse_epoch) {
			total_loss += loss;
		}
		loss_curve.push_back(total_loss);
		std::cout << total_loss << std::endl;
	}
	return loss_curve;
}
std::vector<float> train_model_mini_batch(Network& model, const std::vector<Sample>& data, const TrainConfig& config) {
	std::vector<float> loss_curve;
	for (unsigned int epoch = 0; epoch < config.epochs; ++epoch) {
		Vector cse_epoch(10, 0.0f); // cross-entropy loss for the epoch

		size_t batches = std::ceil(static_cast<float>(data.size()) / static_cast<float>(config.batch_size));

		for (size_t batch = 0; batch < batches; batch++) {

			Network::TrainResult result = {
			    Matrix(config.hidden_size, Network::INPUT_SIZE, 0.0f),
			    Matrix(Network::OUTPUT_SIZE, config.hidden_size, 0.0f),
			    Vector(config.hidden_size, 0.0f),
			    Vector(Network::OUTPUT_SIZE, 0.0f),
			    Vector(10)};

			Network::TrainResult tr = {
			    Matrix(config.hidden_size, Network::INPUT_SIZE, 0.0f),
			    Matrix(Network::OUTPUT_SIZE, config.hidden_size, 0.0f),
			    Vector(config.hidden_size, 0.0f),
			    Vector(Network::OUTPUT_SIZE, 0.0f),
			    Vector(10)};

			size_t batch_begin = batch * config.batch_size;
			size_t batch_end = std::min(batch_begin + config.batch_size, data.size());

			for (size_t i = batch_begin; i < batch_end; i++) {
				Vector sample_data = Vector(data[i].pixels);
				int sample_label = data[i].label;
				Vector label_one_hot(10, 0.0f);
				label_one_hot(sample_label) = 1.0;

				Network::train(sample_data, label_one_hot, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), tr);
				mat_plus_mat(result.weight1_grad, tr.weight1_grad, result.weight1_grad);
				mat_plus_mat(result.weight2_grad, tr.weight2_grad, result.weight2_grad);
				vec_plus_vec(result.bias1_grad, tr.bias1_grad, result.bias1_grad);
				vec_plus_vec(result.bias2_grad, tr.bias2_grad, result.bias2_grad);
				vec_plus_vec(result.cse_delta, tr.cse_delta, result.cse_delta);
				// std::cout << i << " : " << printVector(result.cse_delta) << "\n";
			}

			vec_plus_vec(cse_epoch, result.cse_delta, cse_epoch);
			model.update(config.learning_rate, result);
		}

		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);

		// Print average cross-entropy loss for the epoch
		std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
		float total_loss = 0.0;
		for (const auto& loss : cse_epoch) {
			total_loss += loss;
		}
		loss_curve.push_back(total_loss);
		std::cout << total_loss << std::endl;
	}
	return loss_curve;
}
std::vector<float> train_model_stochastic(Network& model, std::vector<Sample> data, const TrainConfig& config) {
	std::vector<float> loss_curve;
	for (unsigned int epoch = 0; epoch < config.epochs; ++epoch) {
		Vector cse_epoch(10, 0.0f); // cross-entropy loss for the epoch

		Network::TrainResult tr = {
		    Matrix(config.hidden_size, Network::INPUT_SIZE, 0.0f),
		    Matrix(Network::OUTPUT_SIZE, config.hidden_size, 0.0f),
		    Vector(config.hidden_size, 0.0f),
		    Vector(Network::OUTPUT_SIZE, 0.0f),
		    Vector(10)};

		for (unsigned int i = 0; i < data.size(); i++) {
			Vector sample_data = Vector(data[i].pixels);
			int sample_label = data[i].label;
			Vector label_one_hot(10, 0.0f);
			label_one_hot(sample_label) = 1.0;

			Network::train(sample_data, label_one_hot, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), tr);
			model.update(config.learning_rate, tr);

			vec_plus_vec(cse_epoch, tr.cse_delta, cse_epoch);
		}
		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);

		// Print average cross-entropy loss for the epoch
		std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
		float total_loss = 0.0;
		for (const auto& loss : cse_epoch) {
			total_loss += loss;
		}
		loss_curve.push_back(total_loss);
		std::cout << total_loss << std::endl;
	}
	return loss_curve;
}
std::vector<float> train_model(Network& model, const std::vector<Sample>& data, const TrainConfig& config) {
	if (config.batch_size == 1) {
		return train_model_stochastic(model, data, config);
	} else if (config.threads > 1) {
		return train_model_mini_batch_parallel(model, data, config);
	} else {
		return train_model_mini_batch(model, data, config);
	}
}

float evaluate_predictions(const std::vector<Sample>& data, std::vector<int>& predictions) {
	int correct = 0;
	for (size_t i = 0; i < data.size(); i++) {
		int predicted = predictions[i];
		bool is_correct = (predicted == data[i].label);
		if (is_correct)
			correct++;
		predictions[i] = (is_correct ? 1 : 0);
	}
	return static_cast<float>(correct) / data.size();
}

std::vector<int> get_predictions(Network& model, const std::vector<Sample>& data) {
	std::vector<int> ret(data.size());
	for (size_t i = 0; i < data.size(); i++) {
		ret[i] = model.predict(Vector(data[i].pixels));
	}
	return ret;
}