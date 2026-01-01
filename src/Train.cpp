#include "Train.h"
#include "Matrix.h"
#include "Vector.h"
#include "CompileConfig.h"

#include <cstddef>
#include <iostream>
#include <oneapi/tbb.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_MPI
void train_model_mini_batch_mpi(Network& model, const std::vector<Sample>& data, const TrainConfig& config, std::vector<float>* loss_curve_out) {
	int mpi_proc, mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_proc);

	for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
		Vector cse_epoch(10);
		size_t num_batches = std::ceil(static_cast<float>(data.size()) / static_cast<float>(config.batch_size));
		for (size_t batch = 0; batch < num_batches; batch++) {
			
			// Calculate the start, end and actual size of this batch
			size_t batch_start = config.batch_size * batch;
			size_t batch_end = std::min(batch_start + config.batch_size, data.size());
			size_t batch_size = batch_end - batch_start;

			// Calculate the start, end and actual size of the workload of this process
			size_t proc_size = std::ceil(static_cast<float>(batch_size) / static_cast<float>(mpi_size));
			size_t proc_start = batch_start + proc_size * mpi_proc;
			size_t proc_end = std::min(proc_start + proc_size, batch_end);

			Network::TrainResult batch_result = {
			    Matrix(config.hidden_size, Network::INPUT_SIZE, 0.0f),
			    Matrix(Network::OUTPUT_SIZE, config.hidden_size, 0.0f),
			    Vector(config.hidden_size, 0.0f),
			    Vector(Network::OUTPUT_SIZE, 0.0f),
			    Vector(10)
			};

			Network::TrainResult this_proc_scratch_space = {
			    Matrix(config.hidden_size, Network::INPUT_SIZE, 0.0f),
			    Matrix(Network::OUTPUT_SIZE, config.hidden_size, 0.0f),
			    Vector(config.hidden_size, 0.0f),
			    Vector(Network::OUTPUT_SIZE, 0.0f),
			    Vector(10)
			};

			for (size_t i = proc_start; i < proc_end; i++) {
				Vector sample_data = Vector(data[i].pixels);
				int sample_label = data[i].label;
				Vector label_one_hot(10, 0.0f);
				label_one_hot(sample_label) = 1.0;

				Network::train(sample_data, label_one_hot, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), config.use_avx, this_proc_scratch_space);
				mat_plus_mat(batch_result.weight1_grad, this_proc_scratch_space.weight1_grad, batch_result.weight1_grad, config.use_avx);
				mat_plus_mat(batch_result.weight2_grad, this_proc_scratch_space.weight2_grad, batch_result.weight2_grad, config.use_avx);
				vec_plus_vec(batch_result.bias1_grad, this_proc_scratch_space.bias1_grad, batch_result.bias1_grad, config.use_avx);
				vec_plus_vec(batch_result.bias2_grad, this_proc_scratch_space.bias2_grad, batch_result.bias2_grad, config.use_avx);
				vec_plus_vec(batch_result.cse_delta, this_proc_scratch_space.cse_delta, batch_result.cse_delta, config.use_avx);
			}

			MPI_Request reducerequests[5];
			MPI_Iallreduce(MPI_IN_PLACE, batch_result.weight1_grad.data(), batch_result.weight1_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[0]);
			MPI_Iallreduce(MPI_IN_PLACE, batch_result.weight2_grad.data(), batch_result.weight2_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[1]);
			MPI_Iallreduce(MPI_IN_PLACE, batch_result.bias1_grad.data(), batch_result.bias1_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[2]);
			MPI_Iallreduce(MPI_IN_PLACE, batch_result.bias2_grad.data(), batch_result.bias2_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[3]);
			MPI_Iallreduce(MPI_IN_PLACE, batch_result.cse_delta.data(), batch_result.cse_delta.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[4]);
			MPI_Waitall(5, reducerequests, MPI_STATUSES_IGNORE);
			
			vec_plus_vec(cse_epoch, batch_result.cse_delta, cse_epoch, config.use_avx);
			model.update(config.learning_rate, batch_result);
		}
		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);

		// Print average cross-entropy loss for the epoch
		if (mpi_proc == 0) {
			std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
			float total_loss = 0.0;
			for (const auto& loss : cse_epoch) {
				total_loss += loss;
			}
			if (loss_curve_out) loss_curve_out->push_back(total_loss);
			std::cout << total_loss << std::endl;
		}
	}
}
#endif
class ParallelTrainBody {
	const Sample* samples;
	const Matrix* weight1;
	const Matrix* weight2;
	const Vector* bias1;
	const Vector* bias2;
	const size_t _hidden_size;
	const bool useAVX;

	Network::TrainResult tr;

  public:
	Network::TrainResult result;

	void operator()(const tbb::blocked_range<size_t>& r) {
		for (auto i = r.begin(); i != r.end(); i++) {
			Vector sample_data = Vector(samples[i].pixels);
			int sample_label = samples[i].label;
			Vector label_one_hot(10, 0.0f);
			label_one_hot(sample_label) = 1.0;

			Network::train(sample_data, label_one_hot, weight1, weight2, bias1, bias2, useAVX ,tr);
			mat_plus_mat(result.weight1_grad, tr.weight1_grad, result.weight1_grad, useAVX);
			mat_plus_mat(result.weight2_grad, tr.weight2_grad, result.weight2_grad, useAVX);
			vec_plus_vec(result.bias1_grad, tr.bias1_grad, result.bias1_grad, useAVX);
			vec_plus_vec(result.bias2_grad, tr.bias2_grad, result.bias2_grad, useAVX);
			vec_plus_vec(result.cse_delta, tr.cse_delta, result.cse_delta, useAVX);
		}
	}

	ParallelTrainBody(ParallelTrainBody& other, tbb::split)
	    : samples(other.samples),
	      weight1(other.weight1),
	      weight2(other.weight2),
	      bias1(other.bias1),
	      bias2(other.bias2),
	      _hidden_size(other._hidden_size),
		  useAVX(other.useAVX),
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
		mat_plus_mat(result.weight2_grad, const_cast<Matrix&>(other.result.weight2_grad), result.weight2_grad, useAVX);
		mat_plus_mat(result.weight1_grad, const_cast<Matrix&>(other.result.weight1_grad), result.weight1_grad, useAVX);
		vec_plus_vec(result.bias1_grad, const_cast<Vector&>(other.result.bias1_grad), result.bias1_grad, useAVX);
		vec_plus_vec(result.bias2_grad, const_cast<Vector&>(other.result.bias2_grad), result.bias2_grad, useAVX);
		vec_plus_vec(result.cse_delta, const_cast<Vector&>(other.result.cse_delta), result.cse_delta, useAVX);
	}

	ParallelTrainBody(const Sample* samples, const Matrix* weight1, const Matrix* weight2, const Vector* bias1, const Vector* bias2, const size_t hidden_size, const bool useAVX)
	    : samples(samples),
	      weight1(weight1),
	      weight2(weight2),
	      bias1(bias1),
	      bias2(bias2),
	      _hidden_size(hidden_size),
		  useAVX(useAVX),
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
void train_model_mini_batch_tbb(Network& model, const std::vector<Sample>& data, const TrainConfig& config, std::vector<float>* loss_curve_out) {
	for (unsigned int epoch = 0; epoch < config.epochs; ++epoch) {
		Vector cse_epoch(10, 0.0f); // cross-entropy loss for the epoch

		size_t batches = std::ceil(static_cast<float>(data.size()) / static_cast<float>(config.batch_size));

		for (size_t batch = 0; batch < batches; batch++) {
			size_t batch_begin = batch * config.batch_size;
			size_t batch_end = std::min(batch_begin + config.batch_size, data.size());

			const Sample* samples_ptr = data.data();

			ParallelTrainBody ptb(samples_ptr, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), config.hidden_size, config.use_avx);
			tbb::parallel_reduce(tbb::blocked_range<size_t>(batch_begin, batch_end, config.grainsize), ptb);

			model.update(config.learning_rate, ptb.result);
			vec_plus_vec(cse_epoch, ptb.result.cse_delta, cse_epoch, config.use_avx);
		}

		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);

		// Print average cross-entropy loss for the epoch
		std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
		float total_loss = 0.0;
		for (const auto& loss : cse_epoch) {
			total_loss += loss;
		}
		if (loss_curve_out) loss_curve_out->push_back(total_loss);
		std::cout << total_loss << std::endl;
	}
}
#ifdef USE_MPI
void train_model_mini_batch_mpi_tbb(Network& model, const std::vector<Sample>& data, const TrainConfig& config, std::vector<float>* loss_curve_out) {
	int mpi_proc, mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_proc);

	for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
		Vector cse_epoch(10);
		size_t num_batches = std::ceil(static_cast<float>(data.size()) / static_cast<float>(config.batch_size));
		for (size_t batch = 0; batch < num_batches; batch++) {
			
			// Calculate the start, end and actual size of this batch
			size_t batch_start = config.batch_size * batch;
			size_t batch_end = std::min(batch_start + config.batch_size, data.size());
			size_t batch_size = batch_end - batch_start;

			// Calculate the start, end and actual size of the workload of this process
			size_t proc_size = std::ceil(static_cast<float>(batch_size) / static_cast<float>(mpi_size));
			size_t proc_start = batch_start + proc_size * mpi_proc;
			size_t proc_end = std::min(proc_start + proc_size, batch_end);

			Network::TrainResult batch_result = {
			    Matrix(config.hidden_size, Network::INPUT_SIZE, 0.0f),
			    Matrix(Network::OUTPUT_SIZE, config.hidden_size, 0.0f),
			    Vector(config.hidden_size, 0.0f),
			    Vector(Network::OUTPUT_SIZE, 0.0f),
			    Vector(10)
			};

			Network::TrainResult this_proc_scratch_space = {
			    Matrix(config.hidden_size, Network::INPUT_SIZE, 0.0f),
			    Matrix(Network::OUTPUT_SIZE, config.hidden_size, 0.0f),
			    Vector(config.hidden_size, 0.0f),
			    Vector(Network::OUTPUT_SIZE, 0.0f),
			    Vector(10)
			};

			const Sample* samples_ptr = data.data();

			ParallelTrainBody ptb(samples_ptr, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), config.hidden_size, config.use_avx);
			tbb::parallel_reduce(tbb::blocked_range<size_t>(proc_start, proc_end, config.grainsize), ptb);

			MPI_Request reducerequests[5];
			MPI_Iallreduce(ptb.result.weight1_grad.data(), batch_result.weight1_grad.data(), batch_result.weight1_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[0]);
			MPI_Iallreduce(ptb.result.weight2_grad.data(), batch_result.weight2_grad.data(), batch_result.weight2_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[1]);
			MPI_Iallreduce(ptb.result.bias1_grad.data(), batch_result.bias1_grad.data(), batch_result.bias1_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[2]);
			MPI_Iallreduce(ptb.result.bias2_grad.data(), batch_result.bias2_grad.data(), batch_result.bias2_grad.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[3]);
			MPI_Iallreduce(ptb.result.cse_delta.data(), batch_result.cse_delta.data(), batch_result.cse_delta.size(), MPI_NUMBER, MPI_SUM, MPI_COMM_WORLD, &reducerequests[4]);
			MPI_Waitall(5, reducerequests, MPI_STATUSES_IGNORE);
			
			vec_plus_vec(cse_epoch, batch_result.cse_delta, cse_epoch, config.use_avx);
			model.update(config.learning_rate, batch_result);
		}
		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);

		// Print average cross-entropy loss for the epoch
		if (mpi_proc == 0) {
			std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
			float total_loss = 0.0;
			for (const auto& loss : cse_epoch) {
				total_loss += loss;
			}
			if (loss_curve_out) loss_curve_out->push_back(total_loss);
			std::cout << total_loss << std::endl;
		}
	}
}
#endif
void train_model_mini_batch(Network& model, const std::vector<Sample>& data, const TrainConfig& config, std::vector<float>* loss_curve_out) {
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

				Network::train(sample_data, label_one_hot, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), config.use_avx, tr);
				mat_plus_mat(result.weight1_grad, tr.weight1_grad, result.weight1_grad, config.use_avx);
				mat_plus_mat(result.weight2_grad, tr.weight2_grad, result.weight2_grad, config.use_avx);
				vec_plus_vec(result.bias1_grad, tr.bias1_grad, result.bias1_grad, config.use_avx);
				vec_plus_vec(result.bias2_grad, tr.bias2_grad, result.bias2_grad, config.use_avx);
				vec_plus_vec(result.cse_delta, tr.cse_delta, result.cse_delta, config.use_avx);
				// std::cout << i << " : " << printVector(result.cse_delta) << "\n";
			}

			vec_plus_vec(cse_epoch, result.cse_delta, cse_epoch, config.use_avx);
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
		if (loss_curve_out) loss_curve_out->push_back(total_loss);
		std::cout << total_loss << std::endl;
	}
}
void train_model_stochastic(Network& model, const std::vector<Sample>& data, const TrainConfig& config, std::vector<float>* loss_curve_out) {
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

			Network::train(sample_data, label_one_hot, &model.get_weight(1), &model.get_weight(2), &model.get_bias(1), &model.get_bias(2), config.use_avx, tr);
			model.update(config.learning_rate, tr);

			vec_plus_vec(cse_epoch, tr.cse_delta, cse_epoch, config.use_avx);
		}
		Vector data_size_broadcasted = Vector(10, data.size());
		divide_elementwise_vec(cse_epoch, data_size_broadcasted, cse_epoch);

		// Print average cross-entropy loss for the epoch
		std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Cross-Entropy Loss: ";
		float total_loss = 0.0;
		for (const auto& loss : cse_epoch) {
			total_loss += loss;
		}
		if (loss_curve_out)	loss_curve_out->push_back(total_loss);
		std::cout << total_loss << std::endl;
	}
}
void train_model(Network& model, const std::vector<Sample>& data, const TrainConfig& config, std::vector<float>* loss_curve_out) {
	constexpr char MPI_ENABLED = 0b100;
	constexpr char TBB_ENABLED = 0b010;
	constexpr char BATCHING_ENABLED = 0b001;
	constexpr char MPI_DISABLED = 0, TBB_DISABLED = 0, BATCHING_DISABLED = 0;

	unsigned char features = (config.tasks > 1) << 2 | (config.threads > 1) << 1 | (config.batch_size > 1);

	switch (features)
	{
		case MPI_DISABLED | TBB_DISABLED | BATCHING_DISABLED: 
			train_model_stochastic(model, data, config, loss_curve_out); 
			break;
		case MPI_DISABLED | TBB_DISABLED | BATCHING_ENABLED: 
			train_model_mini_batch(model, data, config, loss_curve_out); 
			break;
		case MPI_DISABLED | TBB_ENABLED  | BATCHING_DISABLED: 
			train_model_stochastic(model, data, config, loss_curve_out); 
			break;
		case MPI_DISABLED | TBB_ENABLED  | BATCHING_ENABLED: 
			train_model_mini_batch_tbb(model, data, config, loss_curve_out); 
			break;
		case MPI_ENABLED  | TBB_DISABLED | BATCHING_DISABLED: 
			break;
		case MPI_ENABLED  | TBB_DISABLED | BATCHING_ENABLED: 
			#ifdef USE_MPI
			train_model_mini_batch_mpi(model, data, config, loss_curve_out);
			#else
			train_model_mini_batch(model, data, config, loss_curve_out);
			#endif
			break;
		case MPI_ENABLED  | TBB_ENABLED  | BATCHING_DISABLED: 
			break;
		case MPI_ENABLED  | TBB_ENABLED  | BATCHING_ENABLED: 
			#ifdef USE_MPI
			train_model_mini_batch_mpi_tbb(model, data, config, loss_curve_out); 
			#else
			train_model_mini_batch_tbb(model, data, config, loss_curve_out);
			#endif
			break;
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

std::vector<int> get_predictions(Network& model, const std::vector<Sample>& data, bool useAVX) {
	std::vector<int> ret(data.size());
	for (size_t i = 0; i < data.size(); i++) {
		ret[i] = model.predict(Vector(data[i].pixels), useAVX);
	}
	return ret;
}