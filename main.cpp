#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

#include "Arguments.h"
#include "DataLoader.h"
#include "Network.h"
#include "Train.h"

#include <iostream>
#include <string>
#include <variant>

#include <oneapi/tbb.h>

using namespace std;

std::variant<TrainConfig, int> configure(int argc, char* argv[]) {
	try {
		TrainConfig tc = parse_arguments(argc, argv);

		oneapi::tbb::global_control tbb_global_ctrl(oneapi::tbb::global_control::max_allowed_parallelism, tc.threads);

		std::cout << "[Configuration]\n";
		std::cout << "Random Seed:\t\t" << tc.random_seed << "\n";
		std::cout << "Learning Rate:\t\t" << tc.learning_rate << "\n";
		std::cout << "Epochs:\t\t\t" << tc.epochs << "\n";
		std::cout << "Batch Size:\t\t" << tc.batch_size << "\n";
		std::cout << "Hidden Layer Size:\t" << tc.hidden_size << "\n";
		std::cout << "Threads:\t\t" << tc.threads << "\n";
		std::cout << "Tasks:\t\t\t" << tc.tasks << "\n";
		if (tc.batch_size == 1) {
			std::cout << "Training Method:\t" << "Stochastic" << "\n";
		} else if (tc.threads > 1) { 
			std::cout << "Training Method:\t" << "Mini Batch (Multi-Threaded)" << "\n";
		} else {
			std::cout << "Training Method:\t" << "Mini Batch (Single-Threaded)" << "\n";
		}
		std::cout << std::endl;
		return tc;
	} catch (const std::logic_error&) {
		// Help requested
		return 1;
	} catch (const std::exception& e) {
		std::cerr << "Argument error: " << e.what() << std::endl;
		return 2;
	}
}

void saveResults(const Network& net, const std::vector<int>& predictions, const std::vector<float>& loss_curve, const TrainConfig& config) {
	try {
		// Create base output directory
		struct stat st;
		if (stat("output", &st) != 0) {
			if (mkdir("output", 0755) != 0 && errno != EEXIST) {
				throw runtime_error(string("mkdir failed: ") + std::strerror(errno));
			}
		}

		// Create run-specific directory
		string run_dir = "output/" + config.str();
		if (stat(run_dir.c_str(), &st) != 0) {
			if (mkdir(run_dir.c_str(), 0755) != 0 && errno != EEXIST) {
				throw runtime_error(string("mkdir failed: ") + std::strerror(errno));
			}
		}

		// Save weights with simple name
		string weights_path = run_dir + "/weights.bin";
		net.save_weights(weights_path);
		cout << "Saved weights to " << weights_path << endl;

		// Save predictions with simple name
		string preds_path = run_dir + "/predictions.bin";
		FILE* pf = fopen(preds_path.c_str(), "wb");
		if (!pf)
			throw runtime_error("Failed to open predictions file for writing");
		for (int v : predictions) {
			unsigned char b = static_cast<unsigned char>(v);
			fwrite(&b, 1, 1, pf);
		}
		fclose(pf);
		cout << "Saved predictions to " << preds_path << endl;

		string cel_path = run_dir + "/losses.txt";
		std::stringstream ss;
		for (size_t i = 0; i < loss_curve.size(); i++) {
			ss << std::to_string(loss_curve[i]) << "\n";
		}
		std::fstream fs = std::fstream(cel_path, std::ios_base::out);
		fs << ss.str();
		fs.close();

	} catch (const std::exception& e) {
		cerr << "Failed to save weights or predictions: " << e.what() << endl;
	}

}

int main(int argc, char* argv[]) {

#if true
	TrainConfig config;
	if (auto configresult = configure(argc, argv); const int* errcode = std::get_if<int>(&configresult)) {
		return *errcode;
	} else {
		config = std::get<TrainConfig>(configresult);
	}

	// Create DataLoader instance
	DataLoader loader;
	const std::vector<Sample>& train_data = loader.get_train_data();
	const std::vector<Sample>& test_data = loader.get_test_data();

	// Initialise the network with parsed seed and hidden size
	Network net = Network(config.hidden_size, config.random_seed);

	// Pick a sample to demonstrate the network with
	int eval_data_index = 2;
	std::string real_class_string = loader.get_prediction_string(train_data[eval_data_index].label);

	// Print the individual sample & overall accuracy.
	int untrained_prediction = net.predict(train_data[eval_data_index].pixels);
	std::string untrained_prediction_string = loader.get_prediction_string(untrained_prediction);
	std::cout << "[Untrained]" << std::endl;
	std::cout << "Network predicted the class of test data sample " << eval_data_index << " is " << untrained_prediction_string << ", the real class is: " << real_class_string << std::endl;
	std::vector<int> predictions = get_predictions(net, test_data);
	float accuracy = evaluate_predictions(test_data, predictions);
	std::cout << "Evaluation Accuracy: " << accuracy * 100.0f << "%" << std::endl;
	std::cout << std::endl;

	// Train.
	cout << "[Training]" << endl;
	std::vector<float> cross_entropy_losses = train_model(net, train_data, config);

	// Print the trained prediction of data[eval_data_index].
	int trained_prediction = net.predict(train_data[eval_data_index].pixels);
	std::string trained_prediction_string = loader.get_prediction_string(trained_prediction);
	std::cout << std::endl
	     << "[Trained]" << std::endl;
	std::cout << "Network predicted the class of test data sample " << eval_data_index << " is " << trained_prediction_string << ", the real class is: " << real_class_string << std::endl;

	// Store predictions to save as file later.
	predictions = get_predictions(net, test_data);
	accuracy = evaluate_predictions(test_data, predictions);
	std::cout << "Evaluation Accuracy: " << accuracy * 100.0f << "%" << std::endl;

	saveResults(net, predictions, cross_entropy_losses, config);
	#endif
	return 0;
}
