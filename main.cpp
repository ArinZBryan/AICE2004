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

using namespace std;

int main(int argc, char* argv[]) {
	TrainConfig config;
	try {
		config = parse_arguments(argc, argv);
	} catch (const std::logic_error&) {
		// Help requested
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Argument error: " << e.what() << std::endl;
		return 1;
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
	cout << "[Untrained]" << endl;
	cout << "Network predicted the class of test data sample " << eval_data_index << " is " << untrained_prediction_string << ", the real class is: " << real_class_string << endl;
	evaluate_model(net, test_data, nullptr);
	cout << endl;

	// Train.
	cout << "[Training]" << endl;
	train_model(net, train_data, config);

	// Print the trained prediction of data[eval_data_index].
	int trained_prediction = net.predict(train_data[eval_data_index].pixels);
	std::string trained_prediction_string = loader.get_prediction_string(trained_prediction);
	cout << endl
	     << "[Trained]" << endl;
	cout << "Network predicted the class of test data sample " << eval_data_index << " is " << trained_prediction_string << ", the real class is: " << real_class_string << endl;

	// Store predictions to save as file later.
	std::vector<int> predictions;
	evaluate_model(net, test_data, &predictions);

	// Save the weights and predictions (required for submission)
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
	} catch (const std::exception& e) {
		cerr << "Failed to save weights or predictions: " << e.what() << endl;
	}

	return 0;
}
