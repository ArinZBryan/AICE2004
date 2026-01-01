#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

#include "Arguments.h"
#include "DataLoader.h"
#include "Network.h"
#include "Train.h"
#include "CompileConfig.h"

#include <iostream>
#include <string>
#include <variant>
#include <stdexcept>
#include <exception>

#include <oneapi/tbb.h>
#include <mpi.h>

using namespace std;

std::variant<TrainConfig, int> configure(int argc, char* argv[]) {
	constexpr char MPI_ENABLED = 0b100;
	constexpr char TBB_ENABLED = 0b010;
	constexpr char BATCHING_ENABLED = 0b001;
	constexpr char MPI_DISABLED = 0, TBB_DISABLED = 0, BATCHING_DISABLED = 0;

	try {
		TrainConfig tc = parse_arguments(argc, argv);

		if (tc.tasks > tc.batch_size) {
			throw std::runtime_error("Batch size must be larger than or equal to number of tasks");
		}

		#ifndef USE_MPI
		if (tc.tasks > 1) {
			throw std::runtime_error("Tasks > 1 is only supported when compiled with MPI.");
		}
		#endif

		unsigned char features = (tc.tasks > 1) << 2 | (tc.threads > 1) << 1 | (tc.batch_size > 1);

		oneapi::tbb::global_control tbb_global_ctrl(oneapi::tbb::global_control::max_allowed_parallelism, tc.threads);

		std::cout << "[Configuration]\n";
		std::cout << "Random Seed:\t\t" << tc.random_seed << "\n";
		std::cout << "Learning Rate:\t\t" << tc.learning_rate << "\n";
		std::cout << "Epochs:\t\t\t" << tc.epochs << "\n";
		std::cout << "Batch Size:\t\t" << tc.batch_size << "\n";
		std::cout << "Hidden Layer Size:\t" << tc.hidden_size << "\n";
		std::cout << "Threads:\t\t" << tc.threads << "\n";
		std::cout << "Tasks:\t\t\t" << tc.tasks << "\n";
		std::cout << "Training Method:\t";
		std::string tm;
		switch (features)
		{
			case MPI_DISABLED | TBB_DISABLED | BATCHING_DISABLED : tm = "Stochastic Gradient Descent"; break;
			case MPI_DISABLED | TBB_DISABLED | BATCHING_ENABLED  : tm = "Mini-Batch Gradient Descent"; break;
			case MPI_DISABLED | TBB_ENABLED  | BATCHING_DISABLED : tm = "Stochastic Gradient Descent"; break;
			case MPI_DISABLED | TBB_ENABLED  | BATCHING_ENABLED  : tm = "Multi-Thread Mini-Batch Gradient Descent"; break;
			case MPI_ENABLED  | TBB_DISABLED | BATCHING_DISABLED : throw std::logic_error("Use of multi-processing via MPI requires batches > 1"); break;
			case MPI_ENABLED  | TBB_DISABLED | BATCHING_ENABLED  : tm = "Multi-Process Mini-Batch Gradient Descent"; break;
			case MPI_ENABLED  | TBB_ENABLED  | BATCHING_DISABLED : throw std::logic_error("Use of multi-processing via MPI requires batches > 1"); break;
			case MPI_ENABLED  | TBB_ENABLED  | BATCHING_ENABLED  : tm = "Multi-Thread Multi-Process Mini-Batch Gradient Descent"; break;
		}
		std::cout << tm << "\n";
		std::cout << "Maths Function Accelerators:\n";
		
		#if defined(USE_AVX_MAT_TIMES_VEC) && defined(__AVX__)
		std::cout << "\tMatrix/Vector Multiplication: AVX\n";
		#else
		std::cout << "\tMatrix/Vector Multiplication: None\n";
		#endif 
		#if defined(USE_AVX_MAT_TRANSPOSE_TIMES_VEC) && defined(__AVX__)
		std::cout << "\tMatrix Transpose Vector Multiplication: AVX\n";
		#else
		std::cout << "\tMatrix Transpose Vector Multiplication: None\n";
		#endif 
		#if defined(USE_AVX_MAT_PLUS_MAT) && defined(__AVX__)
		std::cout << "\tMatrix Addition: " << (tc.use_avx ? "AVX" : "TBB") << "\n";
		#else
		std::cout << "\tMatrix Addition: None\n";
		#endif 
		#if defined(USE_AVX_VEC_PLUS_VEC) && defined(__AVX__)
		std::cout << "\tVector Addition: " << (tc.use_avx ? "AVX" : "TBB") << "\n";
		#else
		std::cout << "\tVector Addition: None\n";
		#endif 
		#if defined(USE_AVX_VEC_MINUS_VEC) && defined(__AVX__)
		std::cout << "\tVector Subtraction: AVX\n";
		#else
		std::cout << "\tVector Subtraction: None\n";
		#endif 
		#if defined(USE_AVX_MULTIPLY_ELEMENTWISE_VEC) && defined(__AVX__)
		std::cout << "\tVector Elementwise Multiplication: AVX\n";
		#else
		std::cout << "\tVector Elementwise Multiplication: None\n";
		#endif 
		#if defined(USE_AVX_SIGMOID_VEC) && defined(__AVX__)
		std::cout << "\tVector Elementwise Sigmoid Function: AVX\n";
		#else
		std::cout << "\tVector Elementwise Sigmoid Function: None\n";
		#endif 
		#if defined(USE_AVX_SIGMOID_DERIVATIVE) && defined(__AVX__)
		std::cout << "\tVector Elementwise Sigmoid Derivative Function: AVX\n";
		#else
		std::cout << "\tVector Elementwise Sigmoid Derivative Function: None\n";
		#endif 
		#if defined(USE_AVX_PRECOMPUTED_SIGMOID_DERIVATIVE) && defined(__AVX__)
		std::cout << "\tVector Elementwise Precomputed Sigmoid Sigmoid Derivative Function: AVX\n";
		#else
		std::cout << "\tVector Elementwise Precomputed Sigmoid Sigmoid Derivative Function: None\n";
		#endif 
		#if defined(USE_AVX_SOFTMAX_VEC) && defined(__AVX__)
		std::cout << "\tVector Elementwise Softmax Function: AVX\n";
		#else
		std::cout << "\tVector Elementwise Softmax Function: None\n";
		#endif 
		#if defined(USE_AVX_CROSS_ENTROPY_LOSS) && defined(__AVX__)
		std::cout << "\tCross-Entropy Loss: AVX\n";
		#else
		std::cout << "\tCross-Entropy Loss: None\n";
		#endif 
		#if defined(USE_AVX_OUTER_PRODUCT) && defined(__AVX__)
		std::cout << "\tVector Outer Product: AVX + TBB\n";
		#else
		std::cout << "\tVector Outer Product: None\n";
		#endif 
		std::cout << "\n";
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

#ifdef USE_MPI
	MPI_Init(&argc, &argv);
	int mpi_instance;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_instance);
#else
	int mpi_instance = 0;
#endif

	TrainConfig config;
	int good = 0;
	if (mpi_instance == 0) {
		auto cresult = configure(argc, argv);
		if (TrainConfig* tc = std::get_if<TrainConfig>(&cresult)) {
			config = *tc;
		} else {
			good = std::get<int>(cresult);
		}
	}
	if (good != 0) { 
		#ifdef USE_MPI
		MPI_Finalize();
		#endif
		return good;
	} else {
		#ifdef USE_MPI
		MPI_Bcast(&config, sizeof(TrainConfig), MPI_BYTE, 0, MPI_COMM_WORLD);
		#endif
	}


	// Create DataLoader instance
	DataLoader loader;
	const std::vector<Sample>& train_data = loader.get_train_data();
	const std::vector<Sample>& test_data = loader.get_test_data();

	// Initialise the network with parsed seed and hidden size
	Network net = Network(config.hidden_size, config.random_seed);

	// Evaluate the untrained model only on the main process
	int eval_data_index = 2;
	if (mpi_instance == 0) {
		// Pick a sample to demonstrate the network with
		std::string real_class_string = loader.get_prediction_string(train_data[eval_data_index].label);

		// Print the individual sample & overall accuracy.
		int untrained_prediction = net.predict(train_data[eval_data_index].pixels, config.use_avx);
		std::string untrained_prediction_string = loader.get_prediction_string(untrained_prediction);
		std::cout << "[Untrained]" << std::endl;
		std::cout << "Network predicted the class of test data sample " << eval_data_index << " is " << untrained_prediction_string << ", the real class is: " << real_class_string << std::endl;
		std::vector<int> predictions = get_predictions(net, test_data, config.use_avx);
		float accuracy = evaluate_predictions(test_data, predictions);
		std::cout << "Evaluation Accuracy: " << accuracy * 100.0f << "%" << std::endl;
		std::cout << std::endl;
	}
	
	// Setup training outputs only on the main process
	std::vector<float> cross_entropy_losses = std::vector<float>();
	if (mpi_instance == 0) { 
		std::cout << "[Training]" << std::endl;
	}

	#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	#endif
	train_model(net, train_data, config, &cross_entropy_losses);

	// Evaluate the trained model only on the main process
	if (mpi_instance == 0) {
		// Print the trained prediction of data[eval_data_index].
		std::string real_class_string = loader.get_prediction_string(train_data[eval_data_index].label);
		int trained_prediction = net.predict(train_data[eval_data_index].pixels, config.use_avx);
		std::string trained_prediction_string = loader.get_prediction_string(trained_prediction);
		std::cout << std::endl
			<< "[Trained]" << std::endl;
		std::cout << "Network predicted the class of test data sample " << eval_data_index << " is " << trained_prediction_string << ", the real class is: " << real_class_string << std::endl;

		// Store predictions to save as file later.
		std::vector<int> predictions = get_predictions(net, test_data, config.use_avx);
		float accuracy = evaluate_predictions(test_data, predictions);
		std::cout << "Evaluation Accuracy: " << accuracy * 100.0f << "%" << std::endl;

		saveResults(net, predictions, cross_entropy_losses, config);
	}
	
	#ifdef USE_MPI
	MPI_Finalize();
	#endif
	return 0;
}
