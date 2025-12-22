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

std::vector<float> old_mat_times_vec(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec) {
	std::vector<float> result(mat.size(), 0.0f);
	for (size_t i = 0; i < mat.size(); ++i) {
		for (size_t j = 0; j < vec.size(); ++j) {
			result[i] += mat[i][j] * vec[j];
		}
	}
	return result;
}
std::vector<float> old_vec_plus_vec(const std::vector<float>& vec1, const std::vector<float>& vec2) {
	std::vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] + vec2[i];
	}
	return result;
}
std::vector<float> old_sigmoid_vec(const std::vector<float>& vec) {
	std::vector<float> result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = 1.0f / (1.0f + exp(-vec[i]));
	}
	return result;
}
std::vector<float> old_softmax_vec(const std::vector<float>& vec) {
	std::vector<float> result(vec.size());
	float max_val = *max_element(vec.begin(), vec.end());
	float sum = 0.0f;
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = exp(vec[i] - max_val);
		sum += result[i];
	}
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] /= sum;
	}
	return result;
}
std::vector<float> old_cross_entropy_loss(const std::vector<float>& predicted, const std::vector<float>& actual) {
	std::vector<float> result(predicted.size());
	for (size_t i = 0; i < predicted.size(); ++i) {
		result[i] = -actual[i] * log(predicted[i] + 1e-15f); // add small value to avoid log(0)
	}
	return result;
}
std::vector<float> old_vec_minus_vec(const std::vector<float>& vec1, const std::vector<float>& vec2) {
	std::vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] - vec2[i];
	}
	return result;
}
std::vector<std::vector<float>> old_transpose(const std::vector<std::vector<float>>& x) {
	// x can be mat or vec
	if (x.empty())
		return {};
	size_t rows = x.size();
	size_t cols = x[0].size();
	std::vector<std::vector<float>> result(cols, std::vector<float>(rows));
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result[j][i] = x[i][j];
		}
	}
	return result;
}
std::vector<float> old_sigmoid_derivative(const std::vector<float>& vec) {
	std::vector<float> sig = old_sigmoid_vec(vec);
	std::vector<float> result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = sig[i] * (1.0f - sig[i]);
	}
	return result;
}
std::vector<float> old_multiply_elementwise_vec(const std::vector<float>& vec1, const std::vector<float>& vec2) {
	std::vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] * vec2[i];
	}
	return result;
}
std::vector<std::vector<float>> old_outer_product(const std::vector<float>& a, const std::vector<float>& b) {
	size_t m = a.size();
	size_t n = b.size();
	std::vector<std::vector<float>> out(m, std::vector<float>(n, 0.0f));
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			out[i][j] = a[i] * b[j];
		}
	}
	return out;
}

class Old_Network {
  public:
	static constexpr const int INPUT_SIZE = 784;
	static constexpr const int OUTPUT_SIZE = 10;

	Old_Network(int hidden_size, unsigned int seed) : hidden_size(hidden_size), random_seed(seed) {
		weights = {
		    std::vector<std::vector<float>>(hidden_size, std::vector<float>(INPUT_SIZE)),
		    std::vector<std::vector<float>>(OUTPUT_SIZE, std::vector<float>(hidden_size))};

		bias = {
		    std::vector<float>(hidden_size, 0.0f),
		    std::vector<float>(OUTPUT_SIZE, 0.0f)};

		xavier_initialization(weights[0], INPUT_SIZE, hidden_size);
		xavier_initialization(weights[1], hidden_size, OUTPUT_SIZE);
	}
	void save_weights(const std::string& path) const {
		std::ofstream ofs(path, std::ios::binary);
		if (!ofs) {
			throw std::runtime_error("Unable to open file for writing: " + path);
		}

		// Header: three uint32_t values: INPUT_SIZE, hidden_size, OUTPUT_SIZE
		uint32_t in = static_cast<uint32_t>(INPUT_SIZE);
		uint32_t hidden = static_cast<uint32_t>(hidden_size);
		uint32_t out = static_cast<uint32_t>(OUTPUT_SIZE);
		ofs.write(reinterpret_cast<const char*>(&in), sizeof(in));
		ofs.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
		ofs.write(reinterpret_cast<const char*>(&out), sizeof(out));

		// Write W1 (hidden x INPUT_SIZE)
		for (uint32_t i = 0; i < hidden; ++i) {
			for (uint32_t j = 0; j < static_cast<uint32_t>(INPUT_SIZE); ++j) {
				float v = weights[0][i][j];
				ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
			}
		}

		// Write W2 (OUTPUT_SIZE x hidden)
		for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
			for (uint32_t j = 0; j < hidden; ++j) {
				float v = weights[1][i][j];
				ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
			}
		}

		// Write b1 (hidden)
		for (uint32_t i = 0; i < hidden; ++i) {
			float v = bias[0][i];
			ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
		}

		// Write b2 (OUTPUT_SIZE)
		for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
			float v = bias[1][i];
			ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
		}

		ofs.close();
	}
	~Old_Network() {}
	void xavier_initialization(std::vector<std::vector<float>>& W, int in_dim, int out_dim) {
		std::mt19937 gen(random_seed);

		float limit = sqrt(6.0f / (in_dim + out_dim));
		std::uniform_real_distribution<float> dist(-limit, limit);

		for (int i = 0; i < out_dim; i++) {
			for (int j = 0; j < in_dim; j++) {
				W[i][j] = dist(gen);
			}
		}
	}
	std::vector<float> forward(const std::vector<float>& input) {
		last_input = input;
		// Input to hidden: weights * input + bias
		last_x1 = old_mat_times_vec(weights[0], input);
		last_x2 = old_vec_plus_vec(last_x1, bias[0]);

		// Activation function: Sigmoid
		last_x3 = old_sigmoid_vec(last_x2);

		// Hidden to output: weights * hidden + bias
		last_x4 = old_mat_times_vec(weights[1], last_x3);
		last_x5 = old_vec_plus_vec(last_x4, bias[1]);

		// Apply softmax to get probabilities
		last_output = old_softmax_vec(last_x5);

		return last_output;
	}
	void backpropagate(const std::vector<float>& target, double learning_rate) {
		// Derivative of loss w.r.t. output (cross-entropy loss with softmax)
		std::vector<float> delta_out = old_vec_minus_vec(last_output, target);

		// Gradient for hidden -> output weights and biases
		std::vector<float> z4 = delta_out;
		std::vector<std::vector<float>> gradients_W2 = old_outer_product(z4, last_x3);

		// Gradient for input -> hidden weights and biases
		std::vector<std::vector<float>> W2T = old_transpose(weights[1]);
		std::vector<float> z3 = old_mat_times_vec(W2T, z4);
		std::vector<float> sig_deriv = old_sigmoid_derivative(last_x2);
		std::vector<float> z2 = old_multiply_elementwise_vec(z3, sig_deriv);
		std::vector<std::vector<float>> gradients_W1 = old_outer_product(z2, last_input);

		// Update the biases
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			for (int j = 0; j < hidden_size; j++) {
				weights[1][i][j] -= learning_rate * gradients_W2[i][j];
			}
			bias[1][i] -= learning_rate * z4[i];
		}

		// Update the weights
		for (int i = 0; i < hidden_size; i++) {
			for (int j = 0; j < INPUT_SIZE; j++) {
				weights[0][i][j] -= learning_rate * gradients_W1[i][j];
			}
			bias[0][i] -= learning_rate * z2[i];
		}
	}
	int predict(const std::vector<float>& input) {
		std::vector<float> output = forward(input);
		return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
	}

	const std::vector<std::vector<std::vector<float>>>& get_weights() const {
		return weights;
	}
	const std::vector<std::vector<float>>& get_bias() const {
		return bias;
	}

	int hidden_size;
	unsigned int random_seed;
	std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<float>> bias;

	// Intermediary variables for backpropagation
	std::vector<float> last_input;
	std::vector<float> last_x1;
	std::vector<float> last_x2;
	std::vector<float> last_x3;
	std::vector<float> last_x4;
	std::vector<float> last_x5;
	std::vector<float> last_output;
};

Network::state smuggle(Network& net) {
	return {
	    net,
	    &Network::xavier_initialization,
	    net.hidden_size,
	    net.random_seed,
	    net.weights,
	    net.bias,
	    net.last_input,
	    net.last_x1,
	    net.last_x2,
	    net.last_x3,
	    net.last_x4,
	    net.last_x5,
	    net.last_output};
}

int main(int argc, char* argv[]) {

#if true
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
	std::vector<float> cross_entropy_losses = train_model(net, train_data, config);

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

		string cel_path = run_dir + "/losses.txt";
		std::stringstream ss;
		for (size_t i = 0; i < cross_entropy_losses.size(); i++) {
			ss << std::to_string(cross_entropy_losses[i]) << "\n";
		}
		std::fstream fs = std::fstream(cel_path, std::ios_base::out);
		fs << ss.str();
		fs.close();

	} catch (const std::exception& e) {
		cerr << "Failed to save weights or predictions: " << e.what() << endl;
	}
#endif
	return 0;
}
