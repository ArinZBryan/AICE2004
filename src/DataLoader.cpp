
#include "DataLoader.h"

const std::vector<Sample>& DataLoader::get_train_data() const {
	return train_data;
}

const std::vector<Sample>& DataLoader::get_test_data() const {
	return test_data;
}

DataLoader::DataLoader() {
	bool train_exists = std::ifstream(TRAIN_CSV).good();
	bool test_exists = std::ifstream(TEST_CSV).good();
	if (!train_exists || !test_exists) {
		throw std::runtime_error(
		    "[DataLoader ERROR] In directory " + std::string(CSV_DIR) +
		    ", expected to find the files\n" + std::string(TRAIN_CSV) +
		    ",\n" + std::string(TEST_CSV) +
		    ".\n\nDid you run data/download-fashion-mnist.sh?");
	}

	train_data = load_csv(TRAIN_CSV);
	test_data = load_csv(TEST_CSV);
}

DataLoader::~DataLoader() {
	// Cleanup if needed
}

std::vector<Sample> DataLoader::load_csv(const std::string& filename) {
	std::vector<Sample> data;
	std::ifstream file(filename);
	std::string line;

	// skip header if present
	std::getline(file, line);

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string value;
		Sample s;

		// label
		std::getline(ss, value, ',');
		s.label = std::stoi(value);

		// pixels
		while (std::getline(ss, value, ',')) {
			s.pixels.push_back(std::stof(value) / 255.0f); // normalize
		}

		data.push_back(std::move(s));
	}
	return data;
}

const std::string DataLoader::get_prediction_string(int label) const {
	static const std::unordered_map<int, std::string> fashion_mnist_label_map = {
	    {0, "T-shirt/top"},
	    {1, "Trouser"},
	    {2, "Pullover"},
	    {3, "Dress"},
	    {4, "Coat"},
	    {5, "Sandal"},
	    {6, "Shirt"},
	    {7, "Sneaker"},
	    {8, "Bag"},
	    {9, "Ankle boot"}};

	auto it = fashion_mnist_label_map.find(label);
	if (it != fashion_mnist_label_map.end()) {
		return std::to_string(label) + " (" + it->second + ")";
	}
	return std::to_string(label) + " (Unknown)";
}
