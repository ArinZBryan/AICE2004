#ifndef DATASET_H
#define DATASET_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <string>

struct Sample {
	int                label;
	std::vector<float> pixels; // normalized to [0,1]
};

class DataLoader {
  public:
	static constexpr const char *CSV_DIR   = "data";
	static constexpr const char *TRAIN_CSV = "data/fashion-mnist_train.csv";
	static constexpr const char *TEST_CSV  = "data/fashion-mnist_test.csv";

	DataLoader();
	virtual ~DataLoader();

	const std::vector<Sample> &get_train_data() const;
	const std::vector<Sample> &get_test_data() const;
        const std::string get_prediction_string(int label) const;

  private:
	std::vector<Sample> train_data;
	std::vector<Sample> test_data;
	std::vector<Sample> load_csv(const std::string &filename);
};

#endif /* DATASET_H */
