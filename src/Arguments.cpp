
#include "Arguments.h"
#include <iostream>
#include <stdexcept>
#include <string>

static void print_usage(const char *prog) {
	std::cout << "Usage: " << prog << " [options]\n"
	     << "Options:\n"
	     << "  -s, --random_seed <num>    Random seed (unsigned int).\n"
	     << "  -e, --epochs <n>           Number of epochs.\n"
	     << "  -l, --learning_rate <f>    Learning rate.\n"
	     << "  -b, --batch_size <n>       Batch size.\n"
	     << "  -z, --hidden_size <n>      Hidden layer size.\n"
	     << "      --threads <n>          Number of threads per task (default 1).\n"
	     << "      --tasks <n>            Number of tasks (default 1).\n"
	     << "  -h, --help                 Show this help message\n";
}

TrainConfig parse_arguments(int argc, char *argv[]) {
	TrainConfig out;
	bool has_seed   = false;
	bool has_epochs = false;
	bool has_lr     = false;
	bool has_batch  = false;
	bool has_hidden = false;

	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		try {
			if (a == "-s" || a == "--random_seed") {
				if (i + 1 < argc) {
					out.random_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
					has_seed = true;
				} else {
					print_usage(argv[0]);
					throw std::runtime_error("Missing value for --random_seed");
				}
			} else if (a == "-e" || a == "--epochs") {
				if (i + 1 < argc) {
					out.epochs = std::stoi(argv[++i]);
					has_epochs = true;
				} else {
					print_usage(argv[0]);
					throw std::runtime_error("Missing value for --epochs");
				}
			} else if (a == "-l" || a == "--learning_rate") {
				if (i + 1 < argc) {
					out.learning_rate = std::stod(argv[++i]);
					has_lr = true;
				} else {
					print_usage(argv[0]);
					throw std::runtime_error("Missing value for --learning_rate");
				}
			} else if (a == "-b" || a == "--batch_size") {
				if (i + 1 < argc) {
					out.batch_size = std::stoi(argv[++i]);
					has_batch = true;
				} else {
					print_usage(argv[0]);
					throw std::runtime_error("Missing value for --batch_size");
				}
			} else if (a == "-z" || a == "--hidden_size") {
				if (i + 1 < argc) {
					out.hidden_size = std::stoi(argv[++i]);
					has_hidden = true;
				} else {
					print_usage(argv[0]);
					throw std::runtime_error("Missing value for --hidden_size");
				}
			} else if (a == "--threads") {
				if (i + 1 < argc) {
					out.threads = static_cast<unsigned int>(std::stoul(argv[++i]));
				} else {
					print_usage(argv[0]);
					throw std::runtime_error("Missing value for --threads");
				}
			} else if (a == "--tasks") {
				if (i + 1 < argc) {
					out.tasks = static_cast<unsigned int>(std::stoul(argv[++i]));
				} else {
					print_usage(argv[0]);
					throw std::runtime_error("Missing value for --tasks");
				}
			} else if (a == "--grain") {
				if (i + 1 < argc) {
					out.grainsize = static_cast<unsigned int>(std::stoul(argv[++i]));
				} else {
					throw std::runtime_error("Missing value for --grain");
				}
			} else if (a == "-h" || a == "--help") {
				print_usage(argv[0]);
				throw std::logic_error("Help requested");
			} else {
				print_usage(argv[0]);
				throw std::runtime_error(std::string("Unknown argument: ") + a);
			}
		} catch (const std::invalid_argument &e) {
			print_usage(argv[0]);
			throw std::runtime_error(std::string("Invalid numeric value for argument ") + a);
		} catch (const std::out_of_range &e) {
			print_usage(argv[0]);
			throw std::runtime_error(std::string("Numeric value out of range for argument ") + a);
		}
	}

	if (!has_seed || !has_epochs || !has_lr || !has_batch || !has_hidden) {
		print_usage(argv[0]);
		throw std::runtime_error("Missing required arguments. All of --random_seed, --epochs, --learning_rate, --batch_size, --hidden_size are required.");
	}

	if (out.epochs <= 0) {
		throw std::runtime_error("--epochs must be > 0");
	}
	if (out.learning_rate <= 0.0) {
		throw std::runtime_error("--learning_rate must be > 0");
	}
	if (out.batch_size <= 0) {
		throw std::runtime_error("--batch_size must be > 0");
	}
	if (out.hidden_size <= 0) {
		throw std::runtime_error("--hidden_size must be > 0");
	}

	return out;
}
