#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include "Matrix.h"
#include "Vector.h"
#include "Maths.h"
#include "Network.h"

using namespace Catch::Matchers;

std::vector<float> old_mat_times_vec(const std::vector<std::vector<float>> &mat, const std::vector<float> &vec);
std::vector<float> old_vec_plus_vec(const std::vector<float> &vec1, const std::vector<float> &vec2);
std::vector<float> old_sigmoid_vec(const std::vector<float> &vec);
std::vector<float> old_softmax_vec(const std::vector<float> &vec);
std::vector<float> old_cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &actual);
std::vector<float> old_vec_minus_vec(const std::vector<float> &vec1, const std::vector<float> &vec2);
std::vector<std::vector<float>> old_transpose(const std::vector<std::vector<float>> &x);
std::vector<float> old_sigmoid_derivative(const std::vector<float> &vec);
std::vector<float> old_multiply_elementwise_vec(const std::vector<float> &vec1, const std::vector<float> &vec2);
std::vector<std::vector<float>> old_outer_product(const std::vector<float> &a, const std::vector<float> &b);

class Old_Network {
public:
    static constexpr const int INPUT_SIZE  = 784;
	static constexpr const int OUTPUT_SIZE = 10;

    Old_Network(int hidden_size, unsigned int seed) : hidden_size(hidden_size), random_seed(seed) {
        weights = { 
            std::vector<std::vector<float>>(hidden_size, std::vector<float>(INPUT_SIZE)), 
            std::vector<std::vector<float>>(OUTPUT_SIZE, std::vector<float>(hidden_size))
        };
        
        bias = {
            std::vector<float>(hidden_size, 0.0f),
            std::vector<float>(OUTPUT_SIZE, 0.0f)
        };

        xavier_initialization(weights[0], INPUT_SIZE, hidden_size);
        xavier_initialization(weights[1], hidden_size, OUTPUT_SIZE);
    }
    void save_weights(const std::string &path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Unable to open file for writing: " + path);
        }

        // Header: three uint32_t values: INPUT_SIZE, hidden_size, OUTPUT_SIZE
        uint32_t in     = static_cast<uint32_t>(INPUT_SIZE);
        uint32_t hidden = static_cast<uint32_t>(hidden_size);
        uint32_t out    = static_cast<uint32_t>(OUTPUT_SIZE);
        ofs.write(reinterpret_cast<const char *>(&in), sizeof(in));
        ofs.write(reinterpret_cast<const char *>(&hidden), sizeof(hidden));
        ofs.write(reinterpret_cast<const char *>(&out), sizeof(out));

        // Write W1 (hidden x INPUT_SIZE)
        for (uint32_t i = 0; i < hidden; ++i) {
            for (uint32_t j = 0; j < static_cast<uint32_t>(INPUT_SIZE); ++j) {
                float v = weights[0][i][j];
                ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
            }
        }

        // Write W2 (OUTPUT_SIZE x hidden)
        for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
            for (uint32_t j = 0; j < hidden; ++j) {
                float v = weights[1][i][j];
                ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
            }
        }

        // Write b1 (hidden)
        for (uint32_t i = 0; i < hidden; ++i) {
            float v = bias[0][i];
            ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }

        // Write b2 (OUTPUT_SIZE)
        for (uint32_t i = 0; i < static_cast<uint32_t>(OUTPUT_SIZE); ++i) {
            float v = bias[1][i];
            ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }

        ofs.close();
    }
    ~Old_Network() {}
    void xavier_initialization(std::vector<std::vector<float>> &W, int in_dim, int out_dim) {
        std::mt19937 gen(random_seed);

        float limit = sqrt(6.0f / (in_dim + out_dim));
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (int i = 0; i < out_dim; i++) {
            for (int j = 0; j < in_dim; j++) {
                W[i][j] = dist(gen);
            }
        }
    }
    std::vector<float> forward(const std::vector<float> &input) {
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
    void backpropagate(const std::vector<float> &target, double learning_rate) {
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
    int predict(const std::vector<float> &input) {
        std::vector<float> output = forward(input);
        return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    }

    const std::vector<std::vector<std::vector<float>>> &get_weights() const {
		return weights;
	}
	const std::vector<std::vector<float>> &get_bias() const {
		return bias;
	}

    int                           hidden_size;
	unsigned int                  random_seed;
	std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<float>>         bias;

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
        net.last_output
    };
}

TEST_CASE("constructor", "[Network.cpp]") {
    SECTION("1 Hidden Node") {
        Old_Network on = Old_Network(1, 1);
        Network nn = Network(1, 1);

        Network::state snn = smuggle(nn);
        REQUIRE(on.hidden_size == static_cast<int>(snn.hidden_size));
        REQUIRE(on.random_seed == snn.random_seed);

        REQUIRE(snn.weights[0].rows() == on.weights[0].size());
        REQUIRE(snn.weights[0].cols() == on.weights[0][0].size());
        for (size_t i = 0; i < snn.weights[0].rows(); i++) {
            for (size_t j = 0; j < snn.weights[0].cols(); j++) {
                REQUIRE_THAT(on.weights[0][i][j], WithinULP(snn.weights[0](i,j), 2));
            }
        }
        
        REQUIRE(snn.weights[1].rows() == on.weights[1].size());
        REQUIRE(snn.weights[1].cols() == on.weights[1][0].size());
        for (size_t i = 0; i < snn.weights[1].rows(); i++) {
            for (size_t j = 0; j < snn.weights[1].cols(); j++) {
                REQUIRE_THAT(on.weights[1][i][j], WithinULP(snn.weights[1](i,j), 2));
            }
        }

        REQUIRE(snn.bias[0].size() == on.bias[0].size());
        for (size_t i = 0; i < snn.bias[0].size(); i++) {
            REQUIRE_THAT(on.bias[0][i], WithinULP(snn.bias[0](i), 2));
        }

        REQUIRE(snn.bias[1].size() == on.bias[1].size());
        for (size_t i = 0; i < snn.bias[1].size(); i++) {
            REQUIRE_THAT(on.bias[1][i], WithinULP(snn.bias[1](i), 2));
        }

    }
    SECTION("10 Hidden Nodes") {
        Old_Network on = Old_Network(10, 1);
        Network nn = Network(10, 1);

        Network::state snn = smuggle(nn);
        REQUIRE(on.hidden_size == static_cast<int>(snn.hidden_size));
        REQUIRE(on.random_seed == snn.random_seed);

        REQUIRE(snn.weights[0].rows() == on.weights[0].size());
        REQUIRE(snn.weights[0].cols() == on.weights[0][0].size());
        for (size_t i = 0; i < snn.weights[0].rows(); i++) {
            for (size_t j = 0; j < snn.weights[0].cols(); j++) {
                REQUIRE_THAT(on.weights[0][i][j], WithinULP(snn.weights[0](i,j), 2));
            }
        }
        
        REQUIRE(snn.weights[1].rows() == on.weights[1].size());
        REQUIRE(snn.weights[1].cols() == on.weights[1][0].size());
        for (size_t i = 0; i < snn.weights[1].rows(); i++) {
            for (size_t j = 0; j < snn.weights[1].cols(); j++) {
                REQUIRE_THAT(on.weights[1][i][j], WithinULP(snn.weights[1](i,j), 2));
            }
        }
        
        REQUIRE(snn.bias[0].size() == on.bias[0].size());
        for (size_t i = 0; i < snn.bias[0].size(); i++) {
            REQUIRE_THAT(on.bias[0][i], WithinULP(snn.bias[0](i), 2));
        }

        REQUIRE(snn.bias[1].size() == on.bias[1].size());
        for (size_t i = 0; i < snn.bias[1].size(); i++) {
            REQUIRE_THAT(on.bias[1][i], WithinULP(snn.bias[1](i), 2));
        }
    }
    SECTION("100 Hidden Nodes") {
        Old_Network on = Old_Network(100, 1);
        Network nn = Network(100, 1);

        Network::state snn = smuggle(nn);
        REQUIRE(on.hidden_size == static_cast<int>(snn.hidden_size));
        REQUIRE(on.random_seed == snn.random_seed);

        REQUIRE(snn.weights[0].rows() == on.weights[0].size());
        REQUIRE(snn.weights[0].cols() == on.weights[0][0].size());
        for (size_t i = 0; i < snn.weights[0].rows(); i++) {
            for (size_t j = 0; j < snn.weights[0].cols(); j++) {
                REQUIRE_THAT(on.weights[0][i][j], WithinULP(snn.weights[0](i,j), 2));
            }
        }
        
        REQUIRE(snn.weights[1].rows() == on.weights[1].size());
        REQUIRE(snn.weights[1].cols() == on.weights[1][0].size());
        for (size_t i = 0; i < snn.weights[1].rows(); i++) {
            for (size_t j = 0; j < snn.weights[1].cols(); j++) {
                REQUIRE_THAT(on.weights[1][i][j], WithinULP(snn.weights[1](i,j), 2));
            }
        }

        REQUIRE(snn.bias[0].size() == on.bias[0].size());
        for (size_t i = 0; i < snn.bias[0].size(); i++) {
            REQUIRE_THAT(on.bias[0][i], WithinULP(snn.bias[0](i), 2));
        }

        REQUIRE(snn.bias[1].size() == on.bias[1].size());
        for (size_t i = 0; i < snn.bias[1].size(); i++) {
            REQUIRE_THAT(on.bias[1][i], WithinULP(snn.bias[1](i), 2));
        }        
    }
}

TEST_CASE("xavier_initialization", "[Network.cpp]") {
    SECTION("3x3 Matrix") {
        Old_Network on = Old_Network(1, 1);
        Network nn = Network(1, 1);

        auto onm = std::vector<std::vector<float>>(3, std::vector<float>(3));
        on.xavier_initialization(onm, 3, 3);

        auto snn = smuggle(nn);
        auto nnm = Matrix(3, 3);
        (snn.network.*(snn.func_xavier))(nnm, 3, 3); //call the smuggled xavier_initialization function

        REQUIRE(nnm.rows() == onm.size());
        REQUIRE(nnm.cols() == onm[0].size());

        for (size_t i = 0; i < nnm.rows(); i++) {
            for (size_t j = 0; j < nnm.cols(); j++) {
                REQUIRE_THAT(onm[i][j], WithinULP(nnm(i,j), 2));
            }
        }
    }
    SECTION("5x5 Matrix") {
        Old_Network on = Old_Network(1, 1);
        Network nn = Network(1, 1);

        auto onm = std::vector<std::vector<float>>(5, std::vector<float>(5));
        on.xavier_initialization(onm, 5, 5);

        auto snn = smuggle(nn);
        auto nnm = Matrix(5, 5);
        (snn.network.*(snn.func_xavier))(nnm, 5, 5); //call the smuggled xavier_initialization function

        REQUIRE(nnm.rows() == onm.size());
        REQUIRE(nnm.cols() == onm[0].size());

        for (size_t i = 0; i < nnm.rows(); i++) {
            for (size_t j = 0; j < nnm.cols(); j++) {
                REQUIRE_THAT(onm[i][j], WithinULP(nnm(i,j), 2));
            }
        }
    }
    SECTION("Larger matrix than supplied bounds") {
        Old_Network on = Old_Network(1, 1);
        Network nn = Network(1, 1);

        auto onm = std::vector<std::vector<float>>(5, std::vector<float>(5));
        on.xavier_initialization(onm, 3, 3);

        auto snn = smuggle(nn);
        auto nnm = Matrix(5, 5);
        (snn.network.*(snn.func_xavier))(nnm, 3, 3); //call the smuggled xavier_initialization function

        REQUIRE(nnm.rows() == onm.size());
        REQUIRE(nnm.cols() == onm[0].size());

        for (size_t i = 0; i < nnm.rows(); i++) {
            for (size_t j = 0; j < nnm.cols(); j++) {
                REQUIRE_THAT(onm[i][j], WithinULP(nnm(i,j), 2));
            }
        }
    }
    SECTION("Non-Square Matrix") {
        Old_Network on = Old_Network(1, 1);
        Network nn = Network(1, 1);

        auto onm = std::vector<std::vector<float>>(20, std::vector<float>(5));
        on.xavier_initialization(onm, 5, 20);

        auto snn = smuggle(nn);
        auto nnm = Matrix(20, 5);
        (snn.network.*(snn.func_xavier))(nnm, 5, 20); //call the smuggled xavier_initialization function

        REQUIRE(nnm.rows() == onm.size());
        REQUIRE(nnm.cols() == onm[0].size());

        for (size_t i = 0; i < nnm.rows(); i++) {
            for (size_t j = 0; j < nnm.cols(); j++) {
                REQUIRE_THAT(onm[i][j], WithinULP(nnm(i,j), 2));
            }
        }
    }
}

TEST_CASE("forward", "[Network.cpp]") {
    Old_Network on = Old_Network(100, 1);
    Network nn = Network(100, 1);
    
    std::mt19937 gen = std::mt19937();
    std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(0, 1);
    Vector nnv = Vector(Network::INPUT_SIZE);
    for (size_t i = 0; i < Network::INPUT_SIZE; i++) {
        nnv(i) = dist(gen);
    }

    std::vector<float> onv = to_float_vector(nnv);
    std::vector<float> onr = on.forward(onv);
    std::vector<float> nnr = to_float_vector(nn.forward(nnv));

    REQUIRE(nnr.size() == onr.size());
    for (size_t j = 0; j < nnr.size(); j++) {
        REQUIRE_THAT(onr[j], WithinULP(nnr[j], 3));
    }
}

TEST_CASE("backpropogate", "[Network.cpp]") {
    Old_Network on = Old_Network(100, 1);
    Network nn = Network(100, 1);
    
    std::mt19937 gen = std::mt19937();
    std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(0, 1);
    Vector nnv = Vector(Network::INPUT_SIZE);
    for (size_t i = 0; i < Network::INPUT_SIZE; i++) {
        nnv(i) = dist(gen);
    }

    Vector label_one_hot = Vector(10, 0.0f);
    label_one_hot(0) = 1.0f;
    on.forward(to_float_vector(nnv));
    nn.forward(nnv);
	on.backpropagate(to_float_vector(label_one_hot), 0.1);
    nn.backpropagate(label_one_hot, 0.1);

    Network::state snn = smuggle(nn);

    REQUIRE(snn.weights[0].rows() == on.weights[0].size());
    REQUIRE(snn.weights[0].cols() == on.weights[0][0].size());
    for (size_t i = 0; i < snn.weights[0].rows(); i++) {
        for (size_t j = 0; j < snn.weights[0].cols(); j++) {
            REQUIRE_THAT(on.weights[0][i][j], WithinAbs(snn.weights[0](i,j), 0.0000001));
        }//                                                                  
    }
        
    REQUIRE(snn.weights[1].rows() == on.weights[1].size());
    REQUIRE(snn.weights[1].cols() == on.weights[1][0].size());
    for (size_t i = 0; i < snn.weights[1].rows(); i++) {
        for (size_t j = 0; j < snn.weights[1].cols(); j++) {
            REQUIRE_THAT(on.weights[1][i][j], WithinAbs(snn.weights[1](i,j), 0.0000001));
        }
    }

    REQUIRE(snn.bias[0].size() == on.bias[0].size());
    for (size_t i = 0; i < snn.bias[0].size(); i++) {
        REQUIRE_THAT(on.bias[0][i], WithinAbs(snn.bias[0](i), 0.0000001));
    }
    
    REQUIRE(snn.bias[1].size() == on.bias[1].size());
    for (size_t i = 0; i < snn.bias[1].size(); i++) {
        REQUIRE_THAT(on.bias[1][i], WithinAbs(snn.bias[1](i), 0.0000001));
    }
}

TEST_CASE("predict", "[Network.cpp]") {
    Old_Network on = Old_Network(100, 1);
    Network nn = Network(100, 1);
    
    Network::state snn = smuggle(nn);
    REQUIRE(on.hidden_size == static_cast<int>(snn.hidden_size));
    REQUIRE(on.random_seed == snn.random_seed);
    
    REQUIRE(snn.weights[0].rows() == on.weights[0].size());
    REQUIRE(snn.weights[0].cols() == on.weights[0][0].size());
    for (size_t i = 0; i < snn.weights[0].rows(); i++) {
        for (size_t j = 0; j < snn.weights[0].cols(); j++) {
            REQUIRE(on.weights[0][i][j] == snn.weights[0](i,j));
        }
    }
    
    REQUIRE(snn.weights[1].rows() == on.weights[1].size());
    REQUIRE(snn.weights[1].cols() == on.weights[1][0].size());
    for (size_t i = 0; i < snn.weights[1].rows(); i++) {
        for (size_t j = 0; j < snn.weights[1].cols(); j++) {
            REQUIRE(on.weights[1][i][j] == snn.weights[1](i,j));
        }
    }
    
    REQUIRE(snn.bias[0].size() == on.bias[0].size());
    for (size_t i = 0; i < snn.bias[0].size(); i++) {
        REQUIRE(on.bias[0][i] == snn.bias[0](i));
    }
    
    REQUIRE(snn.bias[1].size() == on.bias[1].size());
    for (size_t i = 0; i < snn.bias[1].size(); i++) {
        REQUIRE(on.bias[1][i] == snn.bias[1](i));
    }
    
    //Training Sample 1
    std::mt19937 gen = std::mt19937();
    std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(0, 1);
    Vector nnv = Vector(Network::INPUT_SIZE);
    for (size_t i = 0; i < Network::INPUT_SIZE; i++) {
        nnv(i) = dist(gen);
    }

    std::vector<float> onv = to_float_vector(nnv);

    int onr = on.predict(onv);
    int nnr = nn.predict(nnv);
    
    REQUIRE(onr == nnr);
}

    




