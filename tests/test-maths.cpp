#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Matrix.h"
#include "Vector.h"
#include "Maths.h"

#include <algorithm>
#include <cmath>
#include <random>

using namespace Catch::Matchers;

std::vector<float> old_mat_times_vec(const std::vector<std::vector<float>> &mat, const std::vector<float> &vec) {
	std::vector<float> result(mat.size(), 0.0f);
	for (size_t i = 0; i < mat.size(); ++i) {
		for (size_t j = 0; j < vec.size(); ++j) {
			result[i] += mat[i][j] * vec[j];
		}
	}
	return result;
}
std::vector<float> old_vec_plus_vec(const std::vector<float> &vec1, const std::vector<float> &vec2) {
	std::vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] + vec2[i];
	}
	return result;
}
std::vector<float> old_sigmoid_vec(const std::vector<float> &vec) {
	std::vector<float> result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = 1.0f / (1.0f + exp(-vec[i]));
	}
	return result;
}
std::vector<float> old_softmax_vec(const std::vector<float> &vec) {
	std::vector<float> result(vec.size());
	float         max_val = *max_element(vec.begin(), vec.end());
	float         sum     = 0.0f;
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = exp(vec[i] - max_val);
		sum += result[i];
	}
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] /= sum;
	}
	return result;
}
std::vector<float> old_cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &actual) {
	std::vector<float> result(predicted.size());
	for (size_t i = 0; i < predicted.size(); ++i) {
		result[i] = -actual[i] * log(predicted[i] + 1e-15f); // add small value to avoid log(0)
	}
	return result;
}
std::vector<float> old_vec_minus_vec(const std::vector<float> &vec1, const std::vector<float> &vec2) {
	std::vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] - vec2[i];
	}
	return result;
}
std::vector<std::vector<float>> old_transpose(const std::vector<std::vector<float>> &x) {
	// x can be mat or vec
	if (x.empty())
		return {};
	size_t                rows = x.size();
	size_t                cols = x[0].size();
	std::vector<std::vector<float>> result(cols, std::vector<float>(rows));
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result[j][i] = x[i][j];
		}
	}
	return result;
}
std::vector<float> old_sigmoid_derivative(const std::vector<float> &vec) {
	std::vector<float> sig = old_sigmoid_vec(vec);
	std::vector<float> result(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		result[i] = sig[i] * (1.0f - sig[i]);
	}
	return result;
}
std::vector<float> old_multiply_elementwise_vec(const std::vector<float> &vec1, const std::vector<float> &vec2) {
	std::vector<float> result(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] * vec2[i];
	}
	return result;
}
std::vector<std::vector<float>> old_outer_product(const std::vector<float> &a, const std::vector<float> &b) {
	size_t                m = a.size();
	size_t                n = b.size();
	std::vector<std::vector<float>> out(m, std::vector<float>(n, 0.0f));
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			out[i][j] = a[i] * b[j];
		}
	}
	return out;
}

// All the 'random' looking floats here were generated in python and copy/pasted into this file
// Catch2's random number generation is really finicky, so just generating some numbers elsewhere was easier

// Some test cases have been ommitted due to intentionally not identical functionality. For instance, in
// `mat_times_vec`, supplying a matrix and vector that are not of compatible shapes fails an assertion,
// whereas in `old_mat_times_vec`, it just barrels on and produces the largest vector it can as an output
// This is just mathematically wrong, so I have changed the functionality

TEST_CASE("mat_times_vec", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("3x3 Idenitiy Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{1, 0, 0}, 
                        {0, 1, 0}, 
                        {0, 0, 1}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Matrices with values in range [0, 1)") {
        Vector new_vec({0.7637562686, 0.51556411739, 0.2964170904});
        Matrix new_mat({{0.7384237358912904, 0.7899228351752887, 0.6963926940566402}, 
                        {0.3909840161319893, 0.43151318320989396, 0.11798587179599618}, 
                        {0.5793062338491267, 0.9885450738501387, 0.4923254007379875}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();
        
        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3col, 2row Matrix times 3row Matrix") {
        Vector new_vec({0.2701601459844052, 0.17044952427181792, 0.9345008763911457});
        Matrix new_mat({{0.041849277474786284, 0.0798752389193994, 0.8974687514089708},
                        {0.6245989833439257, 0.9814549380659491, 0.7210246685111621}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Matrix") {
        Vector new_vec = Vector();
        Matrix new_mat = Matrix();
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Infinite Identity Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{inf, 0, 0}, 
                        {0, inf, 0}, 
                        {0, 0, inf}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Floating Point Minimum Identity Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{min, 0, 0}, 
                        {0, min, 0}, 
                        {0, 0, min}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Floating Point Maximum Identity Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{max, 0, 0}, 
                        {0, max, 0}, 
                        {0, 0, max}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Idenitity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = 1.0f; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Matrices with values in range [0, 1)") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = dist(mt);
        }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18col 9row Matrix times 18row Vector") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(9, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = dist(mt);
        }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Infinite Identity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = inf; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Floating Point Minimum Identity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = min; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Floating Point Maximum Identity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = inf; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_mat, old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
}

TEST_CASE("mat_transpose_times_vec", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("3x3 Idenitiy Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{1, 0, 0}, 
                        {0, 1, 0}, 
                        {0, 0, 1}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        //REQUIRE((static_cast<std::vector<float>>(mat_transpose_times_vec(new_mat, new_vec)) == old_mat_times_vec(old_transpose(old_mat), old_vec)));
        
        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Matrices with values in range [0, 1)") {
        Vector new_vec({0.7637562686, 0.51556411739, 0.2964170904});
        Matrix new_mat({{0.7384237358912904, 0.7899228351752887, 0.6963926940566402}, 
                        {0.3909840161319893, 0.43151318320989396, 0.11798587179599618}, 
                        {0.5793062338491267, 0.9885450738501387, 0.4923254007379875}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();
        
        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("5x5 Matrix with values in range [0, 1)") {
        Vector new_vec({0.683362352485218, 0.35341948021416925, 0.472650224898169, 0.0812847362067518, 0.3110916246965618});
        Matrix new_mat({{0.15393507169773823, 0.5021919151946476, 0.9453530159334795, 0.2831341992680697, 0.939338954746074}, 
                        {0.38523930063952583, 0.8703791546725933, 0.17688040173391872, 0.9167856816112467, 0.5870765871194646}, 
                        {0.24921565140620439, 0.2749222983375553, 0.29582962312829364, 0.049562369703191944, 0.4764981541796566},
                        {0.30388232086641276, 0.2915294693949835, 0.16361560878530168, 0.37598588692590895, 0.9936580866967625},
                        {0.7070666402440435, 0.0580736609181387, 0.831453066635932, 0.5450882330459573, 0.40223997768294617}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();
        
        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3col, 2row Matrix times 3row Matrix") {
        Vector new_vec({0.2701601459844052, 0.17044952427181792, 0.9345008763911457});
        Matrix new_mat({{0.041849277474786284, 0.0798752389193994}, 
                        {0.8974687514089708, 0.6245989833439257}, 
                        {0.9814549380659491, 0.7210246685111621}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();
 
        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Matrix") {
        Vector new_vec = Vector();
        Matrix new_mat = Matrix();
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Infinite Identity Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{inf, 0, 0}, 
                        {0, inf, 0}, 
                        {0, 0, inf}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Floating Point Minimum Identity Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{min, 0, 0}, 
                        {0, min, 0}, 
                        {0, 0, min}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("3x3 Floating Point Maximum Identity Matrix") {
        Vector new_vec({1, 2, 3});
        Matrix new_mat({{max, 0, 0}, 
                        {0, max, 0}, 
                        {0, 0, max}});
        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Idenitity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = 1.0f; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Matrices with values in range [0, 1)") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = dist(mt);
        }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18col 9row Matrix times 18row Vector") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 9, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = dist(mt);
        }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Infinite Identity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = inf; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Floating Point Minimum Identity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = min; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("18x18 Floating Point Maximum Identity Matrix") {
        Vector new_vec(18);
        for (size_t i = 0; i < new_vec.size(); i++) { new_vec(i) = dist(mt); }
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = inf; }

        std::vector<float> old_vec = to_float_vector(new_vec);
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Vector new_res = mat_transpose_times_vec(new_mat, new_vec);
        std::vector<float> old_res = old_mat_times_vec(old_transpose(old_mat), old_vec);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }

}

TEST_CASE("vec_plus_vec", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958, 0.6234469875463479, 0.6355146253711659});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Vectors") {
        const Vector new_vec1({});
        const Vector new_vec2({});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Infinite Vectors") {
        const Vector new_vec1({inf, inf, inf});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Minimum Vectors") {
        const Vector new_vec1({min, min, min});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Maximum Vectors") {
        const Vector new_vec1({max, max, max});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Infinite Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = inf; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Minimum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = min; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Maximum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = max; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }

}

TEST_CASE("sigmoid_vec", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vector [0,1)") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vector [0,1)") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Vector [-6,6)") {
        const Vector new_vec1({0.15564064975599057*12.0f - 6.0f, 0.8102044338919502*12.0f - 6.0f, 0.9363652236747958*12.0f - 6.0f});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vector [-6,6)") {
        const Vector new_vec1({0.15564064975599057*12.0f - 6.0f, 0.8102044338919502*12.0f - 6.0f, 0.9363652236747958*12.0f - 6.0f, 0.6234469875463479*12.0f - 6.0f, 0.6355146253711659*12.0f - 6.0f});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Vectors") {
        const Vector new_vec1({});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Infinite Vectors") {
        const Vector new_vec1({inf, inf, inf});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Minimum Vectors") {
        const Vector new_vec1({min, min, min});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Maximum Vectors") {
        const Vector new_vec1({max, max, max});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Infinite Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = inf; }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        
        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Minimum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = min; }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        
        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Maximum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = max; }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        
        Vector new_res = sigmoid_vec(new_vec1);
        std::vector<float> old_res = old_sigmoid_vec(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }

}

TEST_CASE("softmax_vec", "[Maths.cpp]") {
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vector [0,1)") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = softmax_vec(new_vec1);
        std::vector<float> old_res = old_softmax_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vector [0,1)") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = softmax_vec(new_vec1);
        std::vector<float> old_res = old_softmax_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Vector [-6,6)") {
        const Vector new_vec1({0.15564064975599057*12.0f - 6.0f, 0.8102044338919502*12.0f - 6.0f, 0.9363652236747958*12.0f - 6.0f});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = softmax_vec(new_vec1);
        std::vector<float> old_res = old_softmax_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vector [-6,6)") {
        const Vector new_vec1({0.15564064975599057*12.0f - 6.0f, 0.8102044338919502*12.0f - 6.0f, 0.9363652236747958*12.0f - 6.0f, 0.6234469875463479*12.0f - 6.0f, 0.6355146253711659*12.0f - 6.0f});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = softmax_vec(new_vec1);
        std::vector<float> old_res = old_softmax_vec(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = softmax_vec(new_vec1);
        std::vector<float> old_res = old_softmax_vec(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
}

TEST_CASE("cross_entropy_loss", "[Maths.cpp]") {
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = cross_entropy_loss(new_vec1, new_vec2);
        std::vector<float> old_res = old_cross_entropy_loss(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958, 0.6234469875463479, 0.6355146253711659});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = cross_entropy_loss(new_vec1, new_vec2);
        std::vector<float> old_res = old_cross_entropy_loss(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Vectors") {
        const Vector new_vec1({});
        const Vector new_vec2({});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = cross_entropy_loss(new_vec1, new_vec2);
        std::vector<float> old_res = old_cross_entropy_loss(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Minimum Vectors") {
        const Vector new_vec1({min, min, min});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = cross_entropy_loss(new_vec1, new_vec2);
        std::vector<float> old_res = old_cross_entropy_loss(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Maximum Vectors") {
        const Vector new_vec1({max, max, max});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = cross_entropy_loss(new_vec1, new_vec2);
        std::vector<float> old_res = old_cross_entropy_loss(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Minimum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = min; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Maximum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = max; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_plus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_plus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }

}

TEST_CASE("vec_minus_vec", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958, 0.6234469875463479, 0.6355146253711659});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Vectors") {
        const Vector new_vec1({});
        const Vector new_vec2({});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Infinite Vectors") {
        const Vector new_vec1({inf, inf, inf});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Minimum Vectors") {
        const Vector new_vec1({min, min, min});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Maximum Vectors") {
        const Vector new_vec1({max, max, max});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Infinite Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = inf; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Minimum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = min; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Maximum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = max; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = vec_minus_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_vec_minus_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }

}

TEST_CASE("transpose", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("3x3 Idenitiy Matrix") {
        Matrix new_mat({{1, 0, 0}, 
                        {0, 1, 0}, 
                        {0, 0, 1}});
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("3x3 Matrices with values in range [0, 1)") {
        Matrix new_mat({{0.7384237358912904, 0.7899228351752887, 0.6963926940566402}, 
                        {0.3909840161319893, 0.43151318320989396, 0.11798587179599618}, 
                        {0.5793062338491267, 0.9885450738501387, 0.4923254007379875}});
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();
        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }

        new_mat = Matrix({{0.0009512670031429638, 0.7785503001355455, 0.2430388648578924},
                          {0.04216299558577874, 0.15739663402662818, 0.2301193166046509},
                          {0.8365046871608136, 0.3592856357545595, 0.5596485546210074} });
        old_mat = new_mat.toJaggedArrays();
        new_res = transpose(new_mat);
        old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
        new_mat = Matrix({{0.8461034565199077, 0.2752634698645837, 0.9607637174987697},
                          {0.06691110561859881, 0.6260632575250141, 0.36553697505887084},
                          {0.8468174124614865, 0.47095888831070354, 0.14748502880158865} });
        old_mat = new_mat.toJaggedArrays();
        new_res = transpose(new_mat);
        old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }      
        new_mat = Matrix({{0.10433054234009631, 0.18701821189879564, 0.4097636861716022},
                          {0.7871348033984071, 0.2635997731921401, 0.24332062257941423},
                          {0.9760715188501683, 0.6405153154465641, 0.4602051416454499} });
        old_mat = new_mat.toJaggedArrays();
        new_res = transpose(new_mat);
        old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }      
    }
    SECTION("5x5 Matrix with values in range [0, 1)") {
        Matrix new_mat({{0.15393507169773823, 0.5021919151946476, 0.9453530159334795, 0.2831341992680697, 0.939338954746074}, 
                        {0.38523930063952583, 0.8703791546725933, 0.17688040173391872, 0.9167856816112467, 0.5870765871194646}, 
                        {0.24921565140620439, 0.2749222983375553, 0.29582962312829364, 0.049562369703191944, 0.4764981541796566},
                        {0.30388232086641276, 0.2915294693949835, 0.16361560878530168, 0.37598588692590895, 0.9936580866967625},
                        {0.7070666402440435, 0.0580736609181387, 0.831453066635932, 0.5450882330459573, 0.40223997768294617}});
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("3col, 2row Matrix times 3row Matrix") {
        Matrix new_mat({{0.041849277474786284, 0.0798752389193994, 0.8974687514089708},
                        {0.6245989833439257, 0.9814549380659491, 0.7210246685111621}});
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("3x3 Infinite Identity Matrix") {
        Matrix new_mat({{inf, 0, 0}, 
                        {0, inf, 0}, 
                        {0, 0, inf}});
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("3x3 Floating Point Minimum Identity Matrix") {
        Matrix new_mat({{min, 0, 0}, 
                        {0, min, 0}, 
                        {0, 0, min}});
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("3x3 Floating Point Maximum Identity Matrix") {
        Matrix new_mat({{max, 0, 0}, 
                        {0, max, 0}, 
                        {0, 0, max}});
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("18x18 Idenitity Matrix") {
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) { new_mat(i, i) = 1.0f; }

        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("18x18 Matrices with values in range [0, 1)") {
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = dist(mt);
        }
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("18col 9row Matrix") {
        Matrix new_mat(9, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = dist(mt);
        }

        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("18x18 Infinite Identity Matrix") {
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = inf;
        }
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("18x18 Floating Point Minimum Identity Matrix") {
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = min;
        }
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("18x18 Floating Point Maximum Identity Matrix") {
        Matrix new_mat(18, 18, 0.0f);
        for (size_t i = 0; i < new_mat.rows(); i++) for (size_t j = 0; j < new_mat.cols(); j++) { 
            new_mat(i, j) = max;
        }
        std::vector<std::vector<float>> old_mat = new_mat.toJaggedArrays();

        Matrix new_res = transpose(new_mat);
        std::vector<std::vector<float>> old_res = old_transpose(old_mat);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }


}

TEST_CASE("sigmoid_derivative", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vector [0,1)") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vector [0,1)") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Vector [-6,6)") {
        const Vector new_vec1({0.15564064975599057*12.0f - 6.0f, 0.8102044338919502*12.0f - 6.0f, 0.9363652236747958*12.0f - 6.0f});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vector [-6,6)") {
        const Vector new_vec1({0.15564064975599057*12.0f - 6.0f, 0.8102044338919502*12.0f - 6.0f, 0.9363652236747958*12.0f - 6.0f, 0.6234469875463479*12.0f - 6.0f, 0.6355146253711659*12.0f - 6.0f});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Vectors") {
        const Vector new_vec1({});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Infinite Vectors") {
        const Vector new_vec1({inf, inf, inf});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Minimum Vectors") {
        const Vector new_vec1({min, min, min});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Maximum Vectors") {
        const Vector new_vec1({max, max, max});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);

        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Infinite Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = inf; }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        
        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinAbs(old_res[i], 1e-6));
        }
    }
    SECTION("Size-18 Floating Point Minimum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = min; }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        
        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinAbs(old_res[i], 1e-6));
        }
    }
    SECTION("Size-18 Floating Point Maximum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = max; }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        
        Vector new_res = sigmoid_derivative(new_vec1);
        std::vector<float> old_res = old_sigmoid_derivative(old_vec1);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinAbs(old_res[i], 1e-6));
        }
    }
}

TEST_CASE("multiply_elementwise_vec", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-5 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958, 0.6234469875463479, 0.6355146253711659});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Empty Vectors") {
        const Vector new_vec1({});
        const Vector new_vec2({});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Infinite Vectors") {
        const Vector new_vec1({inf, inf, inf});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Minimum Vectors") {
        const Vector new_vec1({min, min, min});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-3 Floating Point Maximum Vectors") {
        const Vector new_vec1({max, max, max});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE((new_res.size() == old_res.size()));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Infinite Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = inf; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Minimum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = min; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }
    SECTION("Size-18 Floating Point Maximum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = max; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Vector new_res = multiply_elementwise_vec(new_vec1, new_vec2);
        std::vector<float> old_res = old_multiply_elementwise_vec(old_vec1, old_vec2);
        REQUIRE(((new_res.size() == old_res.size())));
        for (size_t i = 0; i < new_res.size(); i++) {
            REQUIRE_THAT(new_res(i), WithinRel(old_res[i]));
        }
    }


}

TEST_CASE("outer_product", "[Maths.cpp]") {
    float inf = std::numeric_limits<number>::infinity();
    float min = std::numeric_limits<number>::min();
    float max = std::numeric_limits<number>::max();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<number> dist(0.0f, 1.0f);

    SECTION("Size-3 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("Size-5 Vectors") {
        const Vector new_vec1({0.44131635835602023, 0.4330904472915381, 0.4504758049261586, 0.693004892325785, 0.30291798447279616});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958, 0.6234469875463479, 0.6355146253711659});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("Empty Vectors") {
        const Vector new_vec1({});
        const Vector new_vec2({});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == 0 && new_res.cols() == 0));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("Size-3 Infinite Vectors") {
        const Vector new_vec1({inf, inf, inf});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("Size-3 Floating Point Minimum Vectors") {
        const Vector new_vec1({min, min, min});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("Size-3 Floating Point Maximum Vectors") {
        const Vector new_vec1({max, max, max});
        const Vector new_vec2({0.15564064975599057, 0.8102044338919502, 0.9363652236747958});
        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) {
            for (size_t j = 0; j < new_res.cols(); j++) {
                REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
            }
        }
    }
    SECTION("Size-18 Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = dist(mt); }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("Size-18 Infinite Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = inf; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("Size-18 Floating Point Minimum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = min; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
    SECTION("Size-18 Floating Point Maximum Vectors") {
        Vector new_vec1(18);
        for (size_t i = 0; i < new_vec1.size(); i++) { new_vec1(i) = max; }
        Vector new_vec2(18);
        for (size_t i = 0; i < new_vec2.size(); i++) { new_vec2(i) = dist(mt); }

        std::vector<float> old_vec1 = to_float_vector(new_vec1);
        std::vector<float> old_vec2 = to_float_vector(new_vec2);

        Matrix new_res = outer_product(new_vec1, new_vec2);
        std::vector<std::vector<float>> old_res = old_outer_product(old_vec1, old_vec2);
        REQUIRE((new_res.rows() == old_res.size() && new_res.cols() == old_res[0].size()));
        for (size_t i = 0; i < new_res.rows(); i++) for (size_t j = 0; j < new_res.cols(); j++) {
            REQUIRE_THAT(new_res(i,j), WithinRel(old_res[i][j]));
        }
    }
}
