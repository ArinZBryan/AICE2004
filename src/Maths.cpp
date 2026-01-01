#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "CompileConfig.h"
#include "Maths.h"
#include "Matrix.h"
#include "Vector.h"

#include <oneapi/tbb.h>



#if defined(__AVX__)
#include <immintrin.h>

/*
    Vectorised Single-Precision Floating Point Base Natural Logarithm

    Calculates the natural logarithm of all floating point values in
    a YMM register. Provided by glibc and linked in using `-lm`
*/
extern "C" __m256 _ZGVdN8v_logf(__m256 x);
extern "C" __m256d _ZGVdN4v_log(__m256d x);
/*
    Vectorised Single-Precision Floating Point Euler's Number Exponentiation

    Calculates Euler's number to the power of all floating point values in
    a YMM register. Provided by glibc and linked in using `-lm`
*/
extern "C" __m256 _ZGVdN8v_expf(__m256 x);
extern "C" __m256d _ZGVdN4v_exp(__m256d x);

#ifdef USE_FP64
#define veclog(x) _ZGVdN4v_log(x)
#define vecexp(x) _ZGVdN4v_exp(x)
#define vecadd(a, b) _mm256_add_pd(a, b)
#define vecsub(a, b) _mm256_sub_pd(a, b)
#define vecmul(a, b) _mm256_mul_pd(a, b)
#define vecdiv(a, b) _mm256_div_pd(a, b)
#define vecfmadd(a, b, c) _mm256_fmadd_pd(a, b, c)
#define vecfnmadd(a, b, c) _mm256_fnmadd_pd(a, b, c)
#define vecfmsub(a, b, c) _mm256_fmsub_pd(a, b, c)
#define vecfnmsub(a, b, c) _mm256_fnmsub_pd(a, b, c)
#define vecxor(a, b) _mm256_xor_pd(a, b)
#define vecsetzero() _mm256_setzero_pd()
#define vecsetvalue(x) _mm256_set1_pd(x)
#define vecload(p) _mm256_loadu_pd(p)
#define vecstore(p, x) _mm256_storeu_pd(p, x)
#define vechadd(x) mm256_hadd_pd_fast(x)
#define vecmax(x) mm256_max_pd_vector(x)
#define vecfit(x) (((x) / 4) * 4)
#define vec_main_for(itvar, maxval) for (size_t itvar = 0; (itvar) < (((maxval) / 4) * 4); (itvar) += 4)
#define vec_res_for(itvar, maxval) for (size_t itvar = (((maxval) / 4) * 4); (itvar) < (maxval); itvar++)
#define ymm __m256d
#else
#define veclog(x) _ZGVdN8v_logf(x)
#define vecexp(x) _ZGVdN8v_expf(x)
#define vecadd(a, b) _mm256_add_ps(a, b)
#define vecsub(a, b) _mm256_sub_ps(a, b)
#define vecmul(a, b) _mm256_mul_ps(a, b)
#define vecdiv(a, b) _mm256_div_ps(a, b)
#define vecfmadd(a, b, c) _mm256_fmadd_ps(a, b, c)
#define vecfnmadd(a, b, c) _mm256_fnmadd_ps(a, b, c)
#define vecfmsub(a, b, c) _mm256_fmsub_ps(a, b, c)
#define vecfnmsub(a, b, c) _mm256_fnmsub_ps(a, b, c)
#define vecxor(a, b) _mm256_xor_ps(a, b)
#define vecsetzero() _mm256_setzero_ps()
#define vecsetvalue(x) _mm256_set1_ps(x)
#define vecload(p) _mm256_loadu_ps(p)
#define vecstore(p, x) _mm256_storeu_ps(p, x)
#define vechadd(x) mm256_hadd_ps_fast(x)
#define vecmax(x) mm256_max_ps_vector(x)
#define vecfit(x) (((x) / 8) * 8)
#define vec_main_for(itvar, maxval) for (size_t itvar = 0; (itvar) < (((maxval) / 8) * 8); (itvar) += 8)
#define vec_res_for(itvar, maxval) for (size_t itvar = (((maxval) / 8) * 8); (itvar) < (maxval); itvar++)
#define ymm __m256
#endif

float mm256_hadd_ps_fast(__m256 x) {
	// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )

	const __m128 hi4floats = _mm256_extractf128_ps(x, 1); // get x7 - x4 in 128-bit xmm register
	const __m128 lo4floats = _mm256_castps256_ps128(x);   // get x3 - x0 in 128-bit xmm register

	const __m128 sum4floats = _mm_add_ps(hi4floats, lo4floats); // add xmm registers

	const __m128 lo2floats = sum4floats;                            // copy xmm register into another
	const __m128 hi2floats = _mm_movehl_ps(sum4floats, sum4floats); // copy top two floats into bottom two floats of xmm register

	const __m128 sum2floats = _mm_add_ps(hi2floats, lo2floats); // add xmm registers

	const __m128 lo = sum2floats;                                  // copy xmm register into another
	const __m128 hi = _mm_shuffle_ps(sum2floats, sum2floats, 0x1); // extract 2nd float from sum2floats into hi first float

	const __m128 sum = _mm_add_ss(lo, hi); // add single floats in first positions
	return _mm_cvtss_f32(sum);
}
double mm256_hadd_pd_fast(__m256d x) {
	const __m128d hi2doubles = _mm256_extractf128_pd(x, 1);
	const __m128d lo2doubles = _mm256_castpd256_pd128(x);
	const __m128d sum2doubles = _mm_add_pd(hi2doubles, lo2doubles);
	const __m128d sum = _mm_hadd_pd(sum2doubles, sum2doubles);
	return _mm_cvtsd_f64(sum);
}

float mm256_max_ps_vector(const std::vector<float>& v) {
	if (v.empty())
		return -std::numeric_limits<float>::infinity();

	size_t size = v.size();
	size_t i = 0;

	__m256 max_vec;
	if (size >= 8) {
		max_vec = _mm256_loadu_ps(&v[0]);
		i = 8;
	} else {
		return *std::max_element(v.begin(), v.end());
	}

	// Process chunks of 8 floats
	for (; i + 7 < size; i += 8) {
		__m256 curr = _mm256_loadu_ps(&v[i]);
		max_vec = _mm256_max_ps(max_vec, curr);
	}

	// Horizontal reduction in registers
	__m128 hi = _mm256_extractf128_ps(max_vec, 1); // high 128 bits
	__m128 lo = _mm256_castps256_ps128(max_vec);   // low 128 bits
	__m128 max128 = _mm_max_ps(lo, hi);            // compare low/high halves

	// further reduce 128-bit vector to single float
	max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
	max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));

	float max_val = _mm_cvtss_f32(max128);

	// Handle remaining elements
	for (; i < size; ++i) {
		max_val = std::max(max_val, v[i]);
	}

	return max_val;
}
double mm256_max_pd_vector(const std::vector<double>& v) {
	if (v.empty())
		return -std::numeric_limits<double>::infinity();

	size_t size = v.size();
	size_t i = 0;

	if (size < 4) {
		return *std::max_element(v.begin(), v.end());
	}
	__m256d max_vec = _mm256_loadu_pd(&v[0]);
	i = 4;

	// Process chunks of 4 doubles
	for (; i + 3 < size; i += 4) {
		__m256d curr = _mm256_loadu_pd(&v[i]);
		max_vec = _mm256_max_pd(max_vec, curr);
	}

	// Horizontal reduction in registers
	__m128d hi = _mm256_extractf128_pd(max_vec, 1); // high 128 bits
	__m128d lo = _mm256_castpd256_pd128(max_vec);   // low 128 bits
	__m128d max128 = _mm_max_pd(lo, hi);            // compare low/high halves

	// further reduce 128-bit vector to single float
	double a = _mm_cvtsd_f64(max128);
	max128 = _mm_shuffle_pd(max128, max128, 0);
	double b = _mm_cvtsd_f64(max128);

	double max_val = a > b ? a : b;

	// Handle remaining elements
	for (; i < size; ++i) {
		max_val = std::max(max_val, v[i]);
	}

	return max_val;
}
#endif

/*
    Generalised matrix vector product for any compatible size matrix and vector

    mat = [m1_1, m1_2, m1_3, ...; m2_1, m2_2, m2_3, ...; m3_1, m3_2, m3_3, ...; ...]
    vec = [x1, x2, x3, ...]
    return = [m1_1*x1 + m1_2*x2 + m1_3*x3 + ..., m2_1*x1 + m2_2*x2 + m2_3*x3 + ..., m3_1*x1 + m3_2*x2 + m3_3*x3 + ..., ...]
*/
Vector mat_times_vec(const Matrix& mat, const Vector& vec) {
	Vector ret;
	mat_times_vec(mat, vec, ret);
	return ret;
}
#if defined(USE_AVX_MAT_TIMES_VEC) && defined(__AVX__)
void mat_times_vec(const Matrix& mat, const Vector& vec, Vector& out) {
	// Assert matrix and vector are compatible
	assert((mat.cols() == vec.size()));

	// store sizes and pointers immediately to negate cost of load from class if compiler does not inline
	size_t mat_width = mat.cols();
	const number* mat_data = mat.data();
	const number* vec_data = vec.data();

	// create result vector
	out.resize(mat.rows());

	for (size_t row = 0; row < mat.rows(); row++) {
		// zero out the accumulators
		ymm acc0 = vecsetzero();

		vec_main_for(col, mat.cols()) {
			// we only need to load this once because we reuse it across rows of the matrix
			const ymm vecvals = vecload(vec_data + col);

			// load the relevant parts of the matrix
			const ymm matvals0 = vecload(mat_data + (mat_width * (row)) + col);

			// _mm256_fmadd_ps multiplys the first two operands, then adds that value
			// to the third operand and returns the value.
			// d = _mm256_fmadd_ps(a, b, c) => d = a*b + c
			acc0 = vecfmadd(matvals0, vecvals, acc0);
		}

		// use the O(log n) horizontal add instead of the O(n) add beacuse we are within a register
		number val0 = vechadd(acc0);

		// deal with the remainder columns
		vec_res_for(col, mat.cols()) {
			val0 += mat_data[mat_width * (row) + col] * vec_data[col];
		}

		// store into result
		out(row) = val0;
	}
}
#else
void mat_times_vec(const Matrix& mat, const Vector& vec, Vector& out) {
	assert((mat.cols() == vec.size()));
	out.resize(mat.rows());
	for (size_t row = 0; row < mat.rows(); ++row) {
		number sum = 0.0;
		for (size_t col = 0; col < mat.cols(); ++col) {
			sum += mat(row, col) * vec(col);
		}
		out(row) = sum;
	}
}
#endif

/*
    Generalised matrix transpose vector product for any compatible size matrix and vector

    mat = [m1_1, m1_2, m1_3, ...; m2_1, m2_2, m2_3, ...; m3_1, m3_2, m3_3, ...; ...]
    vec = [x1, x2, x3, ...]
    return = [m1_1*x1 + m2_1*x2 + m3_1*x3 + ..., m1_2*x1 + m2_2*x2 + m3_2*x3 + ..., m1_3*x1 + m2_3*x2 + m3_3*x3 + ..., ...]
*/
Vector mat_transpose_times_vec(const Matrix& mat, const Vector& vec) {
	Vector ret;
	mat_transpose_times_vec(mat, vec, ret);
	return ret;
}
#if defined(USE_AVX_MAT_TRANSPOSE_TIMES_VEC) && defined(__AVX__)
void mat_transpose_times_vec(const Matrix& mat, const Vector& vec, Vector& out) {
	assert((mat.rows() == vec.size()));
	out.resize(mat.cols());

	number* result_ptr = out.data();
	const number* mat_ptr = mat.data();
	const number* vec_ptr = vec.data();

	size_t rows = mat.rows();
	size_t cols = mat.cols();

	for (size_t row = 0; row < rows; ++row) {
		const ymm vec_elem = vecsetvalue(vec_ptr[row]);

		vec_main_for(col, cols) {
			const ymm result_8 = vecload(result_ptr + col);
			const ymm mat_8 = vecload(mat_ptr + row * cols + col);
			const ymm res_8 = vecfmadd(vec_elem, mat_8, result_8);
			vecstore(result_ptr + col, res_8);
		}
		vec_res_for(col, cols) {
			out(col) += mat(row, col) * vec_ptr[row];
		}
	}
}
#else
void mat_transpose_times_vec(const Matrix& mat, const Vector& vec, Vector& out) {
	assert((mat.rows() == vec.size()));
	out.resize(mat.cols());

	size_t rows = mat.rows();
	size_t cols = mat.cols();

	for (size_t row = 0; row < rows; ++row) {
		for (size_t col = 0; col < cols; ++col) {
			out(col) += mat(row, col) * vec(row);
		}
	}
}
#endif

#if defined(USE_AVX_MAT_PLUS_MAT) && defined(__AVX__)
void mat_plus_mat_avx(Matrix& mat1, Matrix& mat2, Matrix& out) {
	// assert vectors are of same size
	assert((mat1.size() == mat2.size() && mat1.size() == out.size()));

	// store sizes and pointers immediately to negate cost of load from class if compiler does not inline
	const number* mat1_ptr = mat1.data();
	const number* mat2_ptr = mat2.data();
	number* res_ptr = out.data();

	vec_main_for(elem, mat1.size()) {
		const ymm a = vecload(mat1_ptr + elem);
		const ymm b = vecload(mat2_ptr + elem);
		const ymm res = vecadd(a, b);
		vecstore(res_ptr + elem, res);
	}

	vec_res_for(elem, mat1.size()) {
		res_ptr[elem] = mat1_ptr[elem] + mat2_ptr[elem];
	}
}
#endif
void mat_plus_mat_tbb(Matrix& mat1, Matrix& mat2, Matrix& out) {
	assert((mat1.rows() == mat2.rows() && mat1.cols() == mat2.cols() && out.rows() == mat1.rows() && out.cols() == mat1.cols()));
	size_t elems = mat1.rows() * mat1.cols();
	number* mat1_data = mat1.data();
	number* mat2_data = mat2.data();
	number* out_data = out.data();

	tbb::parallel_for(tbb::blocked_range<size_t>(0, elems), [&](tbb::blocked_range<size_t> range){
		for (size_t i = range.begin(); i < range.end(); ++i) {
			out_data[i] = mat1_data[i] + mat2_data[i];
		}
	});
}
Matrix mat_plus_mat(const Matrix& mat1, const Matrix& mat2, bool useAVX) {
	Matrix ret(mat1.rows(), mat1.cols());
	mat_plus_mat(const_cast<Matrix&>(mat1), const_cast<Matrix&>(mat2), ret, useAVX);
	return ret;
}
void mat_plus_mat(Matrix& mat1, Matrix& mat2, Matrix& out, bool useAVX) {
	#if defined(USE_AVX_MAT_PLUS_MAT) && defined(__AVX__)
	if (useAVX) mat_plus_mat_avx(mat1, mat2, out);
	else mat_plus_mat_avx(mat1, mat2, out);
	#else
	mat_plus_mat_tbb(mat1, mat2, out);
	#endif
}

/*
    Vector Addition

    vec1 = [u1, u2, u3, ...]
    vec2 = [v1, v2, v3, ...]
    return = [u1+v1, u2+v2, u3+v3, ...]
*/

#if defined(USE_AVX_VEC_PLUS_VEC) && defined(__AVX__)
void vec_plus_vec_avx(Vector& vec1, Vector& vec2, Vector& out) {
	// assert vectors are of same size
	assert((vec1.size() == vec2.size()));

	// create result vector
	out.resize(vec1.size());

	// store sizes and pointers immediately to negate cost of load from class if compiler does not inline
	const number* vec1_ptr = vec1.data();
	const number* vec2_ptr = vec2.data();
	number* res_ptr = out.data();

	// add elements 8 at a time

	vec_main_for(elem, vec1.size()) {
		const ymm a = vecload(vec1_ptr + elem);
		const ymm b = vecload(vec2_ptr + elem);
		const ymm res = vecadd(a, b);
		vecstore(res_ptr + elem, res);
	}

	// deal with residual elements over the multiple of 8
	vec_res_for(elem, vec1.size()) {
		res_ptr[elem] = vec1_ptr[elem] + vec2_ptr[elem];
	}
}
#endif
void vec_plus_vec_tbb(Vector& vec1, Vector& vec2, Vector& out) {
	assert((vec1.size() == vec2.size()));
	out.resize(vec1.size());
	number* out_ptr = out.data();
	number* vec1_ptr = vec1.data();
	number* vec2_ptr = vec2.data();

	tbb::parallel_for(tbb::blocked_range<size_t>(0, vec1.size()), [&](tbb::blocked_range<size_t> range){
		for (size_t i = range.begin(); i < range.end(); ++i) {
			out_ptr[i] = vec1_ptr[i] + vec2_ptr[i];
		}
	});
	
}
Vector vec_plus_vec(const Vector& vec1, const Vector& vec2, bool useAVX) {
	Vector ret;
	vec_plus_vec(const_cast<Vector&>(vec1), const_cast<Vector&>(vec2), ret, useAVX);
	return ret;
}
void vec_plus_vec(Vector& vec1, Vector& vec2, Vector& out, bool useAVX) {
	#if defined(USE_AVX_VEC_PLUS_VEC) && defined(__AVX__)
	if (useAVX) vec_plus_vec_avx(vec1, vec2, out);
	else vec_plus_vec_tbb(vec1, vec2, out);
	#else
	vec_plus_vec_tbb(vec1, vec2, out);
	#endif
}

/*
    Vector Subtraction

    vec1 = [u1, u2, u3, ...]
    vec2 = [v1, v2, v3, ...]
    return = [u1-v1, u2-v2, u3-v3, ...]
*/
Vector vec_minus_vec(const Vector& vec1, const Vector& vec2) {
	Vector ret;
	vec_minus_vec(const_cast<Vector&>(vec1), const_cast<Vector&>(vec2), ret);
	return ret;
}
#if defined(USE_AVX_VEC_MINUS_VEC) && defined(__AVX__)
void vec_minus_vec(Vector& vec1, Vector& vec2, Vector& out) {
	// assert vectors are of same size
	assert((vec1.size() == vec2.size()));

	// create result vector
	out.resize(vec1.size());

	// store sizes and pointers immediately to negate cost of load from class if compiler does not inline
	const number* vec1_ptr = vec1.data();
	const number* vec2_ptr = vec2.data();
	number* res_ptr = out.data();

	// subtract elements 8 at a time
	vec_main_for(elem, vec1.size()) {
		const ymm a = vecload(vec1_ptr + elem);
		const ymm b = vecload(vec2_ptr + elem);
		const ymm res = vecsub(a, b);
		vecstore(res_ptr + elem, res);
	}

	// deal with residual elements over the multiple of 8
	vec_res_for(elem, vec1.size()) {
		res_ptr[elem] = vec1_ptr[elem] - vec2_ptr[elem];
	}
}
#else
void vec_minus_vec(Vector& vec1, Vector& vec2, Vector& out) {
	assert((vec1.size() == vec2.size()));
	out.resize(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		out(i) = vec1(i) - vec2(i);
	}
}
#endif

/*
    Vector Elementwise Multiplication

    vec1 = [u1, u2, u3, ...]
    vec2 = [v1, v2, v3, ...]
    return = [u1*v1, u2*v2, u3*v3, ...]
*/
Vector multiply_elementwise_vec(const Vector& vec1, const Vector& vec2) {
	Vector ret;
	multiply_elementwise_vec(const_cast<Vector&>(vec1), const_cast<Vector&>(vec2), ret);
	return ret;
}
#if defined(USE_AVX_MULTIPLY_ELEMENTWISE_VEC) && defined(__AVX__)
void multiply_elementwise_vec(Vector& vec1, Vector& vec2, Vector& out) {
	// assert vectors are of same size
	assert((vec1.size() == vec2.size()));

	// create result vector
	out.resize(vec1.size());

	// store sizes and pointers immediately to negate cost of load from class if compiler does not inline
	const number* vec1_ptr = vec1.data();
	const number* vec2_ptr = vec2.data();
	number* res_ptr = out.data();

	// multiply elements 8 at a time
	vec_main_for(elem, vec1.size()) {
		const ymm a = vecload(vec1_ptr + elem);
		const ymm b = vecload(vec2_ptr + elem);
		const ymm res = vecmul(a, b);
		vecstore(res_ptr + elem, res);
	}

	// deal with residual elements over the multiple of 8
	vec_res_for(elem, vec1.size()) {
		res_ptr[elem] = vec1_ptr[elem] * vec2_ptr[elem];
	}
}
#else
void multiply_elementwise_vec(Vector& vec1, Vector& vec2, Vector& out) {
	assert((vec1.size() == vec2.size()));
	out.resize(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		out(i) = vec1(i) * vec2(i);
	}
}
#endif

/*
    Vector Elementwise Division

    vec1 = [u1, u2, u3, ...]
    vec2 = [v1, v2, v3, ...]
    return = [u1/v1, u2/v2, u3/v3, ...]
*/
Vector divide_elementwise_vec(const Vector& vec1, const Vector& vec2) {
	Vector ret;
	divide_elementwise_vec(const_cast<Vector&>(vec1), const_cast<Vector&>(vec2), ret);
	return ret;
}
#if defined(USE_AVX_MULTIPLY_ELEMENTWISE_VEC) && defined(__AVX__)
void divide_elementwise_vec(Vector& vec1, Vector& vec2, Vector& out) {
	assert((vec1.size() == vec2.size()));

	out.resize(vec1.size());

	const number* vec1_ptr = vec1.data();
	const number* vec2_ptr = vec2.data();
	number* res_ptr = out.data();

	vec_main_for(elem, vec1.size()) {
		const ymm a = vecload(vec1_ptr + elem);
		const ymm b = vecload(vec2_ptr + elem);
		const ymm res = vecdiv(a, b);
		vecstore(res_ptr + elem, res);
	}

	vec_res_for(elem, vec1.size()) {
		res_ptr[elem] = vec1_ptr[elem] / vec2_ptr[elem];
	}
}
#else
void divide_elementwise_vec(Vector& vec1, Vector& vec2, Vector& out) {
	assert((vec1.size() == vec2.size()));
	out.resize(vec1.size());
	for (size_t i = 0; i < vec1.size(); ++i) {
		out(i) = vec1(i) / vec2(i);
	}
}
#endif

/*
    Vector Elementwise Sigmoid

    vec = [x1, x2, x3, ...]
    return = [1/(1+exp(-x1)), 1/(1+exp(-x2)), 1/(1+exp(-x3)), ...]
*/
Vector sigmoid_vec(const Vector& vec) {
	Vector ret;
	sigmoid_vec(const_cast<Vector&>(vec), ret);
	return ret;
}
#if defined(USE_AVX_SIGMOID_VEC) && defined(__AVX__)
void sigmoid_vec(Vector& vec, Vector& out) {
	// create result vector
	out.resize(vec.size());

	// process floats 8 at a time
	vec_main_for(cur, vec.size()) {
		ymm vec_8 = vecload(vec.data() + cur);

		// flip sign faster than subtracting from zero
		ymm sign_bit = vecsetvalue(-0.0f);
		vec_8 = vecxor(vec_8, sign_bit);

		// call vectorised exponent from glibc
		vec_8 = vecexp(vec_8);

		// add one, and take the reciprocal
		ymm one = vecsetvalue(1.0f);
		vec_8 = vecadd(one, vec_8);
		vec_8 = vecdiv(one, vec_8);

		// store back into result vector
		vecstore(out.data() + cur, vec_8);
	}

	// handle residual computations
	vec_res_for(cur, vec.size()) {
		out(cur) = 1.0 / (1.0 + std::exp(-vec(cur)));
	}
}
#else
void sigmoid_vec(Vector& vec, Vector& out) {
	out.resize(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		out(i) = 1.0 / (1.0 + std::exp(-vec(i)));
	}
}
#endif

/*
    Vector Elementwise Softmax
    vec = [x1, x2, x3, ...]
    inter = [exp(x1 - max(vec)), exp(x2 - max(vec)), exp(x3 - max(vec)), ...]
    return = [i1 / sum(inter), i2 / sum(inter), i3 / sum(inter)]
*/
Vector softmax_vec(const Vector& vec) {
	Vector ret;
	softmax_vec(const_cast<Vector&>(vec), ret);
	return ret;
}
#if defined(USE_AVX_SOFTMAX_VEC) && defined(__AVX__)
void softmax_vec(Vector& vec, Vector& out) {
	// make result vector
	out.resize(vec.size());

	// get maximum value
	number max_val = vecmax(vec);

	// store pointers immediately to negate cost of load from class if compiler does not inline
	number* result_ptr = out.data();
	const number* vec_ptr = vec.data();

	// zero out accumulator
	ymm acc = vecsetzero();

	vec_main_for(cur, vec.size()) {
		ymm max_val_avx = vecsetvalue(max_val);
		ymm vec_avx = vecload(vec_ptr + cur);
		ymm res = vecsub(vec_avx, max_val_avx);
		res = vecexp(res);
		acc = vecadd(acc, res);
		vecstore(result_ptr + cur, res);
	}
	number sum = vechadd(acc);
	vec_res_for(cur, vec.size()) {
		number res = exp(vec_ptr[cur] - max_val);
		result_ptr[cur] = res;
		sum += res;
	}

	// element-wise division
	vec_main_for(cur, vec.size()) {
		ymm sum_avx = vecsetvalue(sum);
		ymm vec_avx = vecload(result_ptr + cur);
		ymm a = vecdiv(vec_avx, sum_avx);
		vecstore(result_ptr + cur, a);
	}
	vec_res_for(cur, vec.size()) {
		result_ptr[cur] = result_ptr[cur] / sum;
	}
}
#else
void softmax_vec(Vector& vec, Vector& out) {
	out.resize(vec.size());
	number max_val = *(std::max_element(vec.begin(), vec.end()));

	number* result_ptr = out.data();
	const number* vec_ptr = vec.data();

	for (size_t i = 0; i < vec.size(); ++i) {
		result_ptr[i] = exp(vec_ptr[i] - max_val);
	}
	number sum = 0.0f;
	for (size_t i = 0; i < vec.size(); ++i) {
		sum += result_ptr[i];
	}
	for (size_t i = 0; i < vec.size(); ++i) {
		result_ptr[i] /= sum;
	}
}
#endif

/*
    Vector Cross-Entropy Loss

    predicted = [p1, p2, p3, ...]
    actual = [a1, a2, a3, ...]
    return = [-a1*log(p1+1e-15f), -a2*log(p2+1e-15f), -a3*log(p3+1e-15f), ...]
*/
Vector cross_entropy_loss(const Vector& predicted, const Vector& actual) {
	Vector res;
	cross_entropy_loss(const_cast<Vector&>(predicted), const_cast<Vector&>(actual), res);
	return res;
}
#if defined(USE_AVX_CROSS_ENTROPY_LOSS) && defined(__AVX__)
void cross_entropy_loss(Vector& predicted, Vector& actual, Vector& out) {
	assert((predicted.size() == actual.size()));
	
	// create result vector
	out.resize(predicted.size());

	// process floats 8 at a time
	vec_main_for(cur, predicted.size()) {
		// load predicted values
		ymm predicted_8 = vecload(predicted.data() + cur);

		// broadcast constant value to all elements in ymm register
		ymm const_8 = vecsetvalue(1e-15f);

		// add constant value to floats
		predicted_8 = vecadd(predicted_8, const_8);

		// call vectorised logf function from glibc
		predicted_8 = veclog(predicted_8);

		// load actual values
		ymm actual_8 = vecload(actual.data() + cur);

		ymm result_8 = vecfnmsub(actual_8, predicted_8, vecsetzero());

		// store in result vector
		vecstore(out.data() + cur, result_8);
	}

	// deal with residual calculations
	vec_res_for(cur, predicted.size()) {
		out(cur) = -actual(cur) * log(predicted(cur) + 1e-15f);
	}
}
#else
void cross_entropy_loss(Vector& predicted, Vector& actual, Vector& out) {
	assert(predicted.size() == actual.size());
	out.resize(predicted.size());
	for (size_t i = 0; i < predicted.size(); ++i) {
		out(i) = -actual(i) * log(predicted(i) + 1e-15f); // add small value to avoid log(0)
	}
}
#endif

/*
    Matrix Transpose

    M = [m1_1, m1_2, m1_3; m2_1, m2_2, m2_3; m3_1, m3_2, m3_3]
    return = M^T

    I don't really know how to vectorise this. it looks like I can
    split this into blocks and perform a transpose on each of those.
    However, since we only ever transpose then perform a matrix times
    vector operation, we can just use `mat_transpose_times_vec`,
    which is much faster
*/
Matrix transpose(const Matrix& x) {
	Matrix ret(x.cols(), x.rows());
	transpose(x, ret);
	return ret;
}
void transpose(const Matrix& x, Matrix& out) {
	assert((x.rows() == out.cols() && x.cols() == out.rows()));

	size_t rows = x.rows();
	size_t cols = x.cols();

	for (size_t j = 0; j < cols; ++j) {
		for (size_t i = 0; i < rows; ++i) {
			out(j, i) = x(i, j);
		}
	}
}

/*
    Vector Sigmoid Derivative (Elementwise)

    vec = [x1, x2, x3, ...]
    inter = [1/(1+exp(-x1)), 1/(1+exp(-x2)), 1/(1+exp(-x3)), ...]
    return = [inter1 - inter1^2, inter2 - inter2^2 , inter3 - inter3^2, ...]
*/
Vector sigmoid_derivative(const Vector& vec) {
	Vector ret;
	sigmoid_derivative(const_cast<Vector&>(vec), ret);
	return ret;
}
#if defined(USE_AVX_SIGMOID_DERIVATIVE) && defined(__AVX__)
void sigmoid_derivative(Vector& vec, Vector& out) {
	out.resize(vec.size());

	// process floats 8 at a time
	vec_main_for(cur, vec.size()) {
		ymm vec_8 = vecload(vec.data() + cur);

		// flip sign faster than subtracting from zero
		ymm sign_bit = vecsetvalue(-0.0f);
		vec_8 = vecxor(vec_8, sign_bit);

		// call vectorised exponent from glibc
		vec_8 = vecexp(vec_8);

		// add one, and take the reciprocal
		ymm one = vecsetvalue(1.0f);
		vec_8 = vecadd(one, vec_8);
		vec_8 = vecdiv(one, vec_8);

		ymm one_minus = vecsub(vecsetvalue(1.0f), vec_8);
		vec_8 = vecmul(vec_8, one_minus);

		// store back into result vector
		vecstore(out.data() + cur, vec_8);
	}

	// handle residual computations
	vec_res_for(cur, vec.size()) {
		number inter = 1.0f / (1.0f + std::exp(-vec(cur)));
		out(cur) = inter * (1.0f - inter);
	}
}
#else
void sigmoid_derivative(Vector& vec, Vector& out) {
	Vector sig = sigmoid_vec(vec);
	out.resize(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		out(i) = sig(i) * (1.0f - sig(i));
	}
}
#endif

/*
    Vector Sigmoid Derivative (Elementwise)

    vec = [1/(1+exp(-x1)), 1/(1+exp(-x2)), 1/(1+exp(-x3)), ...]
    return = [vec1(1-vec1), vec2(1-vec2), vec3(1-vec3), ...]
*/
Vector precomputed_sigmoid_derivative(const Vector& vec) {
	Vector ret;
	precomputed_sigmoid_derivative(const_cast<Vector&>(vec), ret);
	return ret;
}
#if defined(USE_AVX_PRECOMPUTED_SIGMOID_DERIVATIVE) && defined(__AVX__)
void precomputed_sigmoid_derivative(Vector& vec, Vector& out) {
	out.resize(vec.size());

	// process floats 8 at a time
	vec_main_for(cur, vec.size()) {
		ymm vec_8 = vecload(vec.data() + cur);

		ymm one_minus = vecsub(vecsetvalue(1.0), vec_8);
		vec_8 = vecmul(vec_8, one_minus);

		// store back into result vector
		vecstore(out.data() + cur, vec_8);
	}

	// handle residual computations
	vec_res_for(cur, vec.size()) {
		out(cur) = vec(cur) * (1.0f - vec(cur));
	}
}
#else
void precomputed_sigmoid_derivative(Vector& vec, Vector& out) {
	out.resize(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		out(i) = vec(i) * (1.0f - vec(i));
	}
}
#endif

/*
    Vector Outer Product

    A = [a1, a2, a3, ...]
    B = [b1, b2, b3, ...]
    return = [a1*b1, a1*b2, a1*b3, ...; a2*b1, a2*b2, a2*b3, ...; a3*b1, a3*b2, a3*b3, ...; ...]
*/
Matrix outer_product(const Vector& a, const Vector& b) {
	Matrix ret(a.size(), b.size());
	outer_product(a, b, ret);
	return ret;
}
#if defined(USE_AVX_OUTER_PRODUCT) && defined(__AVX__)
void outer_product(const Vector& a, const Vector& b, Matrix& out) {
	assert((out.rows() == a.size() && out.cols() == b.size()));
	
	size_t rows = a.size();
	size_t cols = b.size();
	const number* a_ptr = a.data();
	const number* b_ptr = b.data();

	number* out_ptr = out.data();
	for (size_t row = 0; row < rows; ++row) {
		const ymm a_vec = vecsetvalue(a_ptr[row]);
		vec_main_for(col, cols) { 
			const ymm b_vec = vecload(b_ptr + col);
			const ymm out_vec = vecmul(a_vec, b_vec);
			vecstore(out_ptr + row * cols + col, out_vec);
		}
		vec_res_for(col, cols) {
			out(row, col) = a(row) * b(col);
		}
	}
}
#else
void outer_product(const Vector& a, const Vector& b, Matrix& out) {
	assert((out.rows() == a.size() && out.cols() == b.size()));
	size_t m = a.size();
	size_t n = b.size();
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			out(i, j) = a(i) * b(j);
		}
	}
}
#endif