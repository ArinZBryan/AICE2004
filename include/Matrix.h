#pragma once
#include <vector>
#include <cassert>
#include <cstddef>

class Matrix {
    std::vector<float> _data;
    size_t _rows;
    size_t _cols;

public:
    Matrix();   
    Matrix(size_t nrows, size_t ncols);
    Matrix(size_t nrows, size_t ncols, const float& fill);
    Matrix(std::initializer_list<std::initializer_list<float>> init);
    
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);

    inline float& operator()(size_t row, size_t col) { assert((row < _rows && col < _cols)); return _data[col + row * _cols]; };
    inline const float& operator()(size_t row, size_t col) const { assert((row < _rows && col < _cols)); return _data[col + row * _cols]; };

    inline size_t rows() const { return _rows; };
    inline size_t cols() const { return _cols; };

    inline float* data() { return _data.data(); };
    inline const float* data() const { return _data.data(); };

    std::vector<std::vector<float>> toJaggedArrays() const;
};