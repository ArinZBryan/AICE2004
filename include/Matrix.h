#pragma once
#include <vector>
#include <cassert>
#include <cstddef>
#include "DataType.h"

class Matrix {
    std::vector<number> _data;
    size_t _rows;
    size_t _cols;

public:
    Matrix();   
    Matrix(size_t nrows, size_t ncols);
    Matrix(size_t nrows, size_t ncols, const number& fill);
    Matrix(std::initializer_list<std::initializer_list<number>> init);
    
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);

    inline number& operator()(size_t row, size_t col) { assert((row < _rows && col < _cols)); return _data[col + row * _cols]; };
    inline const number& operator()(size_t row, size_t col) const { assert((row < _rows && col < _cols)); return _data[col + row * _cols]; };

    inline size_t rows() const { return _rows; };
    inline size_t cols() const { return _cols; };
    inline size_t size() const { return _rows * _cols; }

    inline number* data() { return _data.data(); };
    inline const number* data() const { return _data.data(); };

    std::vector<std::vector<float>> toJaggedArrays() const;
};