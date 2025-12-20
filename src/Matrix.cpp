#include "Matrix.h"

Matrix::Matrix() : _data(), _rows(0), _cols(0) {};
Matrix::Matrix(size_t nrows, size_t ncols) : _data(nrows * ncols), _rows(nrows), _cols(ncols) {};
Matrix::Matrix(size_t nrows, size_t ncols, const float& fill) : _data(nrows * ncols, fill), _rows(nrows), _cols(ncols) {};
Matrix::Matrix(std::initializer_list<std::initializer_list<float>> init) {
    std::vector<std::vector<float>> jagged;
    for (auto inner : init) {
        jagged.emplace_back(std::vector<float>(inner));
    }
    _rows = jagged.size();
    size_t columns = jagged[0].size();
    _cols = columns;
    _data = std::vector<float>(jagged.size() * jagged[0].size());
    for (size_t i = 0; i < _rows; i++) for (size_t j = 0; j < _cols; j++) {
        _data[j + i * columns] = jagged[i][j];
    }
}
Matrix::~Matrix() = default;

Matrix::Matrix(const Matrix& other) : _data(other._data.size()), _rows(other._rows), _cols(other._cols) {
    std::copy(other._data.begin(), other._data.end(), _data.begin());
};
Matrix& Matrix::operator=(const Matrix& other) {
    _data = std::vector<float>((const std::vector<float>&)other._data);
    _rows = other._rows;
    _cols = other._cols;
    return *this;
};
Matrix::Matrix(Matrix&& other) : _data(std::move(other._data)), _rows(other._rows), _cols(other._cols) {};
Matrix& Matrix::operator=(Matrix&& other) {
    _data = std::move(other._data);
    _rows = other._rows;
    _cols = other._cols;
    return *this;
};
std::vector<std::vector<float>> Matrix::toJaggedArrays() const {
    std::vector<std::vector<float>> jagged = std::vector<std::vector<float>>(_rows, std::vector<float>(_cols));
    for (size_t i = 0; i < _rows; i++) for (size_t j = 0; j < _cols; j++) {
        jagged[i][j] = (*this)(i, j);
    }
    return jagged;
}