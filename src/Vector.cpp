#include "Vector.h"

Vector::Vector() : _data() {}; 
Vector::Vector(size_t size) : _data(size) {};
Vector::Vector(size_t size, const float& fill) : _data(size, fill) {};
Vector::Vector(const std::vector<float>& from) : _data(from) {};
Vector::Vector(std::vector<float>&& from) : _data(std::move(from)) {};
Vector::Vector(std::initializer_list<float> init) : _data(init) {}; 
Vector::~Vector() = default;

Vector::Vector(const Vector& other) : _data(other._data) {}
Vector& Vector::operator=(const Vector& other) {
    _data = other._data;
    return *this;
};
Vector::Vector(Vector&& other) : _data(std::move(other._data)) {};
Vector& Vector::operator=(Vector&& other) {
    _data = std::move(other._data);
    return *this;
};