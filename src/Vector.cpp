#include "Vector.h"

Vector::Vector() : _data() {}; 
Vector::Vector(size_t size) : _data(size) {};
Vector::Vector(size_t size, const number& fill) : _data(size, fill) {};
Vector::Vector(const std::vector<float>& from) : _data(from.size()) {
    for (size_t i = 0; i < from.size(); i++) {
        _data[i] = static_cast<number>(from[i]);
    }
};
Vector::Vector(std::initializer_list<number> init) : _data(init) {}; 
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

std::vector<float> to_float_vector(const std::vector<number>& v) {
    std::vector<float> out(v.size());
    std::ranges::transform(v, out.begin(),
                           [](number x) { return static_cast<float>(x); });
    return out;
}