#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstddef>

class Vector {
    std::vector<float> _data;

public:
    Vector();   
    Vector(size_t size);
    Vector(size_t size, const float& fill);
    explicit Vector(const std::vector<float>&);
    explicit Vector(std::vector<float>&&);
    Vector(std::initializer_list<float> init);
    ~Vector();

    Vector(const Vector& other);
    Vector& operator=(const Vector& other);
    Vector(Vector&& other);
    Vector& operator=(Vector&& other);

    inline operator const std::vector<float>&() const { return _data; };
    inline operator std::vector<float>&() { return _data; };

    inline float& operator()(size_t index) { assert((index < _data.size())); return _data[index]; };
    inline const float& operator()(size_t index) const { assert((index < _data.size())); return _data[index]; };

    inline size_t size() const { return _data.size(); }

    inline float* data() { return _data.data(); };
    inline const float* data() const { return _data.data(); };

    inline std::vector<float>::iterator begin() { return _data.begin(); };
    inline std::vector<float>::iterator end() { return _data.end(); };
    inline std::vector<float>::const_iterator begin() const { return _data.begin(); };
    inline std::vector<float>::const_iterator end() const { return _data.end(); };
};