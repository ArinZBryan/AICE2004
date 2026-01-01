#pragma once
#include <vector>
#include <ranges>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include "CompileConfig.h"
#include <algorithm>

class Vector {
    std::vector<number> _data;

public:
    Vector();   
    Vector(size_t size);
    Vector(size_t size, const number& fill);
    explicit Vector(const std::vector<float>&);
    Vector(std::initializer_list<number> init);
    ~Vector();

    Vector(const Vector& other);
    Vector& operator=(const Vector& other);
    Vector(Vector&& other);
    Vector& operator=(Vector&& other);

    inline operator const std::vector<number>&() const { return _data; };
    inline operator std::vector<number>&() { return _data; };

    inline number& operator()(size_t index) { assert((index < _data.size())); return _data[index]; };
    inline const number& operator()(size_t index) const { assert((index < _data.size())); return _data[index]; };

    inline size_t size() const { return _data.size(); }

    inline void resize(size_t new_size) { _data.resize(new_size); }
    inline void reserve(size_t new_capacity) { _data.reserve(new_capacity); }

    inline number* data() { return _data.data(); };
    inline const number* data() const { return _data.data(); };

    inline std::vector<number>::iterator begin() { return _data.begin(); };
    inline std::vector<number>::iterator end() { return _data.end(); };
    inline std::vector<number>::const_iterator begin() const { return _data.begin(); };
    inline std::vector<number>::const_iterator end() const { return _data.end(); };
};

std::vector<float> to_float_vector(const std::vector<number>& v);