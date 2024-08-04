//
// Created by gawain on 6/10/2024.
//

#include "Vector2.h"



Vector2::Vector2() = default;

Vector2::Vector2(const float x, const float y) {
    this->x = x;
    this->y = y;
}


void Vector2::normalize() {
    const float norm = this->norm();
    x = x / norm;
    y = y / norm;
}

float Vector2::dot(const Vector2& v) const {
    return x * v.x + y * v.y;
}

float Vector2::angle(const Vector2& v) const {
    return std::acos(dot(v) / (norm() * v.norm()));
}


Vector2 Vector2::operator*(const float scalar) const {
    return {scalar * x, scalar * y};
}

Vector2 operator*(const float scalar, const Vector2 &v) {
    return v * scalar;
}

Vector2 Vector2::operator+(const Vector2 &v) const {
    return {x + v.x, y + v.y};
}

Vector2 Vector2::operator-(const float scalar) const {
    return {x - scalar, y - scalar};
}

Vector2 operator-(const float scalar, const Vector2 &v) {
    return v - scalar;
}

Vector2 Vector2::operator+(const float scalar) const {
    return {x + scalar, y + scalar};
}

Vector2 operator+(const float scalar, const Vector2 &v) {
    return v + scalar;
}

Vector2 Vector2::sin(const Vector2 &v) {
    return {std::sin(v.x), std::sin(v.y)};
}

float Vector2::distance(const Vector2 &v) const {
    return (*this - v).norm();
}

float Vector2::norm() const {
    return std::sqrt(x * x + y * y);
}

Vector2 Vector2::operator-(const Vector2 &v) const {
    return {x - v.x, y - v.y};
}

std::ostream& operator<<(std::ostream& os, const Vector2& v) {
    os << "{" << v.x << ", " << v.y << "}";
    return os;
}

Vector2 Vector2::random(std::mt19937_64& re, std::uniform_real_distribution<float>& dist) {
    float x = dist(re);
    float y = dist(re);
    return {x, y};
}

Vector2 Vector2::random3() {
    const int a = rand() % 3;
    const int b = rand() % 3;

    float x = 0;
    float y = 0;

    switch (a) {
        case 0: x = MIN_ANGLE; break;
        case 1: x = 0; break;
        case 2: x = MAX_ANGLE; break;
    }

    switch (b) {
        case 0: y = MIN_ANGLE; break;
        case 1: y = 0; break;
        case 2: y = MAX_ANGLE; break;
    }
    return {x, y};
}

std::vector<Vector2> Vector2::sampleSequence(std::mt19937_64& re, std::uniform_real_distribution<float>& dist, const size_t length) {
    std::vector<Vector2> vectors(length);
    for (size_t i = 0; i < length; i++) {
        const Vector2 v = random3();
        vectors[i] = v;
    }
    return vectors;
}

std::vector<Vector2> Vector2::genLinearSignalSquence(float offset1, float slope1, float offset2, float slope2) {
    std::vector<Vector2> signals(HORIZON);
    signals[0] = {offset1, offset2};
    for (int i = 1; i < HORIZON; i++) {
        if (fabsf(offset1 + slope1 * DT) > MAX_ANGLE) {
            slope1 *= -1;
        }
        if (fabsf(offset2 + slope2 * DT) > MAX_ANGLE) {
            slope2 *= -1;
        }
        offset1 += slope1 * DT;
        offset2 += slope2 * DT;
        signals[i] = {offset1, offset2};
    }
    return signals;
}

