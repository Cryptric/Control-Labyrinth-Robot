//
// Created by gawain on 6/10/2024.
//

#ifndef PHYSICSIMULATION_VECTOR2_H
#define PHYSICSIMULATION_VECTOR2_H

#include <cmath>
#include <random>
#include <ostream>


#include "../Config.h"



class Vector2 {

public:
    Vector2();
    Vector2(float x, float y);

    float x = 0;
    float y = 0;

    [[nodiscard]] float distance(const Vector2& v) const;

    [[nodiscard]] float norm() const;

    void normalize();

    [[nodiscard]] float dot(const Vector2& v) const;

    [[nodiscard]] float angle(const Vector2& v) const;

    Vector2 operator*(float scalar) const;
    friend Vector2 operator*(float scalar, const Vector2& v);

    Vector2 operator-(float scalar) const;
    friend Vector2 operator-(float scalar, const Vector2& v);

    Vector2 operator+(float scalar) const;
    friend Vector2 operator+(float scalar, const Vector2& v);

    Vector2 operator+(Vector2 const& v) const;
    Vector2 operator-(const Vector2& v) const;

    friend std::ostream& operator<<(std::ostream& os, const Vector2& v);

    static Vector2 random(std::mt19937_64& re, std::uniform_real_distribution<float>& dist);
    static Vector2 random3();
    static std::vector<Vector2> sampleSequence(std::mt19937_64& re, std::uniform_real_distribution<float>& dist, size_t length);
    static std::vector<Vector2> genLinearSignalSquence(float offset1, float slope1, float offset2, float slope2);

    static Vector2 sin(const Vector2& v);



};


#endif //PHYSICSIMULATION_VECTOR2_H
