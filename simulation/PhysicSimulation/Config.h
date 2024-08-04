//
// Created by gawain on 6/10/2024.
//

#ifndef PHYSICSIMULATION_CONFIG_H
#define PHYSICSIMULATION_CONFIG_H

#include <tuple>
#include <vector>

#define DISPLAY

#ifdef DISPLAY
#include <SFML/Graphics.hpp>
inline sf::RenderWindow window(sf::VideoMode(1000, 1000), "Simple Shapes with SFML");
#endif



#define BALL_RADIUS 6
#define HOLE_RADIUS 10

#define CONTROL_PERIOD 0.02f
#define DT 0.005f
#define ITERATIONS_PER_SIGNAL (static_cast<int>(CONTROL_PERIOD / DT + 0.1))

#define HORIZON 30

#define MAX_SCORE 1000000.0f


#define MM_2_PX(a) (2 * a)
#define POSVEC_2_VECPX(v) (Vector2(2 * v.x + 20, -2 * v.y + 500))
#define VEC2_2_SFML(v) sf::Vector2f((float) v.x, (float) v.y)

#define g (9.81f * 1000)

// servo rotation speed in rad/DT
#define SERVO_SPEED (0.5 * DT)

#define NUM_THREADS 25

#define MIN_ANGLE (-0.02f)
#define MAX_ANGLE   0.02f

#define NUM_SLOPES 11
#define MAX_SLOPE 0.5f
#define CALC_SLOPE(n) (-MAX_SLOPE + static_cast<float>(n) * (2.0f * MAX_SLOPE / (NUM_SLOPES - 1)))
#define CALC_OFFSET(n) (-MAX_ANGLE + static_cast<float>(n) * (2.0f * MAX_ANGLE / 4))


inline bool debug = false;


inline std::vector<std::tuple<float, float, float, float>> wallPoints = {
    {41, 160, 133, 156},
    {128, 160, 133, 117},
    {83, 220, 88, 207},
    {88, 220, 194, 215},
    {194, 242, 199, 131},
    {199, 201, 215, 196},
    {174, 135, 199, 131},
    {174, 135, 179, 79},
    {129, 84, 179, 79},
    {238, 208, 259, 202},
    {259, 208, 265, 159},
    {238, 153, 244, 108},
    {218, 112, 244, 108},
    {218, 112, 224, 52},
    {197, 52, 248, 47},
    {197, 52, 202, 24},
    {34, 250, 313, 242},
    {306, 250, 313, 17},
    {34, 24, 313, 17},
    {34, 250, 41, 17}
};

inline std::vector<std::tuple<float, float>> holePoints = {
    {75, 198},
    {185, 206},
    {56, 71},
    {122, 36},
    {187, 79},
    {230, 139},
    {205, 185},
    {249, 194},
    {250, 151},
    {295, 108},
    {250, 82},
    {273, 39}
};

#define MAX_Y 273
#define MAX_X 227


#endif //PHYSICSIMULATION_CONFIG_H
