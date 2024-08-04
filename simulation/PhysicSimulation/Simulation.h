//
// Created by gawain on 6/10/2024.
//

#ifndef PHYSICSIMULATION_SIMULATION_H
#define PHYSICSIMULATION_SIMULATION_H


#ifdef DISPLAY
#include <SFML/Graphics.hpp>
#endif
#include <random>
#include <algorithm>

#include "Config.h"
#include "Hole.h"
#include "math/Vector2.h"
#include "Wall.h"

class Simulation {
    std::vector<Wall> walls;
    std::vector<Hole> holes;
    std::vector<std::vector<int>> scores;

    std::mt19937_64 re;
    std::uniform_real_distribution<float> controlSignalDist;
    Vector2 negGradField[MAX_Y][MAX_X];

    void servoUpdate();

#ifdef DISPLAY
    sf::CircleShape gBall;
#endif


public:
    Vector2 ballPosition;
    Vector2 ballVelocity;
    Vector2 targetAngle;
    Vector2 angle;


    Simulation(const std::vector<Wall>& walls, const std::vector<Hole> &holes, const std::vector<std::vector<int>>& scores);
    Simulation(const std::vector<Wall>& walls, const std::vector<Hole> &holes);
    void step();

    bool checkHoleDrop();
    void setNegGradField(Vector2 grad[MAX_Y][MAX_X]);

    void checkCollisions();

    /* will apply the control signal and then simulate for one control period */
    bool simulateControlSignal(Vector2 signal);
    /* will simulate all control signals in the vector by consecutive calls to simulateControlSignal, starting at index 0 */
    bool simulateControlSignalVector(const std::vector<Vector2>& signals);

    std::tuple<float, std::vector<Vector2>> testLinearControlSignals(const Vector2& initalPosition, const Vector2& initialVelocity, const Vector2& initialBoardAngle, const std::vector<Vector2>& delayedSignals, float offset1, float offset2);
    float evaluateSignalSequence(const Vector2& initalPosition, const Vector2& initialVelocity, const Vector2& initialBoardAngle, const std::vector<Vector2>& signalSequence);

    std::tuple<float, std::vector<Vector2>> evaluateRandomSignalSequence(const Vector2& initalPosition, const Vector2& initialVelocity, const Vector2& initialBoardAngle, std::vector<Vector2> initSignal, const std::vector<Vector2>& delayedSignals);



#ifdef DISPLAY
    void draw();
#endif

};


#endif //PHYSICSIMULATION_SIMULATION_H
