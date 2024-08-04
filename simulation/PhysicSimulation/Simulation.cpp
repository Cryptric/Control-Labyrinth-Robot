//
// Created by gawain on 6/10/2024.
//

#include "Simulation.h"

#include <iostream>

Simulation::Simulation(const std::vector<Wall> &walls, const std::vector<Hole> &holes, const std::vector<std::vector<int>>& scores) : re(std::random_device{}()), controlSignalDist(MIN_ANGLE, MAX_ANGLE) {
    this->walls = walls;
    this->holes = holes;
    this->scores = scores;

    ballPosition.x = 170;
    ballPosition.y = 235;
    ballVelocity.x = 0;
    ballVelocity.y = 0;

#ifdef DISPLAY
    gBall.setRadius(MM_2_PX(BALL_RADIUS));
    gBall.setPosition(VEC2_2_SFML(POSVEC_2_VECPX(ballPosition)));
#endif
}

Simulation::Simulation(const std::vector<Wall>& walls, const std::vector<Hole>& holes) : re(std::random_device{}()), controlSignalDist(MIN_ANGLE, MAX_ANGLE) {
    this->walls = walls;
    this->holes = holes;

    ballPosition.x = 150;
    ballPosition.y = 220;
    ballVelocity.x = 0;
    ballVelocity.y = 0;

#ifdef DISPLAY
    gBall.setRadius(MM_2_PX(BALL_RADIUS));
    gBall.setPosition(VEC2_2_SFML(POSVEC_2_VECPX((ballPosition - Vector2(BALL_RADIUS, -BALL_RADIUS)))));
#endif
}

void Simulation::step() {
    servoUpdate();

    ballPosition = ballPosition + DT * ballVelocity;
    checkCollisions();
    ballVelocity = ballVelocity + DT * 5.0f/7.0f * g * Vector2::sin(angle);
}

void Simulation::setNegGradField(Vector2 grad[MAX_Y][MAX_X]) {
    for (int i = 0; i < MAX_Y; i++) {
        for (int j = 0; j < MAX_X; j++) {
            negGradField[i][j] = grad[i][j];
        }
    }
}

#ifdef DISPLAY
void Simulation::draw() {

    gBall.setPosition(VEC2_2_SFML(POSVEC_2_VECPX(((ballPosition - Vector2(BALL_RADIUS, -BALL_RADIUS))))));
    window.draw(gBall);
    for (Wall& w : walls) {
        w.draw(window);
    }

    for (Hole& h : holes) {
        h.draw(window);
    }
}
#endif

void Simulation::servoUpdate() {
    Vector2 deltaAlpha = targetAngle - angle;
    deltaAlpha = { fmaxf(fminf(SERVO_SPEED, deltaAlpha.x), -SERVO_SPEED), fmaxf(fminf(SERVO_SPEED, deltaAlpha.y), -SERVO_SPEED)};
    angle = angle + deltaAlpha;
}

void Simulation::checkCollisions() {
    bool collided = false;
    float minLambda = 0;
    Vector2 reflectedVelocity;

    for (Wall& w : walls) {
        if (auto [collision, lambda, velocity] = w.checkCollision(ballPosition, ballVelocity); collision) {
            if (lambda < minLambda) {
                minLambda = lambda;
                reflectedVelocity = velocity;
                collided = true;
            }
        }
    }

    if (collided) {
        ballPosition = ballPosition + minLambda * ballVelocity;
        ballVelocity = reflectedVelocity;
    }

}

bool Simulation::checkHoleDrop() {
    return std::any_of(holes.begin(), holes.end(), [&](const Hole& h) { return h.checkCollision(ballPosition); });
}

bool Simulation::simulateControlSignal(const Vector2 signal) {
    targetAngle = signal;
    bool isFallen = false;
    for (int i = 0; i < ITERATIONS_PER_SIGNAL; i++) {
        step();
        if (checkHoleDrop()) {
            isFallen = true;
        }
    }
    return isFallen;
}

bool Simulation::simulateControlSignalVector(const std::vector<Vector2>& signals) {
    bool isFallen = false;
    for (const Vector2 signal : signals) {
        if (simulateControlSignal(signal)) {
            isFallen = true;
        }
    }
    return isFallen;
}

std::tuple<float, std::vector<Vector2>> Simulation::testLinearControlSignals(const Vector2& initalPosition, const Vector2& initialVelocity, const Vector2& initialBoardAngle, const std::vector<Vector2>& delayedSignals, float offset1, float offset2) {

    std::vector<Vector2> bestSignal = {};
    float bestCost = MAX_SCORE + 1;
    for (int i = 0; i < NUM_SLOPES; i++) {
        const float slope1 = CALC_SLOPE(i);
        for (int j = 0; j < NUM_SLOPES; j++) {
            const float slope2 = CALC_SLOPE(j);
            std::vector<Vector2> signal = Vector2::genLinearSignalSquence(offset1, slope1, offset2, slope2);
            for (int k = 0; k < delayedSignals.size(); k++) {
                signal[k] = delayedSignals[k];
            }
            if (const float score = evaluateSignalSequence(initalPosition, initialVelocity, initialBoardAngle, signal); score < bestCost) {
                bestCost = score;
                bestSignal = signal;
            }
        }
    }
    return {bestCost, bestSignal};
}

float Simulation::evaluateSignalSequence(const Vector2& initalPosition, const Vector2& initialVelocity, const Vector2& initialBoardAngle, const std::vector<Vector2>& signalSequence) {
    ballPosition = initalPosition;
    ballVelocity = initialVelocity;
    angle = initialBoardAngle;
    targetAngle = initialBoardAngle;
    if (simulateControlSignalVector(signalSequence)) {
        return MAX_SCORE;
    }
    float x = (ballPosition.x);
    float y = (ballPosition.y);

    int xp = static_cast<int>(ceilf(x));
    int yp = static_cast<int>(ceilf(y));
    int xm = static_cast<int>(floorf(x));
    int ym = static_cast<int>(floorf(y));

    if (0 <= xm && xp < 273 && 0 <= ym && yp < 227) {
        x = x - static_cast<float>(xm);
        y = y - static_cast<float>(ym);

        float score = (1.0f - x) * (1.0f - y) * scores[ym][xm] + x * (1.0f - y) * scores[ym][xp] + (1.0f - x) * y * scores[yp][xm] + x * y * scores[yp][xp];

        /********
                Vector2 negGrad = negGradField[(int) initalPosition.y][(int) initalPosition.x];
                float angle = negGrad.angle(signalSequence[delayedSignals.size()]);
                float scaledAngle = (((angle / M_PIf)) + 1);
                //std::cout << "factor: " << scaledAngle << std::endl;
                float newScore = score * scaledAngle;
        ********/
        if (score == 0 /* || isnanf(scaledAngle)*/) {
            score = MAX_SCORE;
        }

        return score;
    }
    return MAX_SCORE;
}


std::tuple<float, std::vector<Vector2>> Simulation::evaluateRandomSignalSequence(const Vector2& initalPosition, const Vector2& initialVelocity, const Vector2& initialBoardAngle, std::vector<Vector2> initSignal, const std::vector<Vector2>& delayedSignals) {
    std::vector<Vector2> signalSequence = {};
    if (initSignal.size() >= 15) {
        initSignal.erase(initSignal.begin());
        initSignal.push_back(Vector2::random3());
        signalSequence = initSignal;
        for (int i = 1; i <= 20; i++) {
            signalSequence[HORIZON - i] = Vector2::random3();
        }
    } else {
        signalSequence = Vector2::sampleSequence(re, controlSignalDist, HORIZON);
    }
    for (int i = 0; i < delayedSignals.size(); i++) {
        signalSequence[i] = delayedSignals[i];
    }

    float score = evaluateSignalSequence(initalPosition, initialVelocity, initialBoardAngle, signalSequence);

    return {score, signalSequence};
}





