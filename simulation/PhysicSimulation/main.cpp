#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <vector>
#ifdef DISPLAY
#include <SFML/Graphics.hpp>
#endif

#include "Config.h"
#include "Hole.h"
#include "Simulation.h"
#include "Wall.h"



uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::vector<Wall> wallsMM = {
    {{0, 138}, {95, 143}},
    {{90, 96}, {95, 143}},
    {{43, 205}, {48, 191}},
    {{43, 205}, {159, 200}},
    {{159, 227}, {164, 111}},
    {{164, 180}, {180, 185}},
    {{137, 116}, {164, 111}},
    {{137, 116}, {142, 57}},
    {{89, 57}, {142, 62}},
    {{206, 187}, {233, 192}},
    {{228, 192}, {233, 139}},
    {{205, 134}, {210, 86}},
    {{184, 91}, {210, 86}},
    {{184, 91}, {189, 22}},
    {{159, 27}, {214, 22}},
    {{159, 27}, {164, 0}},

    {{-5, 0}, {273, -5}},
    {{273 + 5, -5}, {278+5, 232}},
    {{278, 232 + 3}, {-5, 227 + 3}},
    {{-5, 227}, {0, -5}},
};

std::vector<std::tuple<float, float>> holesMM = {
    {32, 185},
    {10, 48},
    {150, 190},
    {150, 56},
    {196, 120},
    {171, 170},
    {219, 177},
    {220, 133},
    {219, 59},
    {265, 89},
    {242, 14},
    {80, 12}
};


std::vector<Wall> walls = wallsMM;
std::vector<Hole> holes = Hole::fromList(holesMM);

#ifdef DISPLAY
sf::VertexArray line(sf::LineStrip, 0);
sf::VertexArray predictedTrajectory(sf::LineStrip, 0);

sf::VertexArray simPreds[NUM_THREADS];

#endif

std::vector<Vector2> prevSignal = {};


Simulation playbackSim(walls, holes);
Simulation previewSim(walls, holes);


std::vector<Simulation> mpcSims;
std::vector<Vector2> threadSignal[NUM_THREADS];

double threadScore[NUM_THREADS];


Vector2 negGradientField[MAX_Y][MAX_X];

extern "C" {

    volatile void __attribute__((visibility("default"))) init() {
        std::ifstream  data("/home/gawain/Documents/PhysicSimulation/score3.csv");
        std::string line;
        std::vector<std::vector<int> > scores;
        while(std::getline(data,line)) {
            std::stringstream lineStream(line);
            std::string cell;
            std::vector<int> parsedRow;
            while(std::getline(lineStream,cell,',')) {
                parsedRow.push_back(static_cast<int>(std::stod(cell)));
            }
            scores.push_back(parsedRow);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            simPreds[i] = sf::VertexArray(sf::LinesStrip, 0);
        }

        for (auto & i : threadSignal) {
            mpcSims.emplace_back(walls, holes, scores);
            i = {};
        }
    }

    volatile void __attribute__((visibility("default"))) setVectorField(const float* dx, const float* dy) {
        for (int i = 0; i < MAX_Y; i++) {
            for (int j = 0; j < MAX_X; j++) {
                negGradientField[i][j] = {dx[j], dy[i]};
            }
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            mpcSims[i].setNegGradField(negGradientField);
        }
    }

    volatile void __attribute__((visibility("default"))) setPath(const float* data, const size_t size) {
#ifdef DISPLAY
        sf::Color orange(255, 165, 0);
        line.clear();
        for (size_t i = 0; i < size; i += 2) {
            line.append(sf::Vertex(VEC2_2_SFML(POSVEC_2_VECPX(Vector2(data[i], data[i+1]))), orange));
        }
#endif
    }

    volatile void __attribute__((visibility("default"))) setControlSignal(float* signal) {
        playbackSim.targetAngle = {signal[0], signal[1]};
    }

    volatile void __attribute__((visibility("default"))) step() {
        playbackSim.step();
    }

    volatile void __attribute__((visibility("default"))) closeSimulationPlayback() {
#ifdef DISPLAY
        window.close();
#endif
    }

    volatile bool __attribute__((visibility("default"))) drawSim() {
#ifdef DISPLAY
        window.clear();
        window.draw(line);
        playbackSim.draw();
        window.draw(predictedTrajectory);
        window.display();
        sf::Event event;

        window.pollEvent(event);
        if (event.type == sf::Event::Closed) {
            window.close();
        }
        return window.isOpen();
#else
        return true;
#endif
    }

    volatile void __attribute__((visibility("default"))) getState(float* position, float* velocity, float* angle) {
        position[0] = playbackSim.ballPosition.x;
        position[1] = playbackSim.ballPosition.y;
        velocity[0] = playbackSim.ballVelocity.x;
        velocity[1] = playbackSim.ballVelocity.y;
        angle[0] = playbackSim.angle.x;
        angle[1] = playbackSim.angle.y;
    }

    volatile void __attribute__((visibility("default"))) sampleControlSignal(const float* pos, float* velocity, float* boardAngle, const float* delayedSignals, const size_t stepsDeadtime, float* signal, float* outputPredicted) {

        Vector2 ballPosition = {pos[0], pos[1]};

        bool collided = false;

        for (Wall& w : walls) {
            if (auto [collision, lambda, reflectedVelocity] = w.checkCollision(ballPosition, {0, 0}); collision) {
                collided = true;
                break;
            }
        }

        if (collided) {
            float minLambda = 0;
            Vector2 displacementDirection;
            std::cout << "ball inside wall" << std::endl;

            for (Wall& w : walls) {
                if (auto [collision, lambda, reflectedVelocity] = w.checkCollision(ballPosition, {1, 0}); collision) {
                    if (lambda > -10 && lambda < minLambda) {
                        minLambda = lambda;
                        displacementDirection = {1, 0};
                    }
                }
            }

            for (Wall& w : walls) {
                if (auto [collision, lambda, reflectedVelocity] = w.checkCollision(ballPosition, {-1, 0}); collision) {
                    if (lambda > -10 && lambda < minLambda) {
                        minLambda = lambda;
                        displacementDirection = {-1, 0};
                    }
                }
            }

            for (Wall& w : walls) {
                if (auto [collision, lambda, reflectedVelocity] = w.checkCollision(ballPosition, {0, 1}); collision) {
                    if (lambda > -10 && lambda < minLambda) {
                        minLambda = lambda;
                        displacementDirection = {0, 1};
                    }
                }
            }

            for (Wall& w : walls) {
                if (auto [collision, lambda, reflectedVelocity] = w.checkCollision(ballPosition, {0, -1}); collision) {
                    if (lambda > -10 && lambda < minLambda) {
                        minLambda = lambda;
                        displacementDirection = {0, -1};
                    }
                }
            }
        }

        std::vector<Vector2> signalsDeadTime(stepsDeadtime);
        for (int i = 0; i < stepsDeadtime; i++) {
            Vector2 v = {delayedSignals[2 * i], delayedSignals[2 * i + 1]};
            signalsDeadTime[i] = v;
        }

        std::fill_n(threadScore, NUM_THREADS, MAX_SCORE);
        #pragma omp parallel for num_threads(25) default(shared)
        for (int i = 0; i < NUM_THREADS; i++) {
            int n1 = i % 5;
            int n2 = i / 5;
            float offset1 = CALC_OFFSET(n1);
            float offset2 = CALC_OFFSET(n2);
            auto [score, signal] = mpcSims[i].testLinearControlSignals(ballPosition, {velocity[0], velocity[1]}, {boardAngle[0], boardAngle[1]}, signalsDeadTime, offset1, offset2);
            threadScore[i] = score;
            threadSignal[i] = signal;
        }

        double bestScore = MAX_SCORE;


        std::vector<Vector2> bestSignal;
        std::vector<Vector2> bestPred;
        for (int i = 0; i < NUM_THREADS; i++) {
            if (threadScore[i] < bestScore) {
                bestScore = threadScore[i];
                bestSignal = threadSignal[i];
            }

        }

        std::vector<Vector2> posPreview = {};
        previewSim.ballPosition = ballPosition;
        previewSim.ballVelocity = {velocity[0], velocity[1]};
        previewSim.angle = {boardAngle[0], boardAngle[1]};
        posPreview.emplace_back(previewSim.ballPosition);
        int i = 0;
        for (const Vector2 s : bestSignal) {
            previewSim.simulateControlSignal(s);
            posPreview.emplace_back(previewSim.ballPosition);
            outputPredicted[i++] = previewSim.ballPosition.x;
            outputPredicted[i++] = previewSim.ballPosition.y;
        }

#ifdef DISPLAY
        predictedTrajectory.clear();
        for (Vector2 const p : posPreview) {
            predictedTrajectory.append(sf::Vertex(VEC2_2_SFML(POSVEC_2_VECPX(p)), sf::Color::Magenta));
        }
#endif

        playbackSim.ballPosition = ballPosition;

        prevSignal = bestSignal;

        if (!bestSignal.empty()) {
            std::cout << "signal sampled with score: " << bestScore << ", signal: " << bestSignal[2] << std::endl;
            signal[0] = bestSignal[stepsDeadtime].x;
            signal[1] = bestSignal[stepsDeadtime].y;
        }
    }


    volatile void __attribute__((visibility("default"))) runSim() {
        bool running = true;

        float prevSignal[] = {0, 0};
        float prevPrevSignal[] = {0, 0};

        Vector2 prevPos = playbackSim.ballPosition;
        Vector2 prevPrevPos = playbackSim.ballPosition;

        Vector2 prevVelocity = playbackSim.ballVelocity;
        Vector2 prevPrevVelocity = playbackSim.ballVelocity;

        bool overwrite = false;

        while (running) {
            const float pos[] = { prevPrevPos.x, prevPrevPos.y};
            float velocity[] = { prevPrevVelocity.x, prevPrevVelocity.y};
            float angle[] = {0, 0};
            const float delayedSignals[] = {prevPrevSignal[0], prevPrevSignal[1], prevSignal[0], prevSignal[1]};
            float signal[] = {0, 0};
            float _pred[2 * HORIZON];

            const uint64_t t = timeSinceEpochMillisec();
            sampleControlSignal(pos, velocity, angle, delayedSignals, 2, signal, _pred);
            uint dt = timeSinceEpochMillisec() - t;

            std::cout << "Sampled signal in: " << dt << "ms" << std::endl;

            prevPrevSignal[0] = prevSignal[0];
            prevPrevSignal[1] = prevSignal[1];

            prevSignal[0] = signal[0];
            prevSignal[1] = signal[1];

            if (overwrite) {
                signal[0] = MAX_ANGLE;
                signal[1] = MAX_ANGLE;
            }
            // std::cout << "[" << signal[0] << ", " << signal[1] << "]" << std::endl;
            playbackSim.simulateControlSignal({signal[0], signal[1]});

            prevPrevPos = prevPos;
            prevPos = playbackSim.ballPosition;

            prevPrevVelocity = prevVelocity;
            prevVelocity = playbackSim.ballVelocity;


            running = drawSim();

        }
    }

}

int main() {

    init();
    runSim();

    return 0;
}
