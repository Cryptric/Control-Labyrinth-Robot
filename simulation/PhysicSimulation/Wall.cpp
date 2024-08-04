//
// Created by gawain on 6/11/2024.
//

#include "Wall.h"

Wall::Wall(const Vector2 p1, const Vector2 p3) {
    float minX = std::min(p1.x, p3.x);
    float minY = std::min(p1.y, p3.y);
    float maxX = std::max(p1.x, p3.x);
    float maxY = std::max(p1.y, p3.y);

    this->p1 = {minX, minY};
    this->p2 = {maxX, minY};
    this->p3 = {maxX, maxY};
    this->p4 = {minX, maxY};

#ifdef DISPLAY
    gWall.setSize(sf::Vector2f((float) (MM_2_PX((maxX - minX))), (float) (MM_2_PX((maxY - minY)))));
    gWall.setPosition(VEC2_2_SFML(POSVEC_2_VECPX(this->p4)));
    gWall.setOutlineColor(sf::Color::Red);
    gWall.setOutlineThickness(1);
    gWall.setFillColor(sf::Color::Transparent);
#endif
}

#ifdef DISPLAY
void Wall::draw(sf::RenderWindow &window) const {
    window.draw(gWall);
}
#endif

std::tuple<bool, float, Vector2> Wall::checkCollision(const Vector2 ballPos, Vector2 ballVelocity) const {
    float cmpX = ballPos.x;
    float cmpY = ballPos.y;
    bool xContact = false;
    bool yContact = false;

    if (ballPos.x <= p1.x) {
        cmpX = p1.x;
        xContact = true;
    } else if (ballPos.x >= p3.x) {
        cmpX = p3.x;
        xContact = true;
    }

    if (ballPos.y <= p1.y) {
        cmpY = p1.y;
        yContact = true;
    } else if (ballPos.y >= p3.y) {
        cmpY = p3.y;
        yContact = true;
    }

    const float distance = ballPos.distance({cmpX, cmpY});

    float lambda1 = 0;
    float lambda2 = 0;
    if (distance < BALL_RADIUS) {
        if (xContact && yContact) {
            const float a = ballVelocity.x;
            const float b = ballVelocity.y;
            const float u = ballPos.x;
            const float v = ballPos.y;

            const float sqrtTerm = sqrtf(powf(2 * a * u - 2 * a * cmpX + 2 * b * v - 2 * b * cmpY, 2) - 4 * (a * a + b * b) * (-(BALL_RADIUS * BALL_RADIUS) + u * u - 2 * u * cmpX + v * v - 2 * v * cmpY + cmpX * cmpX + cmpY * cmpY));
            const float term =  - 2 * a * u + 2 * a * cmpX - 2 * b * v + 2 * b * cmpY;
            const float diff = 2 * (a * a + b * b);
            lambda1 = (sqrtTerm + term) / diff;
            lambda2 = (-sqrtTerm + term) / diff;

            float lambda = lambda1;
            if (lambda2 <= 0 && (lambda1 > 0 || lambda1 < lambda2)) {
                lambda = lambda2;
            }

            const Vector2 ballPosOnCollision = ballPos + lambda * ballVelocity;
            Vector2 collisionVector = Vector2(cmpX, cmpY) - ballPosOnCollision;
            collisionVector.normalize();
            ballVelocity = ballVelocity - 2 * (ballVelocity.dot(collisionVector)) * collisionVector;
        } else if (xContact) {
            lambda1 = ( BALL_RADIUS + cmpX - ballPos.x) / ballVelocity.x;
            lambda2 = (-BALL_RADIUS + cmpX - ballPos.x) / ballVelocity.x;
            ballVelocity.x *= -1;
        } else if (yContact) {
            lambda1 = ( BALL_RADIUS + cmpY - ballPos.y) / ballVelocity.y;
            lambda2 = (-BALL_RADIUS + cmpY - ballPos.y) / ballVelocity.y;
            ballVelocity.y *= -1;
        } else {
            // i think this should not happen
        }
    }
    float lambda = lambda1;
    if (lambda2 <= 0 && (lambda1 > 0 || lambda1 < lambda2)) {
        lambda = lambda2;
    }

    return {distance < BALL_RADIUS, lambda, ballVelocity};
}

std::vector<Wall> Wall::fromList(const std::vector<std::tuple<float, float, float, float>>& rectangles) {
    std::vector<Wall> walls;
    for (std::tuple<float, float, float, float> r : rectangles) {
        float x1 = std::get<0>(r);
        float y1 = std::get<1>(r);
        float x2 = std::get<2>(r);
        float y2 = std::get<3>(r);
        walls.push_back(Wall({x1, y1}, {x2, y2}));
    }
    return walls;
}