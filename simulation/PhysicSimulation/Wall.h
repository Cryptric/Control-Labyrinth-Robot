//
// Created by gawain on 6/11/2024.
//

#ifndef PHYSICSIMULATION_WALL_H
#define PHYSICSIMULATION_WALL_H

#ifdef DISPLAY
#include <SFML/Graphics.hpp>
#endif
#include <cmath>

#include "Config.h"
#include "math/Vector2.h"

class Wall {

public:
    Wall(Vector2 p1, Vector2 p3);

    /**
     * p1 should be bottom left corner (lowest x and y coordinates)
     * p2 should be bottem right corner
     * ...
     */
    Vector2 p1;
    Vector2 p2;
    Vector2 p3;
    Vector2 p4;

    std::tuple<bool, float, Vector2> checkCollision(Vector2 ballPos, Vector2 ballVelocity) const;

    static std::vector<Wall> fromList(const std::vector<std::tuple<float, float, float, float>>& rectangles);

#ifdef DISPLAY
    void draw(sf::RenderWindow& window) const;
    sf::RectangleShape gWall;
#endif

};

#endif //PHYSICSIMULATION_WALL_H
