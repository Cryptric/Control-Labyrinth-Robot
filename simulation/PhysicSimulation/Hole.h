//
// Created by gawain on 26.06.24.
//

#ifndef HOLE_H
#define HOLE_H

#ifdef DISPLAY
#include <SFML/Graphics.hpp>
#endif

#include "math/Vector2.h"
#include "Config.h"



class Hole {
public:
    explicit Hole(Vector2 center);

    Vector2 center;

    bool checkCollision(Vector2 ballPos) const;

    static std::vector<Hole> fromList(const std::vector<std::tuple<float, float>>& centers);

#ifdef DISPLAY
    void draw(sf::RenderWindow& window) const;
    sf::CircleShape gHole;
#endif

};



#endif //HOLE_H
