//
// Created by gawain on 26.06.24.
//

#include "Hole.h"

Hole::Hole(const Vector2 center) {
    this->center = center;

#ifdef DISPLAY
    gHole.setRadius(MM_2_PX(HOLE_RADIUS));
    gHole.setPosition(VEC2_2_SFML(POSVEC_2_VECPX((center - Vector2(HOLE_RADIUS, -HOLE_RADIUS)))));
    gHole.setOutlineColor(sf::Color::Green);
    gHole.setOutlineThickness(1);
    gHole.setFillColor(sf::Color::Transparent);
#endif
}

std::vector<Hole> Hole::fromList(const std::vector<std::tuple<float, float>>& centers) {
    std::vector<Hole> holes;
    holes.reserve(centers.size());
    for (std::tuple<float, float> c : centers) {
        holes.emplace_back(Vector2(std::get<0>(c), std::get<1>(c)));
    }
    return holes;
}

bool Hole::checkCollision(const Vector2 ballPos) const {
    return center.distance(ballPos) <= (HOLE_RADIUS);
}

#ifdef DISPLAY
void Hole::draw(sf::RenderWindow &window) const {
    window.draw(gHole);
}
#endif