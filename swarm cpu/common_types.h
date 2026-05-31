#pragma once
#include <cmath>

struct Vector2 {
    float x, y;
    Vector2 operator+(const Vector2& o) const { return {x+o.x, y+o.y}; }
    Vector2 operator-(const Vector2& o) const { return {x-o.x, y-o.y}; }
    Vector2 operator*(float s) const { return {x*s, y*s}; }
    float dot(const Vector2& o) const { return x*o.x + y*o.y; }
    float length() const { return sqrtf(x*x + y*y); }
    Vector2 normalized() const {
        float l = length();
        return (l < 1e-10f) ? Vector2{0,0} : Vector2{x/l, y/l};
    }
};

struct Agent {
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;
};

struct Obstacle {
    Vector2 position;
    float radius;
    bool is_wall;
    Vector2 wall_normal;
};

struct BetaAgent {
    Vector2 position;
    Vector2 velocity;
};

struct ConnectionVertex {   // Вершина для рендеринга
    float x, y;
    float r, g, b;
};
using Vertex = ConnectionVertex;

constexpr float WORLD_BOUNDARY = 10000.0f;
constexpr float PI = 3.14159265358979323846f;