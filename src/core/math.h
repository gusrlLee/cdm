#pragma once

#include <cmath>

constexpr float inv_sqrt3 = 0.57735027f;
constexpr float inv_sqrt2 = 0.70710678f;

struct Float3
{
    float r;
    float g;
    float b;
};

// Add the corresponding color components.
inline Float3 operator+(const Float3 &a, const Float3 &b)
{
    return {a.r + b.r, a.g + b.g, a.b + b.b};
}

// Scale all color components by the same value.
inline Float3 operator*(const Float3 &color, float scale)
{
    return {color.r * scale, color.g * scale, color.b * scale};
}

// Accumulate a color into an existing value.
inline Float3 &operator+=(Float3 &a, const Float3 &b)
{
    a.r += b.r;
    a.g += b.g;
    a.b += b.b;
    return a;
}

// Store the six unique entries of a symmetric 3x3 matrix.
struct SymMat3
{
    float rr;
    float rg;
    float rb;
    float gg;
    float gb;
    float bb;
};

// Subtract the corresponding color components.
inline Float3 operator-(const Float3 &a, const Float3 &b)
{
    return {a.r - b.r, a.g - b.g, a.b - b.b};
}

// Accumulate a weighted outer product into the matrix.
inline void accumulate_outer(SymMat3 &matrix, const Float3 &value, float weight)
{
    matrix.rr += value.r * value.r * weight;
    matrix.rg += value.r * value.g * weight;
    matrix.rb += value.r * value.b * weight;
    matrix.gg += value.g * value.g * weight;
    matrix.gb += value.g * value.b * weight;
    matrix.bb += value.b * value.b * weight;
}

// Compute the dot product of two vectors.
inline float dot(const Float3 &a, const Float3 &b)
{
    return a.r * b.r + a.g * b.g + a.b * b.b;
}

// Compute the squared vector length.
inline float length_sq(const Float3 &value)
{
    return dot(value, value);
}

// Multiply a symmetric matrix by a vector.
inline Float3 multiply(const SymMat3 &matrix, const Float3 &value)
{
    return {
        matrix.rr * value.r + matrix.rg * value.g + matrix.rb * value.b,
        matrix.rg * value.r + matrix.gg * value.g + matrix.gb * value.b,
        matrix.rb * value.r + matrix.gb * value.g + matrix.bb * value.b};
}

// Esitmate the color axis
inline Float3 compute_principal_axis(const SymMat3 &cov)
{
    const float diagonal_sq = cov.rr * cov.rr + cov.gg * cov.gg + cov.bb * cov.bb;
    const float off_diagonal_sq = cov.rg * cov.rg + cov.rb * cov.rb + cov.gb * cov.gb;

    const float matrix_norm_sq = diagonal_sq + 2.0f * off_diagonal_sq;
    const float threshold = matrix_norm_sq * 1e-6f;

    Float3 axis = {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    Float3 next = multiply(cov, axis);
    float next_length_sq = length_sq(next);

    // Try another seed if the response is too small.
    if (next_length_sq <= threshold)
    {
        axis = {inv_sqrt2, -inv_sqrt2, 0.0f};
        next = multiply(cov, axis);
        next_length_sq = length_sq(next);

        if (next_length_sq <= threshold)
        {
            axis = {0.0f, inv_sqrt2, -inv_sqrt2};
            next = multiply(cov, axis);
            next_length_sq = length_sq(next);
        }
    }

    // Use the red axis when the block has no color variation.
    if (next_length_sq <= threshold)
        return {1.0f, 0.0f, 0.0f};

    return next * (1.0f / std::sqrt(next_length_sq));
}

inline Float3 clamp01(const Float3 &color)
{
    const float r = color.r < 0.0f ? 0.0f : (color.r > 1.0f ? 1.0f : color.r);
    const float g = color.g < 0.0f ? 0.0f : (color.g > 1.0f ? 1.0f : color.g);
    const float b = color.b < 0.0f ? 0.0f : (color.b > 1.0f ? 1.0f : color.b);

    return {r, g, b};
}