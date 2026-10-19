#pragma once

#include "platform.h"

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
CDM_INLINE Float3 operator+(const Float3 &a, const Float3 &b)
{
    return {a.r + b.r, a.g + b.g, a.b + b.b};
}

// Scale all color components by the same value.
CDM_INLINE Float3 operator*(const Float3 &color, float scale)
{
    return {color.r * scale, color.g * scale, color.b * scale};
}

// Accumulate a color into an existing value.
CDM_INLINE Float3 &operator+=(Float3 &a, const Float3 &b)
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
CDM_INLINE Float3 operator-(const Float3 &a, const Float3 &b)
{
    return {a.r - b.r, a.g - b.g, a.b - b.b};
}

// Accumulate a weighted outer product into the matrix.
CDM_INLINE void accumulate_outer(SymMat3 &matrix, const Float3 &value, float weight)
{
    matrix.rr += value.r * value.r * weight;
    matrix.rg += value.r * value.g * weight;
    matrix.rb += value.r * value.b * weight;
    matrix.gg += value.g * value.g * weight;
    matrix.gb += value.g * value.b * weight;
    matrix.bb += value.b * value.b * weight;
}

// Compute the dot product of two vectors.
CDM_INLINE float dot(const Float3 &a, const Float3 &b)
{
    return a.r * b.r + a.g * b.g + a.b * b.b;
}

// Compute the squared vector length.
CDM_INLINE float length_sq(const Float3 &value)
{
    return dot(value, value);
}

// Multiply a symmetric matrix by a vector.
CDM_INLINE Float3 multiply(const SymMat3 &matrix, const Float3 &value)
{
    return {matrix.rr * value.r + matrix.rg * value.g + matrix.rb * value.b,
            matrix.rg * value.r + matrix.gg * value.g + matrix.gb * value.b,
            matrix.rb * value.r + matrix.gb * value.g + matrix.bb * value.b};
}

// Esitmate the color axis
CDM_INLINE Float3 compute_principal_axis(const SymMat3 &cov)
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

CDM_INLINE Float3 clamp01(const Float3 &color)
{
    const float r = color.r < 0.0f ? 0.0f : (color.r > 1.0f ? 1.0f : color.r);
    const float g = color.g < 0.0f ? 0.0f : (color.g > 1.0f ? 1.0f : color.g);
    const float b = color.b < 0.0f ? 0.0f : (color.b > 1.0f ? 1.0f : color.b);

    return {r, g, b};
}

struct Float4
{
    float r;
    float g;
    float b;
    float a;
};

// Add the corresponding RGBA components.
CDM_INLINE Float4 operator+(const Float4 &lhs, const Float4 &rhs)
{
    const float r = lhs.r + rhs.r;
    const float g = lhs.g + rhs.g;
    const float b = lhs.b + rhs.b;
    const float a = lhs.a + rhs.a;

    return {r, g, b, a};
}

// Scale all RGBA components by the same value.
CDM_INLINE Float4 operator*(const Float4 &value, float scale)
{
    const float r = value.r * scale;
    const float g = value.g * scale;
    const float b = value.b * scale;
    const float a = value.a * scale;

    return {r, g, b, a};
}

// Subtract the corresponding RGBA components.
CDM_INLINE Float4 operator-(const Float4 &lhs, const Float4 &rhs)
{
    const float r = lhs.r - rhs.r;
    const float g = lhs.g - rhs.g;
    const float b = lhs.b - rhs.b;
    const float a = lhs.a - rhs.a;

    return {r, g, b, a};
}

// Store the ten unique entries of a symmetric 4x4 covariance matrix.
struct SymMat4
{
    float rr;
    float rg;
    float rb;
    float ra;
    float gg;
    float gb;
    float ga;
    float bb;
    float ba;
    float aa;
};

CDM_INLINE Float4 multiply(const SymMat4 &matrix, const Float4 &value)
{
    const float r = matrix.rr * value.r + matrix.rg * value.g + matrix.rb * value.b + matrix.ra * value.a;
    const float g = matrix.rg * value.r + matrix.gg * value.g + matrix.gb * value.b + matrix.ga * value.a;
    const float b = matrix.rb * value.r + matrix.gb * value.g + matrix.bb * value.b + matrix.ba * value.a;
    const float a = matrix.ra * value.r + matrix.ga * value.g + matrix.ba * value.b + matrix.aa * value.a;

    return {r, g, b, a};
}

// Compute the dot product of two RGBA vectors.
CDM_INLINE float dot(const Float4 &lhs, const Float4 &rhs)
{
    const float r = lhs.r * rhs.r;
    const float g = lhs.g * rhs.g;
    const float b = lhs.b * rhs.b;
    const float a = lhs.a * rhs.a;

    const float result = r + g + b + a;
    return result;
}

// Compute the squared length without taking a square root.
CDM_INLINE float length_sq(const Float4 &value)
{
    const float result = dot(value, value);
    return result;
}

// Estimate an RGBA color axis using the same approach as the RGB path.
CDM_INLINE Float4 compute_principal_axis(const SymMat4 &cov)
{
    const float diagonal_sq = cov.rr * cov.rr + cov.gg * cov.gg + cov.bb * cov.bb + cov.aa * cov.aa;
    const float off_diagonal_sq =
        cov.rg * cov.rg + cov.rb * cov.rb + cov.ra * cov.ra + cov.gb * cov.gb + cov.ga * cov.ga + cov.ba * cov.ba;

    const float matrix_norm_sq = diagonal_sq + 2.0f * off_diagonal_sq;
    const float threshold = matrix_norm_sq * 1e-6f;

    // Match src2 fallback directions when the diagonal covariance response vanishes.
    constexpr Float4 seeds[4] = {{0.5f, 0.5f, 0.5f, 0.5f},
                                 {0.70710678f, -0.70710678f, 0.0f, 0.0f},
                                 {0.0f, 0.70710678f, -0.70710678f, 0.0f},
                                 {0.0f, 0.0f, 0.70710678f, -0.70710678f}};

    for (const Float4 &seed : seeds)
    {
        const Float4 next = multiply(cov, seed);
        const float next_length_sq = length_sq(next);

        if (next_length_sq > threshold)
        {
            // shortcut: one matrix step approximates PCA; revisit if quality is insufficient.
            const float inverse_length = 1.0f / std::sqrt(next_length_sq);
            const Float4 axis = next * inverse_length;
            return axis;
        }
    }

    // A constant block has no preferred direction.
    const Float4 axis = {1.0f, 0.0f, 0.0f, 0.0f};
    return axis;
}

// Clamp each RGBA component to the normalized range.
CDM_INLINE Float4 clamp01(const Float4 &color)
{
    const float r = color.r < 0.0f ? 0.0f : (color.r > 1.0f ? 1.0f : color.r);
    const float g = color.g < 0.0f ? 0.0f : (color.g > 1.0f ? 1.0f : color.g);
    const float b = color.b < 0.0f ? 0.0f : (color.b > 1.0f ? 1.0f : color.b);
    const float a = color.a < 0.0f ? 0.0f : (color.a > 1.0f ? 1.0f : color.a);

    return {r, g, b, a};
}