#pragma once

#include "math.h"

#include <immintrin.h>

// Each lane belongs to a different BC1 block.
struct Float3x4
{
    __m128 r; // Red values for blocks 0, 1, 2, and 3.
    __m128 g; // Green values for blocks 0, 1, 2, and 3.
    __m128 b; // Blue values for blocks 0, 1, 2, and 3.
};

inline Float3x4 operator+(const Float3x4 &a, const Float3x4 &b)
{
    return {_mm_add_ps(a.r, b.r), _mm_add_ps(a.g, b.g), _mm_add_ps(a.b, b.b)};
}

inline Float3x4 operator-(const Float3x4 &a, const Float3x4 &b)
{
    return {_mm_sub_ps(a.r, b.r), _mm_sub_ps(a.g, b.g), _mm_sub_ps(a.b, b.b)};
}

inline Float3x4 operator*(const Float3x4 &color, __m128 scale)
{
    return {_mm_mul_ps(color.r, scale), _mm_mul_ps(color.g, scale), _mm_mul_ps(color.b, scale)};
}

inline Float3x4 &operator+=(Float3x4 &a, const Float3x4 &b)
{
    a = a + b;
    return a;
}

// Compute one dot product per block without mixing lanes.
inline __m128 dot(const Float3x4& a, const Float3x4& b)
{
    const __m128 r = _mm_mul_ps(a.r, b.r);
    const __m128 g = _mm_mul_ps(a.g, b.g);
    const __m128 b_value = _mm_mul_ps(a.b, b.b);

    return _mm_add_ps(_mm_add_ps(r, g), b_value);
}

// Compute one squared vector length per block.
inline __m128 length_sq(const Float3x4& value)
{
    return dot(value, value);
}

// Store four symmetric covariance matrices, one per lane.
struct SymMat3x4
{
    __m128 rr;
    __m128 rg;
    __m128 rb;
    __m128 gg;
    __m128 gb;
    __m128 bb;
};

// Accumulate a weighted outer product for each block.
inline void accumulate_outer(SymMat3x4& matrix, const Float3x4& value, __m128 weight)
{
    const __m128 rr = _mm_mul_ps(value.r, value.r);
    const __m128 rg = _mm_mul_ps(value.r, value.g);
    const __m128 rb = _mm_mul_ps(value.r, value.b);
    const __m128 gg = _mm_mul_ps(value.g, value.g);
    const __m128 gb = _mm_mul_ps(value.g, value.b);
    const __m128 bb = _mm_mul_ps(value.b, value.b);

    matrix.rr = _mm_add_ps(matrix.rr, _mm_mul_ps(rr, weight));
    matrix.rg = _mm_add_ps(matrix.rg, _mm_mul_ps(rg, weight));
    matrix.rb = _mm_add_ps(matrix.rb, _mm_mul_ps(rb, weight));
    matrix.gg = _mm_add_ps(matrix.gg, _mm_mul_ps(gg, weight));
    matrix.gb = _mm_add_ps(matrix.gb, _mm_mul_ps(gb, weight));
    matrix.bb = _mm_add_ps(matrix.bb, _mm_mul_ps(bb, weight));
}

// Multiply each block's matrix by its own vector.
inline Float3x4 multiply(const SymMat3x4& matrix, const Float3x4& value)
{
    const __m128 r0 = _mm_mul_ps(matrix.rr, value.r);
    const __m128 r1 = _mm_mul_ps(matrix.rg, value.g);
    const __m128 r2 = _mm_mul_ps(matrix.rb, value.b);

    const __m128 g0 = _mm_mul_ps(matrix.rg, value.r);
    const __m128 g1 = _mm_mul_ps(matrix.gg, value.g);
    const __m128 g2 = _mm_mul_ps(matrix.gb, value.b);

    const __m128 b0 = _mm_mul_ps(matrix.rb, value.r);
    const __m128 b1 = _mm_mul_ps(matrix.gb, value.g);
    const __m128 b2 = _mm_mul_ps(matrix.bb, value.b);

    return {_mm_add_ps(_mm_add_ps(r0, r1), r2), _mm_add_ps(_mm_add_ps(g0, g1), g2), _mm_add_ps(_mm_add_ps(b0, b1), b2)};
}

// Choose b where the mask is true; otherwise choose a.
inline Float3x4 select(const Float3x4& a, const Float3x4& b, __m128 mask)
{
    return {_mm_blendv_ps(a.r, b.r, mask), _mm_blendv_ps(a.g, b.g, mask), _mm_blendv_ps(a.b, b.b, mask)};
}

// Estimate one color axis per block.
inline Float3x4 compute_principal_axis(const SymMat3x4& cov)
{
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);

    const __m128 rr = _mm_mul_ps(cov.rr, cov.rr);
    const __m128 gg = _mm_mul_ps(cov.gg, cov.gg);
    const __m128 bb = _mm_mul_ps(cov.bb, cov.bb);
    const __m128 rg = _mm_mul_ps(cov.rg, cov.rg);
    const __m128 rb = _mm_mul_ps(cov.rb, cov.rb);
    const __m128 gb = _mm_mul_ps(cov.gb, cov.gb);

    const __m128 diagonal_sq = _mm_add_ps(_mm_add_ps(rr, gg), bb);
    const __m128 off_diagonal_sq = _mm_add_ps(_mm_add_ps(rg, rb), gb);
    const __m128 doubled = _mm_mul_ps(off_diagonal_sq, _mm_set1_ps(2.0f));
    const __m128 norm_sq = _mm_add_ps(diagonal_sq, doubled);
    const __m128 threshold = _mm_mul_ps(norm_sq, _mm_set1_ps(1e-6f));

    const __m128 seed3 = _mm_set1_ps(inv_sqrt3);
    const Float3x4 axis0 = {seed3, seed3, seed3};

    Float3x4 next = multiply(cov, axis0);
    __m128 next_length_sq = length_sq(next);
    __m128 mask = _mm_cmple_ps(next_length_sq, threshold);

    // Skip alternate seeds when all four blocks have a valid axis.
    if (_mm_movemask_ps(mask) == 0)
        return next * _mm_rsqrt_ps(next_length_sq);

    const __m128 seed2 = _mm_set1_ps(inv_sqrt2);
    const __m128 negative_seed2 = _mm_set1_ps(-inv_sqrt2);

    // Replace only the lanes that need another seed.
    const Float3x4 axis1 = {seed2, negative_seed2, zero};
    const Float3x4 next1 = multiply(cov, axis1);

    next_length_sq = _mm_blendv_ps(next_length_sq, length_sq(next1), mask);
    next = select(next, next1, mask);

    mask = _mm_cmple_ps(next_length_sq, threshold);

    const Float3x4 axis2 = {zero, seed2, negative_seed2};
    const Float3x4 next2 = multiply(cov, axis2);

    next_length_sq = _mm_blendv_ps(next_length_sq, length_sq(next2), mask);
    next = select(next, next2, mask);

    // Avoid dividing by zero in blocks with no color variation.
    const __m128 flat_mask = _mm_cmple_ps(next_length_sq, threshold);
    const __m128 safe_length_sq = _mm_blendv_ps(next_length_sq, one, flat_mask);

    const Float3x4 result = next * _mm_rsqrt_ps(safe_length_sq);
    const Float3x4 red_axis = {one, zero, zero};

    return select(result, red_axis, flat_mask);
}

// Clamp each block's RGB values to the range [0, 1].
inline Float3x4 clamp01(const Float3x4& color)
{
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);

    const __m128 r = _mm_max_ps(color.r, zero);
    const __m128 g = _mm_max_ps(color.g, zero);
    const __m128 b = _mm_max_ps(color.b, zero);

    return {_mm_min_ps(r, one), _mm_min_ps(g, one), _mm_min_ps(b, one)};
}