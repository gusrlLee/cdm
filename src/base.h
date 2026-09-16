#pragma once

#include <cstdint>
#include <cmath>
#if defined(__CUDACC__)
#define CDM_INLINE __host__ __device__ inline
#else
#define CDM_INLINE inline
#endif

// Raw 8-byte BC1 block layout (two endpoints + packed indices).
struct Block64
{
    uint16_t c0;
    uint16_t c1;
    uint32_t indices;
};

static_assert(sizeof(Block64) == 8);

// Generic 3-component vector, instantiated with float (scalar) or v4f (SSE 4-wide).
template <typename T>
struct Vec3T
{
    T r, g, b;
    CDM_INLINE static Vec3T zero() { return {T(0), T(0), T(0)}; }
    CDM_INLINE Vec3T operator+(const Vec3T &o) const { return {r + o.r, g + o.g, b + o.b}; }
    CDM_INLINE Vec3T operator-(const Vec3T &o) const { return {r - o.r, g - o.g, b - o.b}; }
    CDM_INLINE Vec3T operator*(const T &s) const { return {r * s, g * s, b * s}; }
    CDM_INLINE Vec3T &operator+=(const Vec3T &o)
    {
        r = r + o.r;
        g = g + o.g;
        b = b + o.b;
        return *this;
    }
};

template <typename T>
CDM_INLINE T dot(const Vec3T<T> &a, const Vec3T<T> &b)
{
    return a.r * b.r + a.g * b.g + a.b * b.b;
}

template <typename T>
CDM_INLINE T length_sq(const Vec3T<T> &v)
{
    return dot(v, v);
}

template <typename T>
CDM_INLINE Vec3T<T> clamp01(const Vec3T<T> &v)
{
    auto c = [](float x)
    { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); };
    return {c(v.r), c(v.g), c(v.b)};
}

// Symmetric 3x3 covariance matrix, accumulated incrementally via accumulate_outer.
template <typename T>
struct SymMat3T
{
    T rr, gg, bb;
    T rg, rb, gb;
    CDM_INLINE static SymMat3T zero() { return {T(0), T(0), T(0), T(0), T(0), T(0)}; }
    CDM_INLINE void accumulate_outer(const Vec3T<T> &d, const T &weight)
    {
        rr = rr + d.r * d.r * weight;
        gg = gg + d.g * d.g * weight;
        bb = bb + d.b * d.b * weight;
        rg = rg + d.r * d.g * weight;
        rb = rb + d.r * d.b * weight;
        gb = gb + d.g * d.b * weight;
    }

    CDM_INLINE Vec3T<T> multiply(const Vec3T<T> &v) const
    {
        return {
            rr * v.r + rg * v.g + rb * v.b,
            rg * v.r + gg * v.g + gb * v.b,
            rb * v.r + gb * v.g + bb * v.b};
    }
};

using Float3 = Vec3T<float>;
using SymMat3 = SymMat3T<float>;

// Principal axis for scalar Float3 (Fallback for degenerate axis)
CDM_INLINE Float3 compute_principal_axis(const SymMat3 &cov)
{
    const float inv_sqrt3 = 0.57735027f;
    Float3 axis = {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    Float3 next = cov.multiply(axis);
    float lsq = length_sq(next);

    // If variation is orthogonal to (1,1,1), test candidate orthogonal axes
    if (lsq < 1e-9f)
    {
        const float inv_sqrt2 = 0.70710678f;
        axis = {inv_sqrt2, -inv_sqrt2, 0.0f};
        next = cov.multiply(axis);
        lsq = length_sq(next);
        if (lsq < 1e-9f)
        {
            axis = {0.0f, inv_sqrt2, -inv_sqrt2};
            next = cov.multiply(axis);
            lsq = length_sq(next);
        }
    }

    float inv_len = (lsq > 1e-20f) ? (1.0f / sqrtf(lsq)) : 0.0f;
    return (lsq > 1e-20f) ? (next * inv_len) : Float3{1.0f, 0.0f, 0.0f};
}
