#pragma once

#include <cstdint>
#include <cmath>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
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
template <typename T> struct Vec3T
{
    T r, g, b;
    CDM_INLINE static Vec3T zero()
    {
        return {T(0), T(0), T(0)};
    }
    CDM_INLINE Vec3T operator+(const Vec3T &o) const
    {
        return {r + o.r, g + o.g, b + o.b};
    }
    CDM_INLINE Vec3T operator-(const Vec3T &o) const
    {
        return {r - o.r, g - o.g, b - o.b};
    }
    CDM_INLINE Vec3T operator*(const T &s) const
    {
        return {r * s, g * s, b * s};
    }
    CDM_INLINE Vec3T &operator+=(const Vec3T &o)
    {
        r = r + o.r;
        g = g + o.g;
        b = b + o.b;
        return *this;
    }
};

template <typename T> CDM_INLINE T dot(const Vec3T<T> &a, const Vec3T<T> &b)
{
    return a.r * b.r + a.g * b.g + a.b * b.b;
}

template <typename T> CDM_INLINE T length_sq(const Vec3T<T> &v)
{
    return dot(v, v);
}

template <typename T> CDM_INLINE Vec3T<T> clamp01(const Vec3T<T> &v)
{
    auto c = [](float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); };
    return {c(v.r), c(v.g), c(v.b)};
}

// Symmetric 3x3 covariance matrix, accumulated incrementally via accumulate_outer.
template <typename T> struct SymMat3T
{
    T rr, gg, bb;
    T rg, rb, gb;
    CDM_INLINE static SymMat3T zero()
    {
        return {T(0), T(0), T(0), T(0), T(0), T(0)};
    }
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
        return {rr * v.r + rg * v.g + rb * v.b, rg * v.r + gg * v.g + gb * v.b, rb * v.r + gb * v.g + bb * v.b};
    }
};

using Float3 = Vec3T<float>;
using SymMat3 = SymMat3T<float>;

template <typename T> struct Vec4T
{
    T r, g, b, a;
    CDM_INLINE static Vec4T zero()
    {
        return {T(0), T(0), T(0), T(0)};
    }
    CDM_INLINE Vec4T operator+(const Vec4T &o) const
    {
        return {r + o.r, g + o.g, b + o.b, a + o.a};
    }
    CDM_INLINE Vec4T operator-(const Vec4T &o) const
    {
        return {r - o.r, g - o.g, b - o.b, a - o.a};
    }
    CDM_INLINE Vec4T operator*(const T &s) const
    {
        return {r * s, g * s, b * s, a * s};
    }
    CDM_INLINE Vec4T &operator+=(const Vec4T &o)
    {
        r = r + o.r;
        g = g + o.g;
        b = b + o.b;
        a = a + o.a;
        return *this;
    }
};

template <typename T> CDM_INLINE T dot(const Vec4T<T> &x, const Vec4T<T> &y)
{
    return x.r * y.r + x.g * y.g + x.b * y.b + x.a * y.a;
}
template <typename T> CDM_INLINE T length_sq(const Vec4T<T> &v)
{
    return dot(v, v);
}

template <typename T> struct SymMat4T
{
    T rr, gg, bb, aa, rg, rb, ra, gb, ga, ba;
    CDM_INLINE static SymMat4T zero()
    {
        return {T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)};
    }
    CDM_INLINE void accumulate_outer(const Vec4T<T> &d, const T &w)
    {
        rr = rr + d.r * d.r * w;
        gg = gg + d.g * d.g * w;
        bb = bb + d.b * d.b * w;
        aa = aa + d.a * d.a * w;
        rg = rg + d.r * d.g * w;
        rb = rb + d.r * d.b * w;
        ra = ra + d.r * d.a * w;
        gb = gb + d.g * d.b * w;
        ga = ga + d.g * d.a * w;
        ba = ba + d.b * d.a * w;
    }
    CDM_INLINE Vec4T<T> multiply(const Vec4T<T> &v) const
    {
        return {rr * v.r + rg * v.g + rb * v.b + ra * v.a, rg * v.r + gg * v.g + gb * v.b + ga * v.a,
                rb * v.r + gb * v.g + bb * v.b + ba * v.a, ra * v.r + ga * v.g + ba * v.b + aa * v.a};
    }
};

using Float4 = Vec4T<float>;
using SymMat4 = SymMat4T<float>;

CDM_INLINE Float4 compute_principal_axis(const SymMat4 &cov)
{
    // Scale before measuring columns so low-contrast covariance is not mistaken for zero.
    const float scale = fmaxf(fmaxf(fmaxf(fabsf(cov.rr), fabsf(cov.gg)), fmaxf(fabsf(cov.bb), fabsf(cov.aa))),
                              fmaxf(fmaxf(fmaxf(fabsf(cov.rg), fabsf(cov.rb)), fmaxf(fabsf(cov.ra), fabsf(cov.gb))),
                                    fmaxf(fabsf(cov.ga), fabsf(cov.ba))));
    if (!(scale > 0.0f) || !isfinite(scale))
        return {1.0f, 0.0f, 0.0f, 0.0f};

    const float inv_scale = 1.0f / scale;
    Float4 axis = {cov.rr * inv_scale, cov.rg * inv_scale, cov.rb * inv_scale, cov.ra * inv_scale};
    float best_norm = length_sq(axis);
    Float4 candidate = {cov.rg * inv_scale, cov.gg * inv_scale, cov.gb * inv_scale, cov.ga * inv_scale};
    float n = length_sq(candidate);
    if (n > best_norm)
    {
        axis = candidate;
        best_norm = n;
    }
    candidate = {cov.rb * inv_scale, cov.gb * inv_scale, cov.bb * inv_scale, cov.ba * inv_scale};
    n = length_sq(candidate);
    if (n > best_norm)
    {
        axis = candidate;
        best_norm = n;
    }
    candidate = {cov.ra * inv_scale, cov.ga * inv_scale, cov.ba * inv_scale, cov.aa * inv_scale};
    n = length_sq(candidate);
    if (n > best_norm)
    {
        axis = candidate;
        best_norm = n;
    }
    return axis * (1.0f / sqrtf(best_norm));
}

// Use deterministic fallback directions when the first covariance response vanishes.
CDM_INLINE Float3 compute_principal_axis(const SymMat3 &cov)
{
    const float matrix_norm_sq = cov.rr * cov.rr + cov.gg * cov.gg + cov.bb * cov.bb +
                                 2.0f * (cov.rg * cov.rg + cov.rb * cov.rb + cov.gb * cov.gb);
    const float degenerate_threshold = matrix_norm_sq * 1e-6f;
    const float inv_sqrt3 = 0.57735027f;
    Float3 axis = {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    Float3 next = cov.multiply(axis);
    float lsq = length_sq(next);

    // If variation is orthogonal to (1,1,1), test candidate orthogonal axes
    if (lsq <= degenerate_threshold)
    {
        const float inv_sqrt2 = 0.70710678f;
        axis = {inv_sqrt2, -inv_sqrt2, 0.0f};
        next = cov.multiply(axis);
        lsq = length_sq(next);
        if (lsq <= degenerate_threshold)
        {
            axis = {0.0f, inv_sqrt2, -inv_sqrt2};
            next = cov.multiply(axis);
            lsq = length_sq(next);
        }
    }

    return lsq > degenerate_threshold ? next * (1.0f / sqrtf(lsq)) : Float3{1.0f, 0.0f, 0.0f};
}

#if defined(__CUDACC__)
__constant__ float g_srgb_to_linear[256];
__constant__ float g_threshold31[31];
__constant__ float g_threshold63[63];

__device__ __forceinline__ uint32_t clamp_index(uint32_t value, uint32_t limit)
{
    return value < limit ? value : limit - 1;
}

template <bool Srgb> __device__ __forceinline__ float decode_channel(uint32_t value)
{
    if constexpr (Srgb)
        return g_srgb_to_linear[value];
    return float(value) * (1.0f / 255.0f);
}

__device__ __forceinline__ unsigned half_warp_mask()
{
    return 0xffffu << (threadIdx.x & 16);
}

__device__ __forceinline__ float sum4(float value, unsigned mask)
{
    value += __shfl_xor_sync(mask, value, 1, 4);
    return value + __shfl_xor_sync(mask, value, 2, 4);
}

__device__ __forceinline__ float sum16(float value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset, 16);
    return __shfl_sync(mask, value, 0, 16);
}

__device__ __forceinline__ float min16(float value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value = fminf(value, __shfl_down_sync(mask, value, offset, 16));
    return __shfl_sync(mask, value, 0, 16);
}

__device__ __forceinline__ float max16(float value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(mask, value, offset, 16));
    return __shfl_sync(mask, value, 0, 16);
}

__device__ __forceinline__ uint32_t or16(uint32_t value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value |= __shfl_down_sync(mask, value, offset, 16);
    return __shfl_sync(mask, value, 0, 16);
}

__device__ __forceinline__ SymMat3 half_warp_moments(Float3 sample, Float3 &mean)
{
    const unsigned mask = half_warp_mask();
    mean = {sum16(sample.r, mask) / 16, sum16(sample.g, mask) / 16, sum16(sample.b, mask) / 16};
    const Float3 delta = sample - mean;
    return {sum16(delta.r * delta.r, mask) / 16, sum16(delta.g * delta.g, mask) / 16,
            sum16(delta.b * delta.b, mask) / 16, sum16(delta.r * delta.g, mask) / 16,
            sum16(delta.r * delta.b, mask) / 16, sum16(delta.g * delta.b, mask) / 16};
}
#endif
