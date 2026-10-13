#pragma once

#include "math.h"

#include <cstdint>
#include <cuda_runtime.h>

// Select the lower or upper group of 16 threads in a warp.
__device__ __forceinline__ unsigned half_warp_mask()
{
    return 0xffffu << (threadIdx.x & 16u);
}

// Sum each group of four threads and share the result.
__device__ __forceinline__ float sum4(float value, unsigned mask)
{
    value += __shfl_xor_sync(mask, value, 1, 4);
    return value + __shfl_xor_sync(mask, value, 2, 4);
}

// Sum 16 threads and broadcast the result from lane 0.
__device__ __forceinline__ float sum16(float value, unsigned mask)
{
    for (int offset = 8; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(mask, value, offset, 16);
    }

    return __shfl_sync(mask, value, 0, 16);
}

// Find the minimum across 16 threads and share the result.
__device__ __forceinline__ float min16(float value, unsigned mask)
{
    for (int offset = 8; offset > 0; offset >>= 1)
    {
        const float other = __shfl_down_sync(mask, value, offset, 16);
        value = fminf(value, other);
    }

    return __shfl_sync(mask, value, 0, 16);
}

// Find the maximum across 16 threads and share the result.
__device__ __forceinline__ float max16(float value, unsigned mask)
{
    for (int offset = 8; offset > 0; offset >>= 1)
    {
        const float other = __shfl_down_sync(mask, value, offset, 16);
        value = fmaxf(value, other);
    }

    return __shfl_sync(mask, value, 0, 16);
}

// Combine the packed selectors from all 16 threads.
__device__ __forceinline__ uint32_t or16(uint32_t value, unsigned mask)
{
    for (int offset = 8; offset > 0; offset >>= 1)
    {
        value |= __shfl_down_sync(mask, value, offset, 16);
    }

    return __shfl_sync(mask, value, 0, 16);
}

__device__ __forceinline__ SymMat3 half_warp_moments(const Float3 &sample, Float3 &mean)
{
    const unsigned mask = half_warp_mask();
    const float weight = 1.0f / 16.0f;

    mean = {sum16(sample.r, mask) * weight, sum16(sample.g, mask) * weight, sum16(sample.b, mask) * weight};

    const Float3 delta = sample - mean;
    SymMat3 covariance{};
    covariance.rr = sum16(delta.r * delta.r, mask) * weight;
    covariance.rg = sum16(delta.r * delta.g, mask) * weight;
    covariance.rb = sum16(delta.r * delta.b, mask) * weight;
    covariance.gg = sum16(delta.g * delta.g, mask) * weight;
    covariance.gb = sum16(delta.g * delta.b, mask) * weight;
    covariance.bb = sum16(delta.b * delta.b, mask) * weight;

    return covariance;
}