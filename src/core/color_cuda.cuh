#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Store the sRGB lookup table on the GPU.
__constant__ float g_srgb_to_linear[256];
__constant__ float g_threshold31[31];
__constant__ float g_threshold63[63];

__device__ __forceinline__ float decode_channel_cuda(uint32_t value, bool is_srgb)
{
    if (is_srgb)
        return g_srgb_to_linear[value];

    return static_cast<float>(value) * (1.0f / 255.0f);
}