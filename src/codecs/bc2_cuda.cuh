#pragma once

#include "bc2.h"
#include "bc1_cuda.cuh"

namespace bc2
{
    __device__ __forceinline__ Float3 device_rgb_quadrant_mean(const bc1::Block &block, uint32_t q, bool srgb,
                                                               uint32_t width, uint32_t height)
    {
        uint32_t counts[4]{};
        for (uint32_t dy = 0; dy < 2; ++dy)
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                const uint32_t x = (q & 1u) * 2 + dx;
                const uint32_t y = (q >> 1u) * 2 + dy;
                const uint32_t texel = (y < height ? y : height - 1) * 4 + (x < width ? x : width - 1);
                ++counts[(block.indices >> (texel * 2)) & 3u];
            }
        Float3 mean{};
        for (uint32_t s = 0; s < 4; ++s)
            mean += bc1::device_palette_color(block.c0, block.c1, s, srgb, true) * (counts[s] * 0.25f);
        return mean;
    }

    __device__ __forceinline__ Float4 device_quadrant_mean(const Block &block, uint32_t q, bool srgb, uint32_t width,
                                                           uint32_t height)
    {
        const Float3 rgb = device_rgb_quadrant_mean(block.rgb, q, srgb, width, height);
        return {rgb.r, rgb.g, rgb.b, alpha_quadrant_mean(block.alpha, q, width, height)};
    }

    __device__ __forceinline__ void encode_half_warp(const Float4 &sample, bool srgb, Block *output)
    {
        bc1::encode_half_warp({sample.r, sample.g, sample.b}, srgb, &output->rgb);
        const uint32_t lane = threadIdx.x & 15u;
        const uint32_t texel = (lane & 1u) | ((lane & 4u) >> 1u) | ((lane & 2u) << 1u) | (lane & 8u);
        const uint64_t bits = static_cast<uint64_t>(quantize_alpha(sample.a)) << (4 * texel);
        const uint32_t low = or16(static_cast<uint32_t>(bits), half_warp_mask());
        const uint32_t high = or16(static_cast<uint32_t>(bits >> 32), half_warp_mask());
        if (lane == 0)
            output->alpha = low | (static_cast<uint64_t>(high) << 32);
    }
}
