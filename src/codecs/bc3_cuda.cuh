#pragma once

#include "bc3.h"
#include "bc2_cuda.cuh"
#include "bc4_cuda.cuh"

namespace bc3
{
    __device__ __forceinline__ Float4 device_quadrant_mean(const Block &block, uint32_t q, bool srgb,
                                                         uint32_t width, uint32_t height)
    {
        const Float3 rgb = bc2::device_rgb_quadrant_mean(block.rgb, q, srgb, width, height);
        uint32_t counts[8]{};
        for (uint32_t dy = 0; dy < 2; ++dy)
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                const uint32_t x = (q & 1u) * 2 + dx;
                const uint32_t y = (q >> 1u) * 2 + dy;
                ++counts[bc4::device_get_selector(block.alpha, x < width ? x : width - 1, y < height ? y : height - 1)];
            }
        float alpha = 0.0f;
        for (uint32_t s = 0; s < 8; ++s)
            alpha += alpha_palette_value(block.alpha.e0, block.alpha.e1, s) * (counts[s] * 0.25f);
        return {rgb.r, rgb.g, rgb.b, alpha};
    }

    __device__ __forceinline__ void encode_half_warp(const Float4 &sample, bool srgb, Block *output)
    {
        bc1::encode_half_warp({sample.r, sample.g, sample.b}, srgb, &output->rgb);
        float low, high;
        bc4::fit_endpoints_half_warp(sample.a, low, high);
        const uint8_t e0 = bc4::device_quantize_unorm(high);
        const uint8_t e1 = bc4::device_quantize_unorm(low);
        float palette[8];
        for (uint32_t s = 0; s < 8; ++s)
            palette[s] = alpha_palette_value(e0, e1, s);
        const uint32_t selector = bc4::device_find_best_selector(sample.a, palette);
        const uint32_t lane = threadIdx.x & 15u;
        const uint32_t texel = (lane & 1u) | ((lane & 4u) >> 1u) | ((lane & 2u) << 1u) | (lane & 8u);
        const uint64_t packed = static_cast<uint64_t>(selector) << (texel * 3);
        const uint32_t low_bits = or16(static_cast<uint32_t>(packed), half_warp_mask());
        const uint32_t high_bits = or16(static_cast<uint32_t>(packed >> 32), half_warp_mask());
        if (lane == 0)
        {
            output->alpha.e0 = e0;
            output->alpha.e1 = e1;
            const uint64_t bits = low_bits | (static_cast<uint64_t>(high_bits) << 32);
            for (uint32_t byte = 0; byte < 6; ++byte)
                output->alpha.selectors[byte] = static_cast<uint8_t>(bits >> (byte * 8));
        }
    }
}
