#pragma once

#include "bc5.h"
#include "bc4_cuda.cuh"

namespace bc5
{
    __device__ __forceinline__ float device_signed_quadrant_mean(const bc4::Block &block, uint32_t q, uint32_t width,
                                                                 uint32_t height)
    {
        uint32_t counts[8]{};
        for (uint32_t dy = 0; dy < 2; ++dy)
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                const uint32_t x = (q & 1u) * 2 + dx;
                const uint32_t y = (q >> 1u) * 2 + dy;
                ++counts[bc4::device_get_selector(block, x < width ? x : width - 1, y < height ? y : height - 1)];
            }
        float mean = 0.0f;
        for (uint32_t s = 0; s < 8; ++s)
            mean += signed_palette_value(block.e0, block.e1, s) * (counts[s] * 0.25f);
        return mean;
    }

    __device__ __forceinline__ void encode_signed_half_warp(float sample, bc4::Block *output)
    {
        float low, high;
        bc4::fit_endpoints_half_warp(sample, low, high);
        const uint8_t e0 = quantize_signed(high), e1 = quantize_signed(low);
        float palette[8];
        for (uint32_t s = 0; s < 8; ++s)
            palette[s] = signed_palette_value(e0, e1, s);
        const uint32_t selected = bc4::device_find_best_selector(sample, palette);
        const uint32_t lane = threadIdx.x & 15u;
        const uint32_t texel = (lane & 1u) | ((lane & 4u) >> 1u) | ((lane & 2u) << 1u) | (lane & 8u);
        const uint64_t packed = static_cast<uint64_t>(selected) << (texel * 3);
        const uint32_t low_bits = or16(static_cast<uint32_t>(packed), half_warp_mask());
        const uint32_t high_bits = or16(static_cast<uint32_t>(packed >> 32), half_warp_mask());
        if (lane == 0)
        {
            output->e0 = e0;
            output->e1 = e1;
            const uint64_t bits = low_bits | (static_cast<uint64_t>(high_bits) << 32);
            for (uint32_t byte = 0; byte < 6; ++byte)
                output->selectors[byte] = static_cast<uint8_t>(bits >> (byte * 8));
        }
    }

    __device__ __forceinline__ Float3 device_quadrant_mean(const Block &block, uint32_t quadrant, bool is_signed,
                                                           uint32_t width, uint32_t height)
    {
        if (is_signed)
            return {device_signed_quadrant_mean(block.red, quadrant, width, height),
                    device_signed_quadrant_mean(block.green, quadrant, width, height), 0.0f};
        return {bc4::device_quadrant_mean(block.red, quadrant, width, height),
                bc4::device_quadrant_mean(block.green, quadrant, width, height), 0.0f};
    }

    __device__ __forceinline__ void encode_half_warp(const Float3 &sample, bool is_signed, Block *output)
    {
        if (is_signed)
        {
            encode_signed_half_warp(sample.r, &output->red);
            encode_signed_half_warp(sample.g, &output->green);
        }
        else
        {
            bc4::encode_half_warp(sample.r, &output->red);
            bc4::encode_half_warp(sample.g, &output->green);
        }
    }
}
