#pragma once

#include "bc4.h"
#include "core/warp.cuh"
#include "core/mipmap.h"

#include <cuda_runtime.h>

namespace bc4
{
    __device__ __forceinline__ uint32_t device_get_selector(const Block &block, uint32_t x, uint32_t y)
    {
        const uint32_t bit_offset = (y * 4 + x) * 3;
        const uint32_t byte_index = bit_offset / 8;
        const uint32_t bit_in_byte = bit_offset % 8;

        uint32_t bits = block.selectors[byte_index];

        if (bit_in_byte > 5)
        {
            bits |= static_cast<uint32_t>(block.selectors[byte_index + 1]) << 8;
        }

        return (bits >> bit_in_byte) & 7u;
    }

    __device__ __forceinline__ float device_palette_value_unorm(uint8_t e0, uint8_t e1, uint32_t selector)
    {
        if (selector == 0)
        {
            return static_cast<float>(e0) * scale;
        }

        if (selector == 1)
        {
            return static_cast<float>(e1) * scale;
        }

        if (e0 <= e1 && selector >= 6)
        {
            return selector == 6 ? 0.0f : 1.0f;
        }

        const uint32_t divisor = e0 > e1 ? 7u : 5u;
        const uint32_t weight = selector - 1;
        const uint32_t numerator = (divisor - weight) * e0 + weight * e1;

        return static_cast<float>(numerator) * (scale / static_cast<float>(divisor));
    }

    __device__ __forceinline__ float device_quadrant_mean(const Block &block, uint32_t quadrant,
                                                          uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        const uint32_t start_x = (quadrant & 1u) * 2;
        const uint32_t start_y = (quadrant >> 1u) * 2;

        uint32_t counts[8] = {};

        for (uint32_t dy = 0; dy < 2; ++dy)
        {
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                uint32_t x = start_x + dx;
                uint32_t y = start_y + dy;

                if (x >= valid_width)
                    x = valid_width - 1;
                if (y >= valid_height)
                    y = valid_height - 1;

                ++counts[device_get_selector(block, x, y)];
            }
        }

        float mean = 0.0f;

        for (uint32_t selector = 0; selector < 8; ++selector)
        {
            const float weight = static_cast<float>(counts[selector]) * 0.25f;
            mean += device_palette_value_unorm(block.e0, block.e1, selector) * weight;
        }

        return mean;
    }

    __device__ __forceinline__ float repeat_small_sample(float sample, uint32_t valid_width, uint32_t valid_height)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t lane = threadIdx.x & 15u;

        const uint32_t x = (lane & 1u) | ((lane >> 1u) & 2u);
        const uint32_t y = ((lane >> 1u) & 1u) | ((lane >> 2u) & 2u);

        const uint32_t source_x = x % valid_width;
        const uint32_t source_y = y % valid_height;

        const uint32_t source_lane =
            (source_x & 1u) | ((source_y & 1u) << 1u) | ((source_x & 2u) << 1u) | ((source_y & 2u) << 2u);

        return __shfl_sync(mask, sample, source_lane, 16);
    }

    __device__ __forceinline__ void fit_endpoints_half_warp(float sample, float &low, float &high)
    {
        const unsigned mask = half_warp_mask();
        const float weight = 1.0f / 16.0f;

        const float mean = sum16(sample, mask) * weight;

        low = min16(sample, mask);
        high = max16(sample, mask);

        const float delta = sample - mean;
        const float variance = sum16(delta * delta * weight, mask);

        if (variance < 1e-10f)
        {
            low = mean;
            high = mean;
        }
    }

    __device__ __forceinline__ uint8_t device_quantize_unorm(float value)
    {
        const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
        const float scaled = __fmul_rn(clamped, 255.0f);
        const float rounded = __fadd_rn(scaled, 0.5f);

        return static_cast<uint8_t>(__float2uint_rz(rounded));
    }

    __device__ __forceinline__ uint32_t device_find_best_selector(float sample, const float palette[8])
    {
        uint32_t best_selector = 0;
        const float first_delta = sample - palette[0];
        float best_error = first_delta * first_delta;

        for (uint32_t selector = 1; selector < 8; ++selector)
        {
            const float delta = sample - palette[selector];
            const float err = delta * delta;

            if (err < best_error)
            {
                best_error = err;
                best_selector = selector;
            }
        }

        return best_selector;
    }

    __device__ __forceinline__ void encode_half_warp(float sample, Block *output)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t lane = threadIdx.x & 15u;

        float low;
        float high;
        fit_endpoints_half_warp(sample, low, high);

        const uint8_t e0 = device_quantize_unorm(high);
        const uint8_t e1 = device_quantize_unorm(low);

        float palette[8];
        for (uint32_t selector = 0; selector < 8; ++selector)
        {
            palette[selector] = device_palette_value_unorm(e0, e1, selector);
        }

        const uint32_t selector = device_find_best_selector(sample, palette);

        const uint32_t x = (lane & 1u) | ((lane >> 1u) & 2u);
        const uint32_t y = ((lane >> 1u) & 1u) | ((lane >> 2u) & 2u);
        const uint32_t bit_offset = (y * 4 + x) * 3;

        const uint64_t packed = static_cast<uint64_t>(selector) << bit_offset;

        const uint32_t bits_low = or16(static_cast<uint32_t>(packed), mask);
        const uint32_t bits_high = or16(static_cast<uint32_t>(packed >> 32), mask);

        if (lane == 0)
        {
            output->e0 = e0;
            output->e1 = e1;

            const uint64_t bits = static_cast<uint64_t>(bits_low) | (static_cast<uint64_t>(bits_high) << 32);

            for (uint32_t byte = 0; byte < 6; ++byte)
            {
                output->selectors[byte] = static_cast<uint8_t>(bits >> (byte * 8));
            }
        }
    }

    void generate_base_mip_cuda(const Block *source, MipLevel source_level, Block *destination,
                                MipLevel destination_level, MeanImage means, uint32_t groups, uint32_t threads);
    void generate_mean_mip_cuda(MeanImage source, Block *destination, MipLevel destination_level, MeanImage next_means,
                                uint32_t groups, uint32_t threads);

} // namespace bc4