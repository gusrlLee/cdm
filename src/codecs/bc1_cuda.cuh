#pragma once

#include "bc1.h"
#include "core/color_cuda.cuh"
#include "core/warp.cuh"

namespace bc1
{
    // Decode one palette color to linear RGB.
    __device__ __forceinline__ Float3 device_palette_color(uint16_t c0, uint16_t c1, uint32_t selector, bool is_srgb,
                                                           bool opaque = false)
    {
        const Rgb8 color = palette_color_rgb8(c0, c1, selector, opaque);
        return {decode_channel_cuda(color.r, is_srgb), decode_channel_cuda(color.g, is_srgb),
                decode_channel_cuda(color.b, is_srgb)};
    }

    __device__ __forceinline__ Float3 device_quadrant_mean(const Block &block, uint32_t quadrant, bool is_srgb,
                                                           uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t group_start = (threadIdx.x & 15u) & ~3u;

        const Float3 owned = device_palette_color(block.c0, block.c1, quadrant, is_srgb);
        uint32_t indices = block.indices;

        if (valid_width < 4 || valid_height < 4)
        {
            indices = 0;
            for (uint32_t texel = 0; texel < 16; ++texel)
            {
                const uint32_t x = texel & 3u;
                const uint32_t y = texel >> 2u;
                const uint32_t sx = x < valid_width ? x : valid_width - 1;
                const uint32_t sy = y < valid_height ? y : valid_height - 1;

                const uint32_t source = sy * 4 + sx;
                const uint32_t selector = (block.indices >> (source * 2)) & 3u;

                indices |= selector << (texel * 2);
            }
        }

        const uint32_t counts = selector_region_counts(indices, quadrant);
        const uint32_t shift = (quadrant & 1u) * 4u + (quadrant >> 1u) * 16u;

        Float3 mean{};

        // Share each palette color and its packed counts
        for (uint32_t selector = 0; selector < 4; ++selector)
        {
            const uint32_t lane = group_start + selector;
            const Float3 color = {__shfl_sync(mask, owned.r, lane, 16), __shfl_sync(mask, owned.g, lane, 16),
                                  __shfl_sync(mask, owned.b, lane, 16)};
            const uint32_t packed = __shfl_sync(mask, counts, lane, 16);
            const uint32_t count = (packed >> shift) & 15u;

            mean += color * (static_cast<float>(count) * 0.25f);
        }

        return mean;
    }

    // Reconstruct four texel colors per lane, then average the quadrant.
    __device__ __forceinline__ Float3 device_quadrant_mean_texels(const Block &block, uint32_t quadrant, bool is_srgb, uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t group_start = (threadIdx.x & 15u) & ~3u;

        // Four lanes share the four linear palette colors.
        const Float3 owned = device_palette_color(block.c0, block.c1, quadrant, is_srgb);

        const uint32_t x0 = (quadrant & 1u) * 2u;
        const uint32_t y0 = (quadrant >> 1u) * 2u;

        Float3 sum{};
        for (uint32_t dy = 0; dy < 2; ++dy)
        {
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                const uint32_t x = x0 + dx;
                const uint32_t y = y0 + dy;
                const uint32_t sx = x < valid_width ? x : valid_width - 1;
                const uint32_t sy = y < valid_height ? y : valid_height - 1;

                const uint32_t texel = sy * 4 + sx;
                const uint32_t selector = (block.indices >> (texel * 2u)) & 3u;

                const uint32_t source_lane = group_start + selector;
                const Float3 color = {
                    __shfl_sync(mask, owned.r, source_lane, 16),
                    __shfl_sync(mask, owned.g, source_lane, 16),
                    __shfl_sync(mask, owned.b, source_lane, 16)
                };

                sum += color;
            }
        }

        return sum * 0.25f;
    }

    // Match the CPU upper-bound search for sRGB quantization.
    template <uint32_t MaxValue> __device__ __forceinline__ uint32_t device_quantize_channel(float value)
    {
        const float *thresholds = MaxValue == 31 ? g_threshold31 : g_threshold63;

        uint32_t begin = 0;
        uint32_t end = MaxValue;

        while (begin < end)
        {
            const uint32_t middle = begin + (end - begin) / 2;

            if (value < thresholds[middle])
                end = middle;
            else
                begin = middle + 1;
        }

        return begin;
    }

    // Quantize linear RGB and pack an RGB565 endpoint.
    __device__ __forceinline__ uint16_t device_encode_rgb565(const Float3 &color, bool is_srgb)
    {
        const Float3 clamped = clamp01(color);
        uint32_t r, g, b;

        if (is_srgb)
        {
            r = device_quantize_channel<31>(clamped.r);
            g = device_quantize_channel<63>(clamped.g);
            b = device_quantize_channel<31>(clamped.b);
        }
        else
        {
            r = static_cast<uint32_t>(clamped.r * 31.0f + 0.5f);
            g = static_cast<uint32_t>(clamped.g * 63.0f + 0.5f);
            b = static_cast<uint32_t>(clamped.b * 31.0f + 0.5f);
        }

        return static_cast<uint16_t>((r << 11) | (g << 5) | b);
    }

    __device__ __forceinline__ void encode_half_warp(const Float3 &sample, bool is_srgb, Block *output)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t lane = threadIdx.x & 15u;

        Float3 mean;
        const SymMat3 covariance = half_warp_moments(sample, mean);
        const bool flat = (covariance.rr + covariance.gg + covariance.bb) < 1e-10f;

        Float3 axis{};
        if (lane == 0)
            axis = compute_principal_axis(covariance);

        axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16), __shfl_sync(mask, axis.b, 0, 16)};

        const float projection = dot(sample - mean, axis);
        const float minimum = min16(projection, mask);
        const float maximum = max16(projection, mask);

        // Quantize the endpoints once per block.
        uint32_t c0 = 0;
        uint32_t c1 = 0;

        if (lane == 0)
        {
            const Float3 p0 = flat ? mean : mean + axis * minimum;
            const Float3 p1 = flat ? mean : mean + axis * maximum;

            c0 = device_encode_rgb565(p0, is_srgb);
            c1 = device_encode_rgb565(p1, is_srgb);

            if (c0 < c1)
            {
                const uint32_t temp = c0;
                c0 = c1;
                c1 = temp;
            }
        }

        c0 = __shfl_sync(mask, c0, 0, 16);
        c1 = __shfl_sync(mask, c1, 0, 16);

        float best_distance = 1e30f;
        uint32_t best_selector = 0;

        // Strict comparison keeps the lower selector on ties.
        for (uint32_t selector = 0; selector < 4; ++selector)
        {
            const Float3 color =
                device_palette_color(static_cast<uint16_t>(c0), static_cast<uint16_t>(c1), selector, is_srgb, true);

            const float palette_projection = dot(color - mean, axis);
            const float delta = projection - palette_projection;
            const float distance = delta * delta;

            if (distance < best_distance)
            {
                best_distance = distance;
                best_selector = selector;
            }
        }

        // Convert quadrant-grouped lanes to row-major texel positions.
        const uint32_t x = (lane & 1u) | ((lane >> 1u) & 2u);
        const uint32_t y = ((lane >> 1u) & 1u) | ((lane >> 2u) & 2u);
        const uint32_t texel = y * 4 + x;

        const uint32_t indices = or16(best_selector << (texel * 2), mask);
        if (lane == 0)
        {
            output->c0 = static_cast<uint16_t>(c0);
            output->c1 = static_cast<uint16_t>(c1);
            output->indices = indices;
        }
    }

    // Repeat valid samples to fill a partial 4x4 block
    __device__ __forceinline__ Float3 repeat_small_sample(const Float3 &sample, uint32_t valid_width,
                                                          uint32_t valid_height)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t lane = threadIdx.x & 15u;

        const uint32_t x = (lane & 1u) | ((lane >> 1u) & 2u);
        const uint32_t y = ((lane >> 1u) & 1u) | ((lane >> 2u) & 2u);

        const uint32_t source_x = x % valid_width;
        const uint32_t source_y = y % valid_height;

        // Convert texel coordinates back to a quadrant-grouped lane.
        const uint32_t source_lane =
            (source_x & 1u) | ((source_y & 1u) << 1u) | ((source_x & 2u) << 1u) | ((source_y & 2u) << 2u);
        return {__shfl_sync(mask, sample.r, source_lane, 16), __shfl_sync(mask, sample.g, source_lane, 16),
                __shfl_sync(mask, sample.b, source_lane, 16)};
    }

} // namespace bc1
