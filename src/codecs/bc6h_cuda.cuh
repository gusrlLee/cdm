#pragma once

#include "bc6h.h"
#include "core/warp.cuh"

namespace bc6h
{
    __device__ __forceinline__ Float3 device_quadrant_mean(const Block &block, uint32_t quadrant, bool, uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        const SymbolicBC6H symbolic = parse_symbolic_block(block);
        if (!symbolic.valid)
            return {};

        constexpr uint32_t weights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
        constexpr uint32_t weights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

        const uint32_t *weights = symbolic.selector_bits == 4 ? weights4 : weights3;

        // [subset][selector]
        uint32_t counts[2][16] = {};

        const uint32_t x0 = (quadrant & 1u) * 2;
        const uint32_t y0 = (quadrant >> 1u) * 2;

        for (uint32_t dy = 0; dy < 2; ++dy)
        {
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                const uint32_t source = valid_texel_index((y0 + dy) * 4 + x0 + dx, valid_width, valid_height);

                const uint32_t subset = symbolic.subset[source];
                const uint32_t selector = symbolic.selector[source];
                ++counts[subset][selector];
            }
        }

        const uint32_t subset_count = symbolic.selector_bits == 4 ? 1u : 2u;
        const uint32_t selector_count = 1u << symbolic.selector_bits;

        Float3 result{};
        for (uint32_t subset = 0; subset < subset_count; ++subset)
        {
            const int *endpoint0 = symbolic.endpoints[subset * 2];
            const int *endpoint1 = symbolic.endpoints[subset * 2 + 1];
            for (uint32_t selector = 0; selector < selector_count; ++selector)
            {
                const uint32_t count = counts[subset][selector];
                if (count == 0)
                    continue;

                const int weight = static_cast<int>(weights[selector]);
                const int r = (endpoint0[0] * (64 - weight) + endpoint1[0] * weight + 32) >> 6;
                const int g = (endpoint0[1] * (64 - weight) + endpoint1[1] * weight + 32) >> 6;
                const int b = (endpoint0[2] * (64 - weight) + endpoint1[2] * weight + 32) >> 6;
                const Float3 color = {
                    half_to_float(finish_unquantize(r, 0)),
                    half_to_float(finish_unquantize(g, 0)),
                    half_to_float(finish_unquantize(b, 0))};

                result += color * (static_cast<float>(count) * 0.25f);
            }
        }

        return result;
    }

    __device__ __forceinline__ void encode_half_warp(Float3 sample, bool, Block *output)
    {
        const uint32_t lane = threadIdx.x & 15u;
        const unsigned mask = half_warp_mask();
        
        sample = sanitize_hdr(sample);
        Float3 mean;
        
        const SymMat3 covariance = half_warp_moments(sample, mean);
        Float3 axis = {1, 0, 0};
        if (lane == 0)
            axis = compute_principal_axis(covariance);
        
        axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16), __shfl_sync(mask, axis.b, 0, 16)};
        const float projection = dot(sample - mean, axis);
        Float3 p0 = mean + axis * min16(projection, mask);
        Float3 p1 = mean + axis * max16(projection, mask);
        bool degenerate = lane == 0 && hdr_covariance_degenerate(covariance, mean);
        
        degenerate = __shfl_sync(mask, degenerate, 0, 16);
        if (degenerate)
            p0 = p1 = mean;
        
        uint32_t endpoints[2][3] = {};
        if (lane == 0)
        {
            endpoints[0][0] = quantize_endpoint(p0.r);
            endpoints[0][1] = quantize_endpoint(p0.g);
            endpoints[0][2] = quantize_endpoint(p0.b);
            endpoints[1][0] = quantize_endpoint(p1.r);
            endpoints[1][1] = quantize_endpoint(p1.g);
            endpoints[1][2] = quantize_endpoint(p1.b);
        }
        
        for (int e = 0; e < 2; ++e)
            for (int c = 0; c < 3; ++c)
                endpoints[e][c] = __shfl_sync(mask, endpoints[e][c], 0, 16);
        
        const Float3 owned = palette_color(endpoints, lane);
        const float owned_projection = dot(owned - mean, axis);
        float best = 1e30f;
        
        uint32_t selector = 0;
        for (uint32_t s = 0; s < 16; ++s)
        {
            const float delta = projection - __shfl_sync(mask, owned_projection, s, 16);
            const float error = delta * delta;
            if (error < best)
            {
                best = error;
                selector = s;
            }
        }
        
        const bool reverse = __shfl_sync(mask, selector >= 8, 0, 16);
        if (reverse)
        {
            selector = 15 - selector;
            for (int c = 0; c < 3; ++c)
            {
                uint32_t t = endpoints[0][c];
                endpoints[0][c] = endpoints[1][c];
                endpoints[1][c] = t;
            }
        }
        
        const uint32_t texel = texel_index(lane);
        uint64_t selector_bits = texel == 0 ? uint64_t(selector) << 1 : uint64_t(selector) << (4 * texel);
        selector_bits |= uint64_t(or16(uint32_t(selector_bits), mask));
        
        const uint32_t high_half = or16(uint32_t(selector_bits >> 32), mask);
        if (lane == 0)
        {
            const uint64_t low = 3u | (uint64_t(endpoints[0][0]) << 5) | (uint64_t(endpoints[0][1]) << 15) |
                                 (uint64_t(endpoints[0][2]) << 25) | (uint64_t(endpoints[1][0]) << 35) |
                                 (uint64_t(endpoints[1][1]) << 45) | (uint64_t(endpoints[1][2] & 0x1ffu) << 55);
            
            const uint64_t high = uint64_t(endpoints[1][2] >> 9) | uint32_t(selector_bits) | (uint64_t(high_half) << 32);
            *output = {low, high};
        }
    }

}
