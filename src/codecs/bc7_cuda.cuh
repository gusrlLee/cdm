#pragma once

#include "bc7.h"
#include "core/warp.cuh"
#include "core/mipmap.h"
#include <cuda_runtime.h>

namespace bc7
{
    // Called once per CUDA context before launching BC7 work.
    inline cudaError_t upload_tables_cuda()
    {
        const auto lut = make_mode6_quantization_lut();
        cudaError_t result;
        result = cudaMemcpyToSymbol(g_srgb_to_linear, c_srgb_to_linear, sizeof(c_srgb_to_linear));
        if (result != cudaSuccess)
            return result;
        result = cudaMemcpyToSymbol(device_partition2, g_bc7_partition2, sizeof(g_bc7_partition2));
        if (result != cudaSuccess)
            return result;
        result = cudaMemcpyToSymbol(device_partition3, g_bc7_partition3, sizeof(g_bc7_partition3));
        if (result != cudaSuccess)
            return result;
        result = cudaMemcpyToSymbol(device_anchor2, g_bc7_table_anchor_index_second_subset,
                                    sizeof(g_bc7_table_anchor_index_second_subset));
        if (result != cudaSuccess)
            return result;
        result = cudaMemcpyToSymbol(device_anchor3a, g_bc7_table_anchor_index_third_subset_1,
                                    sizeof(g_bc7_table_anchor_index_third_subset_1));
        if (result != cudaSuccess)
            return result;
        result = cudaMemcpyToSymbol(device_anchor3b, g_bc7_table_anchor_index_third_subset_2,
                                    sizeof(g_bc7_table_anchor_index_third_subset_2));
        if (result != cudaSuccess)
            return result;
        result = cudaMemcpyToSymbol(device_partition_masks, kPartitionMasks2.data(), sizeof(uint16_t) * 64);
        if (result != cudaSuccess)
            return result;
        result = cudaMemcpyToSymbol(device_mode6_lut, lut.data(), sizeof(uint8_t) * 4096);
        if (result != cudaSuccess)
            return result;
        return cudaSuccess;
    }

    __device__ __forceinline__ Float4 broadcast_color(Float4 value, unsigned mask)
    {
        return {__shfl_sync(mask, value.r, 0, 16), __shfl_sync(mask, value.g, 0, 16), __shfl_sync(mask, value.b, 0, 16),
                __shfl_sync(mask, value.a, 0, 16)};
    }

    // One lane owns one row-major texel. Both subsets use the whole half-warp for reductions.
    __device__ __forceinline__ PartitionSubsetStats device_subset_stats(Float4 sample, bool included, bool has_alpha,
                                                                        unsigned mask, float &projection)
    {
        PartitionSubsetStats stats{};
        const float weight = included ? 1.0f : 0.0f;
        stats.count = static_cast<uint32_t>(sum16(weight, mask));
        const float inverse_count = 1.0f / static_cast<float>(stats.count);
        stats.mean = {sum16(sample.r * weight, mask) * inverse_count, sum16(sample.g * weight, mask) * inverse_count,
                      sum16(sample.b * weight, mask) * inverse_count,
                      has_alpha ? sum16(sample.a * weight, mask) * inverse_count : 1.0f};
        const Float4 d = (sample - stats.mean) * weight;
        SymMat3 rgb{};
        rgb.rr = sum16(d.r * d.r, mask) * inverse_count;
        rgb.rg = sum16(d.r * d.g, mask) * inverse_count;
        rgb.rb = sum16(d.r * d.b, mask) * inverse_count;
        rgb.gg = sum16(d.g * d.g, mask) * inverse_count;
        rgb.gb = sum16(d.g * d.b, mask) * inverse_count;
        rgb.bb = sum16(d.b * d.b, mask) * inverse_count;
        stats.cov = {rgb.rr, rgb.rg, rgb.rb, 0, rgb.gg, rgb.gb, 0, rgb.bb, 0, 0};
        if (has_alpha)
        {
            stats.cov.ra = sum16(d.r * d.a, mask) * inverse_count;
            stats.cov.ga = sum16(d.g * d.a, mask) * inverse_count;
            stats.cov.ba = sum16(d.b * d.a, mask) * inverse_count;
            stats.cov.aa = sum16(d.a * d.a, mask) * inverse_count;
        }
        Float4 axis{};
        if ((threadIdx.x & 15u) == 0)
        {
            if (has_alpha)
            {
                axis = compute_principal_axis(stats.cov);
            }
            else
            {
                const Float3 direction = compute_principal_axis(rgb);
                axis = {direction.r, direction.g, direction.b, 0.0f};
            }
        }
        stats.axis = broadcast_color(axis, mask);
        projection = dot(sample - stats.mean, stats.axis);
        stats.lo = min16(included ? projection : 1e30f, mask);
        stats.hi = max16(included ? projection : -1e30f, mask);
        return stats;
    }

    // Four-texel histogram avoids repeating reconstruction within one quadrant.
    __device__ __forceinline__ Float4 device_quadrant_mean(const Block &block, uint32_t quadrant, uint32_t width,
                                                           uint32_t height)
    {
        const SymbolicBlock symbolic = parse_symbolic_block(block);
        if (symbolic.mode == 8)
            return {};
        uint16_t keys[4]{};
        uint8_t counts[4]{};
        uint32_t entries = 0;
        for (uint32_t y = 0; y < 2; ++y)
        {
            for (uint32_t x = 0; x < 2; ++x)
            {
                const uint32_t sx = min((quadrant & 1u) * 2 + x, width - 1);
                const uint32_t sy = min((quadrant >> 1u) * 2 + y, height - 1);
                const uint32_t t = sy * 4 + sx;
                const uint16_t key =
                    symbolic.subset[t] | (symbolic.color_index[t] << 2) | (symbolic.alpha_index[t] << 6);
                uint32_t entry = 0;
                while (entry < entries && keys[entry] != key)
                    ++entry;
                if (entry == entries)
                    keys[entries++] = key;
                ++counts[entry];
            }
        }
        Float4 result{};
        for (uint32_t entry = 0; entry < entries; ++entry)
        {
            const uint16_t key = keys[entry];
            uint8_t rgba[4];
            reconstruct_color(symbolic, key & 3u, (key >> 2) & 15u, (key >> 6) & 15u, rgba);
            result = result + rgba8_to_linear(rgba) * (counts[entry] * 0.25f);
        }
        return result;
    }

    __device__ __forceinline__ Float4 repeat_small_sample(Float4 sample, uint32_t width, uint32_t height)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t lane = threadIdx.x & 15u;
        const uint32_t source = ((lane / 4) % height) * 4 + (lane % 4) % width;
        return {__shfl_sync(mask, sample.r, source, 16), __shfl_sync(mask, sample.g, source, 16),
                __shfl_sync(mask, sample.b, source, 16), __shfl_sync(mask, sample.a, source, 16)};
    }

    __device__ __forceinline__ void encode_half_warp(Float4 sample, Block *output)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t lane = threadIdx.x & 15u;
        const bool has_alpha = __any_sync(mask, sample.a < 1.0f) != 0;
        float projection;
        const PartitionSubsetStats full = device_subset_stats(sample, true, has_alpha, mask, projection);
        const bool flat = full.cov.rr + full.cov.gg + full.cov.bb + full.cov.aa < 1e-10f;
        const Float4 low = flat ? full.mean : clamp01(full.mean + full.axis * full.lo);
        const Float4 high = flat ? full.mean : clamp01(full.mean + full.axis * full.hi);
        uint32_t mode = 6;
        uint32_t partition = 0;
        PartitionSubsetStats stats[2];
        float subset_projection[2];
        if (!flat)
        {
            const uint16_t desired = static_cast<uint16_t>(or16(uint32_t(projection > 0.0f) << lane, mask));
            if (lane == 0)
                partition = choose_partition(desired);
            partition = __shfl_sync(mask, partition, 0, 16);
            const uint32_t subset = partition_subset(partition, lane);
            for (uint32_t s = 0; s < 2; ++s)
            {
                stats[s] = device_subset_stats(sample, subset == s, has_alpha, mask, subset_projection[s]);
            }
            if (lane == 0)
            {
                mode = choose_partitioned_mode(stats, has_alpha, full.cov, full.axis, low, high);
            }
            mode = __shfl_sync(mask, mode, 0, 16);
        }
        const uint32_t subsets = mode == 6 ? 1 : 2;
        const uint32_t bits = mode == 6 ? 4 : mode == 1 ? 3 : 2;
        const uint32_t size = 1u << bits;
        uint8_t quantized[4][4]{};
        uint8_t pbits[4]{};
        uint8_t endpoints[4][4]{};
        if (lane == 0)
        {
            if (mode == 6)
            {
                quantize_mode6_endpoint(low, quantized[0], pbits[0]);
                quantize_mode6_endpoint(high, quantized[1], pbits[1]);
            }
            else
            {
                Float4 targets[4];
                for (uint32_t s = 0; s < 2; ++s)
                {
                    targets[s * 2] = clamp01(stats[s].mean + stats[s].axis * stats[s].lo);
                    targets[s * 2 + 1] = clamp01(stats[s].mean + stats[s].axis * stats[s].hi);
                }
                quantize_partition_endpoints(targets, mode, quantized, pbits);
            }
            for (uint32_t e = 0; e < subsets * 2; ++e)
            {
                for (uint32_t c = 0; c < 4; ++c)
                {
                    endpoints[e][c] = mode == 6             ? (quantized[e][c] << 1) | pbits[e]
                                      : mode == 1 && c == 3 ? 255
                                                            : expand_partition_channel(quantized[e][c], pbits[e], mode);
                }
            }
        }
        for (uint32_t e = 0; e < subsets * 2; ++e)
        {
            for (uint32_t c = 0; c < 4; ++c)
            {
                endpoints[e][c] = static_cast<uint8_t>(__shfl_sync(mask, uint32_t(endpoints[e][c]), 0, 16));
            }
        }
        // Each lane builds one palette entry, then shares its projection through shuffles.
        float palette_projections[2];
        for (uint32_t s = 0; s < subsets; ++s)
        {
            uint8_t rgba[4];
            for (uint32_t c = 0; c < 4; ++c)
            {
                rgba[c] = interpolate_channel(endpoints[s * 2][c], endpoints[s * 2 + 1][c], lane % size, bits);
            }
            const Float4 mean = mode == 6 ? full.mean : stats[s].mean;
            const Float4 axis = mode == 6 ? full.axis : stats[s].axis;
            palette_projections[s] = dot(rgba8_to_linear(rgba) - mean, axis);
        }
        const uint32_t subset = mode == 6 ? 0 : partition_subset(partition, lane);
        const float target = mode == 6 ? projection : subset_projection[subset];
        float best_error = 1e30f;
        uint32_t selector = 0;
        for (uint32_t index = 0; index < size; ++index)
        {
            const float p0 = __shfl_sync(mask, palette_projections[0], index, 16);
            const float p1 = mode == 6 ? p0 : __shfl_sync(mask, palette_projections[1], index, 16);
            const float delta = target - (subset == 0 ? p0 : p1);
            const float error = delta * delta;
            if (error < best_error)
            {
                best_error = error;
                selector = index;
            }
        }
        uint8_t selectors[16];
        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t selected = __shfl_sync(mask, selector, i, 16);
            if (lane == 0)
                selectors[i] = static_cast<uint8_t>(selected);
        }
        if (lane == 0)
        {
            fix_anchors(mode, partition, quantized, pbits, selectors);
            *output = pack_block(mode, partition, quantized, pbits, selectors);
        }
    }
}

namespace bc7
{
    // Reduce each spatial 2x2 group in row-major lane order.
    __device__ __forceinline__ Float4 quadrant_mean_half_warp(Float4 sample)
    {
        const unsigned mask = half_warp_mask();
        const uint32_t lane = threadIdx.x & 15u;
        const uint32_t first = (lane / 8) * 8 + ((lane % 4) / 2) * 2;
        Float4 values[4];
        const uint32_t offsets[4] = {0, 1, 4, 5};
        for (uint32_t i = 0; i < 4; ++i)
        {
            const uint32_t source = first + offsets[i];
            values[i] = {__shfl_sync(mask, sample.r, source, 16), __shfl_sync(mask, sample.g, source, 16),
                         __shfl_sync(mask, sample.b, source, 16), __shfl_sync(mask, sample.a, source, 16)};
        }
        return (values[0] + values[1] + values[2] + values[3]) * 0.25f;
    }

    static __global__ void generate_base_mip_cuda(const Block *source, MipLevel previous, Block *destination,
                                                  MipLevel current, MeanImage4 means)
    {
        const uint32_t output = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
        if (output >= current.block_count_x * current.block_count_y)
            return;
        const uint32_t lane = threadIdx.x & 15u;
        const uint32_t x = lane % 4;
        const uint32_t y = lane / 4;
        const uint32_t bx = output % current.block_count_x;
        const uint32_t by = output / current.block_count_x;
        const uint32_t sx = min(bx * 2 + x / 2, previous.block_count_x - 1);
        const uint32_t sy = min(by * 2 + y / 2, previous.block_count_y - 1);
        const uint32_t width = min(4u, previous.width - sx * 4);
        const uint32_t height = min(4u, previous.height - sy * 4);
        Float4 sample = device_quadrant_mean(source[static_cast<size_t>(sy) * previous.block_count_x + sx],
                                             (y % 2) * 2 + x % 2, width, height);
        sample = repeat_small_sample(sample, min(4u, current.width - bx * 4), min(4u, current.height - by * 4));
        const Float4 mean = quadrant_mean_half_warp(sample);
        if ((x & 1u) == 0 && (y & 1u) == 0)
        {
            means.set(bx * 2 + x / 2, by * 2 + y / 2, mean);
        }
        encode_half_warp(sample, destination + output);
    }

    static __global__ void generate_mean_mip_cuda(MeanImage4 source, Block *destination, MipLevel current,
                                                  MeanImage4 next)
    {
        const uint32_t output = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
        if (output >= current.block_count_x * current.block_count_y)
            return;
        const uint32_t lane = threadIdx.x & 15u;
        const uint32_t x = lane % 4;
        const uint32_t y = lane / 4;
        const uint32_t bx = output % current.block_count_x;
        const uint32_t by = output / current.block_count_x;
        Float4 sample = source.get(bx * 4 + x, by * 4 + y);
        const Float4 mean = quadrant_mean_half_warp(sample);
        if (next.r != nullptr && (x & 1u) == 0 && (y & 1u) == 0)
        {
            next.set(bx * 2 + x / 2, by * 2 + y / 2, mean);
        }
        sample = repeat_small_sample(sample, min(4u, current.width - bx * 4), min(4u, current.height - by * 4));
        encode_half_warp(sample, destination + output);
    }
}
