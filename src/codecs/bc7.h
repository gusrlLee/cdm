#pragma once

#include "core/color.h"
#include "core/math.h"
#include "core/mean.h"

#include "bc7_tables.h"
#if defined(__CUDACC__)
#include "core/color_cuda.cuh"
#endif

#include <cassert>
#include <cstdint>
#include <array>
#include <utility>
#include <algorithm>

namespace bc7
{
#if defined(__CUDACC__)
    __constant__ uint8_t device_partition2[1024];
    __constant__ uint8_t device_partition3[1024];
    __constant__ uint8_t device_anchor2[64];
    __constant__ uint8_t device_anchor3a[64];
    __constant__ uint8_t device_anchor3b[64];
    __constant__ uint16_t device_partition_masks[64];
    __constant__ uint8_t device_mode6_lut[2][2048];
#endif

    CDM_INLINE uint8_t partition_subset(uint32_t partition, uint32_t texel, bool three = false)
    {
#if defined(__CUDA_ARCH__)
        return three ? device_partition3[partition * 16 + texel] : device_partition2[partition * 16 + texel];
#else
        return three ? g_bc7_partition3[partition * 16 + texel] : g_bc7_partition2[partition * 16 + texel];
#endif
    }

    CDM_INLINE uint8_t partition_anchor(uint32_t partition, uint32_t kind = 0)
    {
#if defined(__CUDA_ARCH__)
        return kind == 0   ? device_anchor2[partition]
               : kind == 1 ? device_anchor3a[partition]
                           : device_anchor3b[partition];
#else
        return kind == 0   ? g_bc7_table_anchor_index_second_subset[partition]
               : kind == 1 ? g_bc7_table_anchor_index_third_subset_1[partition]
                           : g_bc7_table_anchor_index_third_subset_2[partition];
#endif
    }

    CDM_INLINE float decode_bc7_channel(uint32_t value, bool is_srgb)
    {
#if defined(__CUDA_ARCH__)
        return decode_channel_cuda(value, is_srgb);
#else
        return decode_channel(value, is_srgb);
#endif
    }

    struct Block
    {
        uint64_t low;
        uint64_t high;
    };

    struct SymbolicBlock
    {
        uint8_t mode = 8;
        uint8_t subset_count = 0;

        uint8_t endpoints[6][4]{};
        uint8_t subset[16]{};

        uint8_t color_index[16]{};
        uint8_t alpha_index[16]{};

        uint8_t color_bits = 0;
        uint8_t alpha_bits = 0;
        uint8_t rotation = 0;
    };

    struct HistogramEntry
    {
        uint16_t key;
        uint8_t counts[4];
    };

    static_assert(sizeof(Block) == 16, "A BC7 block must be 16 bytes");

    CDM_INLINE uint64_t get_bits(const Block &block, uint32_t bit, uint32_t count)
    {
        assert(count <= 64 && bit <= 128 - count);

        if (count == 0)
        {
            return 0;
        }

        uint64_t value;
        if (bit < 64)
        {
            value = block.low >> bit;
            if (bit + count > 64)
            {
                value |= block.high << (64 - bit);
            }
        }
        else
        {
            value = block.high >> (bit - 64);
        }

        return count == 64 ? value : value & ((uint64_t{1} << count) - 1);
    }

    CDM_INLINE uint8_t get_mode(const Block &block)
    {
        for (uint8_t mode = 0; mode < 8; ++mode)
        {
            if ((block.low >> mode) & uint64_t{1})
            {
                return mode;
            }
        }

        // Not found mode.
        return 8;
    }

    CDM_INLINE SymbolicBlock parse_symbolic_block(const Block &block)
    {
#if defined(__CUDA_ARCH__)
        constexpr uint8_t subset_counts[8] = {3, 2, 3, 2, 1, 1, 1, 2};
        constexpr uint8_t color_precision[8] = {4, 6, 5, 7, 5, 7, 7, 5};
        constexpr uint8_t alpha_precision[8] = {0, 0, 0, 0, 6, 8, 7, 5};
#endif
        SymbolicBlock out{};
        out.mode = get_mode(block);

        if (out.mode == 8)
        {
            return out;
        }

        out.subset_count = subset_counts[out.mode];

        // Skip the unary mode prefix before reading the mode-specific fields.
        uint32_t bit = out.mode + 1;
        uint32_t partition = 0;
        if (out.subset_count > 1)
        {
            const uint32_t partition_bits = out.mode == 0 ? 4 : 6;
            partition = static_cast<uint32_t>(get_bits(block, bit, partition_bits));
            bit += partition_bits;
        }

        if (out.mode == 4 || out.mode == 5)
        {
            out.rotation = static_cast<uint8_t>(get_bits(block, bit, 2));
            bit += 2;
        }

        uint32_t index_selection = 0;
        if (out.mode == 4)
        {
            index_selection = static_cast<uint32_t>(get_bits(block, bit, 1));
            bit += 1;
        }

        const uint32_t endpoint_count = out.subset_count * 2;
        const uint32_t rgb_bits = color_precision[out.mode];
        const uint32_t alpha_bits = alpha_precision[out.mode];

        // Endpoints are stored by channel: all R values, then G, B, and A.
        for (uint32_t channel = 0; channel < 3; ++channel)
        {
            for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
            {
                out.endpoints[endpoint][channel] = static_cast<uint8_t>(get_bits(block, bit, rgb_bits));
                bit += rgb_bits;
            }
        }

        if (alpha_bits != 0)
        {
            for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
            {
                out.endpoints[endpoint][3] = static_cast<uint8_t>(get_bits(block, bit, alpha_bits));
                bit += alpha_bits;
            }
        }

        const bool has_pbits = out.mode == 0 || out.mode == 1 || out.mode == 3 || out.mode == 6 || out.mode == 7;

        if (has_pbits)
        {
            const uint32_t channel_count = alpha_bits != 0 ? 4 : 3;

            if (out.mode == 1)
            {
                // Mode 1 shares one P-bit between both endpoints of each subset.
                for (uint32_t subset = 0; subset < out.subset_count; ++subset)
                {
                    const uint8_t pbit = static_cast<uint8_t>(get_bits(block, bit, 1));
                    bit += 1;

                    for (uint32_t endpoint = subset * 2; endpoint < subset * 2 + 2; ++endpoint)
                    {
                        for (uint32_t channel = 0; channel < channel_count; ++channel)
                        {
                            out.endpoints[endpoint][channel] =
                                static_cast<uint8_t>((out.endpoints[endpoint][channel] << 1) | pbit);
                        }
                    }
                }
            }
            else
            {
                for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
                {
                    const uint8_t pbit = static_cast<uint8_t>(get_bits(block, bit, 1));
                    bit += 1;

                    for (uint32_t channel = 0; channel < channel_count; ++channel)
                    {
                        out.endpoints[endpoint][channel] =
                            static_cast<uint8_t>((out.endpoints[endpoint][channel] << 1) | pbit);
                    }
                }
            }
        }

        const uint32_t rgb_precision = rgb_bits + (has_pbits ? 1u : 0u);
        const uint32_t alpha_endpoint_precision = alpha_bits + (has_pbits ? 1u : 0u);

        // Expand to eight bits by copying the high bits into the empty low bits.
        for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
        {
            for (uint32_t channel = 0; channel < 3; ++channel)
            {
                const uint32_t value = static_cast<uint32_t>(out.endpoints[endpoint][channel]) << (8 - rgb_precision);
                out.endpoints[endpoint][channel] = static_cast<uint8_t>(value | (value >> rgb_precision));
            }

            if (alpha_bits != 0)
            {
                const uint32_t value = static_cast<uint32_t>(out.endpoints[endpoint][3])
                                       << (8 - alpha_endpoint_precision);
                out.endpoints[endpoint][3] = static_cast<uint8_t>(value | (value >> alpha_endpoint_precision));
            }
            else
            {
                out.endpoints[endpoint][3] = 255;
            }
        }

        out.color_bits = (out.mode == 0 || out.mode == 1) ? 3 : (out.mode == 6 ? 4 : 2);
        out.alpha_bits = out.mode == 4 ? 3 : (out.mode == 5 ? 2 : 0);

        // Anchor indices omit their most significant bit, which is always zero.
        for (uint32_t texel = 0; texel < 16; ++texel)
        {
            bool is_anchor = texel == 0;
            if (out.subset_count == 2)
            {
                out.subset[texel] = partition_subset(partition, texel);
                is_anchor |= texel == partition_anchor(partition);
            }
            else if (out.subset_count == 3)
            {
                out.subset[texel] = partition_subset(partition, texel, true);
                is_anchor |= texel == partition_anchor(partition, 1) || texel == partition_anchor(partition, 2);
            }

            const uint32_t count = out.color_bits - is_anchor;
            out.color_index[texel] = static_cast<uint8_t>(get_bits(block, bit, count));
            bit += count;
        }

        if (out.alpha_bits != 0)
        {
            // The second stream has its own anchor at texel zero.
            for (uint32_t texel = 0; texel < 16; ++texel)
            {
                const uint32_t count = out.alpha_bits - (texel == 0);
                out.alpha_index[texel] = static_cast<uint8_t>(get_bits(block, bit, count));
                bit += count;
            }
        }

        // Normalize Mode 4 so each stream drives its named channels.
        if (out.mode == 4 && index_selection != 0)
        {
            for (uint32_t texel = 0; texel < 16; ++texel)
            {
                const uint8_t index = out.color_index[texel];
                out.color_index[texel] = out.alpha_index[texel];
                out.alpha_index[texel] = index;
            }

            const uint8_t bits = out.color_bits;
            out.color_bits = out.alpha_bits;
            out.alpha_bits = bits;
        }

        // Every valid BC7 mode consumes exactly one 128-bit block.
        assert(bit == 128);
        return out;
    }

    inline std::array<std::array<uint8_t, 2048>, 2> make_mode6_quantization_lut()
    {
        std::array<std::array<uint8_t, 2048>, 2> table{};

        for (uint32_t pbit = 0; pbit < 2; ++pbit)
        {
            uint32_t quantized = 0;

            for (uint32_t bucket = 0; bucket < 2048; ++bucket)
            {
                const float target = static_cast<float>(bucket) / 2048.0f;

                // Store the nearest candidate at the start of each linear interval.
                while (quantized < 127)
                {
                    const uint32_t lower_value = (quantized << 1) | pbit;
                    const uint32_t upper_value = ((quantized + 1) << 1) | pbit;

                    const float lower = decode_bc7_channel(lower_value, true);
                    const float upper = decode_bc7_channel(upper_value, true);
                    const float midpoint = (lower + upper) * 0.5f;

                    if (target <= midpoint)
                    {
                        break;
                    }

                    ++quantized;
                }

                table[pbit][bucket] = static_cast<uint8_t>(quantized);
            }
        }

        return table;
    }

    inline const std::array<std::array<uint8_t, 2048>, 2> &get_mode6_quantization_lut()
    {
        static const auto table = make_mode6_quantization_lut();
        return table;
    }

    CDM_INLINE uint8_t interpolate_channel(uint8_t e0, uint8_t e1, uint32_t index, uint32_t index_bits)
    {
        assert(index_bits >= 2 && index_bits <= 4);
        assert(index < (1u << index_bits));

#if defined(__CUDA_ARCH__)
        constexpr uint8_t weights2[4] = {0, 21, 43, 64};
        constexpr uint8_t weights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
        constexpr uint8_t weights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
#endif
        const uint8_t *weights = index_bits == 2 ? weights2 : index_bits == 3 ? weights3 : weights4;
        const uint32_t weight = weights[index];

        // BC7 uses integer weights and rounds before dividing by 64.
        return static_cast<uint8_t>(((64u - weight) * e0 + weight * e1 + 32u) >> 6);
    }

    CDM_INLINE void reconstruct_color(const SymbolicBlock &block, uint32_t subset, uint32_t color_index,
                                      uint32_t alpha_index, uint8_t rgba[4])
    {
        assert(block.mode < 8);
        assert(subset < block.subset_count);
        assert(block.rotation <= 3);

        const uint8_t *e0 = block.endpoints[subset * 2];
        const uint8_t *e1 = block.endpoints[subset * 2 + 1];

        for (uint32_t channel = 0; channel < 3; ++channel)
        {
            rgba[channel] = interpolate_channel(e0[channel], e1[channel], color_index, block.color_bits);
        }

        // Modes without a sperate alpha stream share the color index.
        const bool seperate_alpha = block.alpha_bits != 0;
        rgba[3] = interpolate_channel(e0[3], e1[3], seperate_alpha ? alpha_index : color_index,
                                      seperate_alpha ? block.alpha_bits : block.color_bits);

        if (block.rotation != 0)
        {
            const uint32_t channel = block.rotation - 1;
            const uint32_t value = rgba[channel];
            rgba[channel] = rgba[3];
            rgba[3] = value;
        }
    }

    CDM_INLINE Float4 rgba8_to_linear(const uint8_t rgba[4])
    {
        float r = decode_bc7_channel(rgba[0], true);
        float g = decode_bc7_channel(rgba[1], true);
        float b = decode_bc7_channel(rgba[2], true);
        float a = static_cast<float>(rgba[3]) * inv_255;
        return {r, g, b, a};
    }

    inline uint32_t build_quadrant_histogram(const SymbolicBlock &block, HistogramEntry entries[16],
                                             uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        assert(block.mode < 8);
        assert(valid_width >= 1 && valid_width <= 4);
        assert(valid_height >= 1 && valid_height <= 4);

        uint32_t entry_count = 0;
        for (uint32_t texel = 0; texel < 16; ++texel)
        {
            const uint32_t x = texel % 4;
            const uint32_t y = texel / 4;
            const uint32_t quadrant = (y / 2) * 2 + x / 2;

            // Repeat the last valid texel at partial block edges.
            const uint32_t sx = x < valid_width ? x : valid_width - 1;
            const uint32_t sy = y < valid_height ? y : valid_height - 1;
            const uint32_t source = sy * 4 + sx;

            const uint16_t key = static_cast<uint16_t>(block.subset[source] | (block.color_index[source] << 2) |
                                                       (block.alpha_index[source] << 6));
            uint32_t entry = 0;

            while (entry < entry_count && entries[entry].key != key)
            {
                ++entry;
            }

            if (entry == entry_count)
            {
                entries[entry_count++] = {key, {}};
            }
            ++entries[entry].counts[quadrant];
        }

        return entry_count;
    }

    inline void get_quadrant_means(const Block &block, Float4 means[4], uint32_t valid_width = 4,
                                   uint32_t valid_height = 4)
    {
        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
        {
            means[quadrant] = {};
        }

        const SymbolicBlock symbolic = parse_symbolic_block(block);

        // Invalid mode prefixes decode to transparent block.
        if (symbolic.mode == 8)
        {
            return;
        }

        HistogramEntry entries[16];
        const uint32_t entry_count = build_quadrant_histogram(symbolic, entries, valid_width, valid_height);

        for (uint32_t entry = 0; entry < entry_count; ++entry)
        {
            const uint16_t key = entries[entry].key;
            const uint32_t subset = key & 3u;
            const uint32_t color_index = (key >> 2) & 15u;
            const uint32_t alpha_index = (key >> 6) & 15u;

            // Reconstruct each unique combination once for all four quadrants.
            uint8_t rgba[4];
            reconstruct_color(symbolic, subset, color_index, alpha_index, rgba);

            const Float4 color = rgba8_to_linear(rgba);

            for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
            {
                const float weight = static_cast<float>(entries[entry].counts[quadrant]) * 0.25f;

                means[quadrant].r += color.r * weight;
                means[quadrant].g += color.g * weight;
                means[quadrant].b += color.b * weight;
                means[quadrant].a += color.a * weight;
            }
        }
    }

    inline void compute_child_moments(const Float4 samples[16], Float4 &mean, SymMat4 &covariance, uint32_t count = 16)
    {
        const float weight = 1.0f / static_cast<float>(count);
        mean = {};

        for (uint32_t i = 0; i < count; ++i)
        {
            mean.r += samples[i].r;
            mean.g += samples[i].g;
            mean.b += samples[i].b;
            mean.a += samples[i].a;
        }

        mean.r *= weight;
        mean.g *= weight;
        mean.b *= weight;
        mean.a *= weight;

        // Center samples before accumulating covariance to reduce cancellation.
        covariance = {};
        for (uint32_t i = 0; i < count; ++i)
        {
            const float r = samples[i].r - mean.r;
            const float g = samples[i].g - mean.g;
            const float b = samples[i].b - mean.b;
            const float a = samples[i].a - mean.a;

            covariance.rr += r * r * weight;
            covariance.rg += r * g * weight;
            covariance.rb += r * b * weight;
            covariance.ra += r * a * weight;
            covariance.gg += g * g * weight;
            covariance.gb += g * b * weight;
            covariance.ga += g * a * weight;
            covariance.bb += b * b * weight;
            covariance.ba += b * a * weight;
            covariance.aa += a * a * weight;
        }
    }

    inline bool has_alpha_samples(const Float4 samples[16])
    {
        for (uint32_t i = 0; i < 16; ++i)
        {
            if (samples[i].a < 1.0f)
            {
                return true;
            }
        }
        return false;
    }

    inline void compute_encoding_moments(const Float4 *samples, uint32_t count, bool has_alpha, Float4 &mean,
                                         SymMat4 &covariance, Float4 &axis)
    {
        const float weight = 1.0f / static_cast<float>(count);
        if (!has_alpha)
        {
            Float3 rgb_mean{};
            for (uint32_t i = 0; i < count; ++i)
            {
                rgb_mean += {samples[i].r, samples[i].g, samples[i].b};
            }
            rgb_mean = rgb_mean * weight;

            SymMat3 rgb_covariance{};
            for (uint32_t i = 0; i < count; ++i)
            {
                const Float3 rgb = {samples[i].r, samples[i].g, samples[i].b};
                accumulate_outer(rgb_covariance, rgb - rgb_mean, weight);
            }
            const Float3 rgb_axis = compute_principal_axis(rgb_covariance);

            mean = {rgb_mean.r, rgb_mean.g, rgb_mean.b, 1.0f};
            axis = {rgb_axis.r, rgb_axis.g, rgb_axis.b, 0.0f};

            // Embed RGB statistics only to share the mode estimates and packing path.
            covariance = {rgb_covariance.rr,
                          rgb_covariance.rg,
                          rgb_covariance.rb,
                          0.0f,
                          rgb_covariance.gg,
                          rgb_covariance.gb,
                          0.0f,
                          rgb_covariance.bb,
                          0.0f,
                          0.0f};
            return;
        }

        compute_child_moments(samples, mean, covariance, count);
        axis = compute_principal_axis(covariance);
    }

    inline void project_samples(const Float4 *samples, uint32_t count, const Float4 &mean, const Float4 &axis,
                                float *projections, float &minimum, float &maximum)
    {
        minimum = 1e30f;
        maximum = -1e30f;
        for (uint32_t i = 0; i < count; ++i)
        {
            const float projection = dot(samples[i] - mean, axis);
            projections[i] = projection;
            minimum = std::min(minimum, projection);
            maximum = std::max(maximum, projection);
        }
    }

    inline void fit_endpoints_scalar(const Float4 samples[16], Float4 &low, Float4 &high, Float4 &mean, Float4 &axis,
                                     float projections[16])
    {
        SymMat4 covariance;
        compute_encoding_moments(samples, 16, has_alpha_samples(samples), mean, covariance, axis);
        float minimum, maximum;
        project_samples(samples, 16, mean, axis, projections, minimum, maximum);
        const float variance = covariance.rr + covariance.gg + covariance.bb + covariance.aa;
        low = variance < 1e-10f ? mean : mean + axis * minimum;
        high = variance < 1e-10f ? mean : mean + axis * maximum;
    }

    // Keep quantization ties stable under the release build's fast-math flags.
#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#endif

    CDM_INLINE uint8_t quantize_mode6_channel(float target, uint32_t pbit, bool is_srgb)
    {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
#if defined(__CUDA_ARCH__)
        const uint32_t bucket = static_cast<uint32_t>(target * 2048.0f);
        uint32_t candidate;
        if (is_srgb)
        {
            candidate = device_mode6_lut[pbit][bucket < 2048 ? bucket : 2047];
        }
        else
        {
            const float position = (target * 255.0f - pbit) * 0.5f;
            candidate = static_cast<uint32_t>(fminf(127.0f, fmaxf(0.0f, position)));
        }
        uint32_t best = candidate;
        const float delta = target - decode_bc7_channel((candidate << 1) | pbit, is_srgb);
        float best_error = delta * delta;
        const uint32_t first = candidate > 0 ? candidate - 1 : 0;
        const uint32_t last = candidate < 127 ? candidate + 1 : 127;
        for (uint32_t q = first; q <= last; ++q)
        {
            const float d = target - decode_bc7_channel((q << 1) | pbit, is_srgb);
            const float error = d * d;
            if (error < best_error || (error == best_error && q < best))
            {
                best = q;
                best_error = error;
            }
        }
        return static_cast<uint8_t>(best);
#else
        assert(target >= 0.0f && target <= 1.0f);
        assert(pbit <= 1);

        // Constant values prevent fused arithmetic from changing midpoint ties.
        static constexpr auto alpha_values = []()
        {
            std::array<float, 256> values{};
            for (uint32_t i = 0; i < 256; ++i)
            {
                values[i] = static_cast<float>(i) * inv_255;
            }
            return values;
        }();

        uint32_t candidate;

        if (is_srgb)
        {
            // Build once, then use a direct lookup for each RGB channel.
            const auto &table = get_mode6_quantization_lut();

            const uint32_t bucket = static_cast<uint32_t>(target * 2048.0f);
            const uint32_t index = bucket < 2048 ? bucket : 2047;

            candidate = table[pbit][index];
        }
        else
        {
            // Alpha has uniformly spaced reconstruction values.
            const float position = (target * 255.0f - pbit) * 0.5f;
            const float clamped = position < 0.0f ? 0.0f : (position > 127.0f ? 127.0f : position);
            candidate = static_cast<uint32_t>(clamped);
        }

        uint32_t best = candidate;
        const uint32_t value = (candidate << 1) | pbit;
        const float decoded = is_srgb ? c_srgb_to_linear[value] : alpha_values[value];
        const float difference = target - decoded;
        float best_error = difference * difference;

        // Check the predecessor to handle rounded LUT boundaries and ties.
        if (candidate > 0)
        {
            const uint32_t previous = candidate - 1;
            const uint32_t previous_value = (previous << 1) | pbit;
            const float previous_decoded = is_srgb ? c_srgb_to_linear[previous_value] : alpha_values[previous_value];
            const float previous_difference = target - previous_decoded;
            const float previous_error = previous_difference * previous_difference;

            if (previous_error <= best_error)
            {
                best = previous;
                best_error = previous_error;
            }
        }

        // Correct the candidate when the target crosses the next boundary.
        if (candidate < 127)
        {
            const uint32_t next = candidate + 1;
            const uint32_t next_value = (next << 1) | pbit;
            const float next_decoded = is_srgb ? c_srgb_to_linear[next_value] : alpha_values[next_value];
            const float next_difference = target - next_decoded;
            const float next_error = next_difference * next_difference;

            if (next_error < best_error)
            {
                best = next;
            }
        }

        return static_cast<uint8_t>(best);
#endif
    }

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif

    CDM_INLINE void quantize_mode6_endpoint(const Float4 &endpoint, uint8_t quantized[4], uint8_t &pbit)
    {
        const Float4 color = clamp01(endpoint);
        const float target[4] = {color.r, color.g, color.b, color.a};

        float best_error = 1e30f;

        // Alpha 255 requires P-bit 1 to preserve an opaque endpoint exactly.
        const uint32_t first_pbit = color.a == 1.0f ? 1u : 0u;

        for (uint32_t candidate_pbit = first_pbit; candidate_pbit < 2; ++candidate_pbit)
        {
            uint8_t candidate[4];
            uint8_t reconstructed[4];

            for (uint32_t channel = 0; channel < 4; ++channel)
            {
                const bool is_srgb = channel < 3;
                candidate[channel] = quantize_mode6_channel(target[channel], candidate_pbit, is_srgb);
                reconstructed[channel] = static_cast<uint8_t>((candidate[channel] << 1) | candidate_pbit);
            }

            // Compare endpoint error in linear RGBA space.
            const Float4 decoded = rgba8_to_linear(reconstructed);
            const Float4 difference = color - decoded;
            const float error = length_sq(difference);

            if (error < best_error)
            {
                best_error = error;
                pbit = static_cast<uint8_t>(candidate_pbit);

                for (uint32_t channel = 0; channel < 4; ++channel)
                {
                    quantized[channel] = candidate[channel];
                }
            }
        }
    }

    CDM_INLINE void put_bits(Block &block, uint32_t &bit, uint32_t count, uint32_t value)
    {
        assert(count > 0 && count < 32 && bit + count <= 128);
        assert(value < (1u << count));

        if (bit < 64)
        {
            block.low |= static_cast<uint64_t>(value) << bit;
            if (bit + count > 64)
            {
                block.high |= static_cast<uint64_t>(value) >> (64 - bit);
            }
        }
        else
        {
            block.high |= static_cast<uint64_t>(value) << (bit - 64);
        }
        bit += count;
    }

    inline uint16_t desired_partition_mask(const float projections[16])
    {
        uint16_t mask = 0;
        for (uint32_t i = 0; i < 16; ++i)
        {
            mask |= uint16_t(projections[i] > 0.0f) << i;
        }
        return mask;
    }

    static constexpr std::array<uint16_t, 64> kPartitionMasks2 = []
    {
        std::array<uint16_t, 64> masks{};
        for (uint32_t p = 0; p < 64; ++p)
        {
            for (uint32_t t = 0; t < 16; ++t)
            {
                masks[p] |= uint16_t(g_bc7_partition2[p * 16 + t]) << t;
            }
        }
        return masks;
    }();

    CDM_INLINE uint16_t partition_mask(uint32_t partition)
    {
#if defined(__CUDA_ARCH__)
        return device_partition_masks[partition];
#else
        return kPartitionMasks2[partition];
#endif
    }

    CDM_INLINE uint32_t popcount16(uint32_t x)
    {
        x -= (x >> 1) & 0x5555u;
        x = (x & 0x3333u) + ((x >> 2) & 0x3333u);
        x = (x + (x >> 4)) & 0x0f0fu;
        return (x + (x >> 8)) & 0x1fu;
    }

    CDM_INLINE uint32_t choose_partition(uint16_t desired)
    {
        uint32_t best_partition = 0;
        // Subset labels may be swapped without changing the partition.
        uint32_t best_distance = 17;
        for (uint32_t p = 0; p < 64; ++p)
        {
            const uint32_t distance = popcount16(desired ^ partition_mask(p));
            const uint32_t match = (distance < 16u - distance ? distance : 16u - distance);
            if (match < best_distance)
            {
                best_distance = match;
                best_partition = p;
                if (match == 0)
                {
                    break;
                }
            }
        }
        return best_partition;
    }

    struct PartitionSubsetStats
    {
        uint32_t count;
        Float4 mean;
        SymMat4 cov;
        Float4 axis;
        float lo, hi;
    };

    inline void compute_partition_subset_stats(const Float4 samples[16], uint32_t partition,
                                               PartitionSubsetStats stats[2], bool has_alpha,
                                               float sample_projections[16])
    {
        for (uint32_t subset = 0; subset < 2; ++subset)
        {
            Float4 grouped[16];
            uint32_t texels[16];
            uint32_t count = 0;
            for (uint32_t texel = 0; texel < 16; ++texel)
            {
                if (partition_subset(partition, texel) == subset)
                {
                    texels[count] = texel;
                    grouped[count++] = samples[texel];
                }
            }
            assert(count > 0);

            Float4 mean, axis;
            SymMat4 covariance;
            compute_encoding_moments(grouped, count, has_alpha, mean, covariance, axis);
            float projections[16];
            float minimum, maximum;
            project_samples(grouped, count, mean, axis, projections, minimum, maximum);
            for (uint32_t i = 0; i < count; ++i)
            {
                sample_projections[texels[i]] = projections[i];
            }
            stats[subset] = {count, mean, covariance, axis, minimum, maximum};
        }
    }

    // Covariance is normalized by the number of samples in the group.
    CDM_INLINE float estimate_rgb_line_sse(const SymMat4 &cov, const Float4 &axis, uint32_t sample_count)
    {
        const Float4 rgb_axis = {axis.r, axis.g, axis.b, 0.0f};
        const float total = cov.rr + cov.gg + cov.bb;
        const float along_axis = dot(rgb_axis, multiply(cov, rgb_axis));
        const float residual = total - along_axis;
        return sample_count * (residual > 0.0f ? residual : 0.0f);
    }

    CDM_INLINE float estimate_rgb_index_sse(const Float4 &low, const Float4 &high, uint32_t sample_count,
                                            uint32_t index_levels)
    {
        const Float4 span = high - low;
        const float span_sq = span.r * span.r + span.g * span.g + span.b * span.b;
        const float steps = float(index_levels - 1);

        return float(sample_count) * span_sq / (12.0f * steps * steps);
    }

    CDM_INLINE float estimate_rgb_endpoint_sse(uint32_t sample_count, uint32_t endpoint_levels, uint32_t index_levels)
    {
        const float step = 1.0f / float(endpoint_levels - 1);
        const float levels = float(index_levels);
        const float endpoint_weight = (2.0f * levels - 1.0f) / (3.0f * (levels - 1.0f));

        return float(sample_count) * 3.0f * step * step * endpoint_weight / 12.0f;
    }

    CDM_INLINE float estimate_mode1_sse(const PartitionSubsetStats stats[2])
    {
        float error = 0.0f;
        for (uint32_t s = 0; s < 2; ++s)
        {
            const Float4 low = clamp01(stats[s].mean + stats[s].axis * stats[s].lo);
            const Float4 high = clamp01(stats[s].mean + stats[s].axis * stats[s].hi);

            error += estimate_rgb_line_sse(stats[s].cov, stats[s].axis, stats[s].count);
            error += estimate_rgb_index_sse(low, high, stats[s].count, 8);
            error += estimate_rgb_endpoint_sse(stats[s].count, 64, 8);
        }

        return error;
    }

    CDM_INLINE float estimate_mode6_sse(const SymMat4 &cov, const Float4 &axis, const Float4 &low, const Float4 &high)
    {
        return estimate_rgb_line_sse(cov, axis, 16) + estimate_rgb_index_sse(low, high, 16, 16) +
               estimate_rgb_endpoint_sse(16, 128, 16);
    }

    CDM_INLINE float estimate_rgba_sse(const SymMat4 &cov, const Float4 &axis, const Float4 &low, const Float4 &high,
                                       uint32_t count, uint32_t endpoint_levels, uint32_t index_levels)
    {
        const float residual = (cov.rr + cov.gg + cov.bb + cov.aa) - dot(axis, multiply(cov, axis));
        const float steps = float(index_levels - 1);
        return float(count) * fmaxf(0.0f, residual) + float(count) * length_sq(high - low) / (12.0f * steps * steps) +
               estimate_rgb_endpoint_sse(count, endpoint_levels, index_levels) * (4.0f / 3.0f);
    }

    CDM_INLINE uint32_t choose_partitioned_mode(const PartitionSubsetStats stats[2], bool has_alpha, const SymMat4 &cov,
                                                const Float4 &axis, const Float4 &low, const Float4 &high)
    {
        if (!has_alpha)
        {
            return estimate_mode1_sse(stats) < estimate_mode6_sse(cov, axis, low, high) ? 1u : 6u;
        }
        float error = 0.0f;
        for (uint32_t s = 0; s < 2; ++s)
        {
            error +=
                estimate_rgba_sse(stats[s].cov, stats[s].axis, clamp01(stats[s].mean + stats[s].axis * stats[s].lo),
                                  clamp01(stats[s].mean + stats[s].axis * stats[s].hi), stats[s].count, 32, 4);
        }
        return error < estimate_rgba_sse(cov, axis, low, high, 16, 128, 16) ? 7u : 6u;
    }

    CDM_INLINE uint8_t expand_partition_channel(uint32_t quantized, uint32_t pbit, uint32_t mode)
    {
        const uint32_t precision = mode == 1 ? 7 : 6;
        const uint32_t value = ((quantized << 1) | pbit) << (8 - precision);
        const uint8_t expanded = static_cast<uint8_t>(value | (value >> precision));
        return expanded;
    }

    CDM_INLINE uint8_t quantize_partition_channel(float target, uint32_t pbit, uint32_t mode, bool is_srgb)
    {
        uint32_t lower = 0;
        uint32_t upper = mode == 1 ? 63 : 31;
        while (lower < upper)
        {
            const uint32_t middle = (lower + upper) / 2;
            const uint8_t value = expand_partition_channel(middle, pbit, mode);
            const float decoded = decode_bc7_channel(value, is_srgb);
            if (decoded < target)
            {
                lower = middle + 1;
            }
            else
            {
                upper = middle;
            }
        }

        if (lower > 0)
        {
            const uint8_t a = expand_partition_channel(lower - 1, pbit, mode);
            const uint8_t b = expand_partition_channel(lower, pbit, mode);
            const float da = target - decode_bc7_channel(a, is_srgb);
            const float db = target - decode_bc7_channel(b, is_srgb);
            if (da * da <= db * db)
            {
                --lower;
            }
        }
        return static_cast<uint8_t>(lower);
    }

    CDM_INLINE void quantize_partition_endpoints(const Float4 endpoints[4], uint32_t mode, uint8_t quantized[4][4],
                                                 uint8_t pbits[4])
    {
        const uint32_t channels = mode == 1 ? 3 : 4;
        const uint32_t group_size = mode == 1 ? 2 : 1;
        for (uint32_t first = 0; first < 4; first += group_size)
        {
            float best_error = 1e30f;
            const uint32_t first_pbit = mode == 7 && endpoints[first].a == 1.0f ? 1 : 0;
            for (uint32_t pbit = first_pbit; pbit < 2; ++pbit)
            {
                uint8_t candidate[2][4]{};
                float error = 0.0f;
                for (uint32_t local = 0; local < group_size; ++local)
                {
                    const Float4 color = endpoints[first + local];
                    const float target[4] = {color.r, color.g, color.b, color.a};
                    for (uint32_t channel = 0; channel < channels; ++channel)
                    {
                        const bool is_srgb = channel < 3;
                        const uint8_t q = quantize_partition_channel(target[channel], pbit, mode, is_srgb);
                        candidate[local][channel] = q;
                        const uint8_t value = expand_partition_channel(q, pbit, mode);
                        const float difference = target[channel] - decode_bc7_channel(value, is_srgb);
                        error += difference * difference;
                    }
                }
                if (error < best_error)
                {
                    best_error = error;
                    for (uint32_t local = 0; local < group_size; ++local)
                    {
                        pbits[first + local] = static_cast<uint8_t>(pbit);
                        for (uint32_t channel = 0; channel < channels; ++channel)
                        {
                            quantized[first + local][channel] = candidate[local][channel];
                        }
                    }
                }
            }
        }
    }

    inline void make_palette(uint32_t mode, const uint8_t quantized[4][4], const uint8_t pbits[4],
                             Float4 palette[2][16])
    {
        const uint32_t subsets = mode == 6 ? 1 : 2;
        const uint32_t bits = mode == 6 ? 4 : (mode == 1 ? 3 : 2);
        const uint32_t palette_size = 1u << bits;
        for (uint32_t subset = 0; subset < subsets; ++subset)
        {
            uint8_t endpoints[2][4];
            for (uint32_t endpoint = 0; endpoint < 2; ++endpoint)
            {
                const uint32_t source = subset * 2 + endpoint;
                for (uint32_t channel = 0; channel < 4; ++channel)
                {
                    uint8_t value;
                    if (mode == 6)
                    {
                        value = static_cast<uint8_t>((quantized[source][channel] << 1) | pbits[source]);
                    }
                    else if (mode == 1 && channel == 3)
                    {
                        value = 255;
                    }
                    else
                    {
                        value = expand_partition_channel(quantized[source][channel], pbits[source], mode);
                    }
                    endpoints[endpoint][channel] = value;
                }
            }
            for (uint32_t index = 0; index < palette_size; ++index)
            {
                uint8_t rgba[4];
                for (uint32_t channel = 0; channel < 4; ++channel)
                {
                    rgba[channel] = interpolate_channel(endpoints[0][channel], endpoints[1][channel], index, bits);
                }
                palette[subset][index] = rgba8_to_linear(rgba);
            }
        }
    }

    // Choose indices by cached sample projections against the quantized palette.
    inline float assign_selectors(const Float4 samples[16], uint32_t mode, uint32_t partition,
                                  const uint8_t quantized[4][4], const uint8_t pbits[4], uint8_t selectors[16],
                                  const Float4 means[2], const Float4 axes[2], const float sample_projections[16])
    {
        const uint32_t subsets = mode == 6 ? 1 : 2;
        const uint32_t palette_size = mode == 6 ? 16 : mode == 1 ? 8 : 4;
        Float4 palette[2][16];
        make_palette(mode, quantized, pbits, palette);

        float projections[2][16];
        for (uint32_t subset = 0; subset < subsets; ++subset)
        {
            for (uint32_t index = 0; index < palette_size; ++index)
            {
                projections[subset][index] = dot(palette[subset][index] - means[subset], axes[subset]);
            }
        }
        float total_error = 0.0f;
        for (uint32_t texel = 0; texel < 16; ++texel)
        {
            const uint32_t subset = subsets == 1 ? 0 : partition_subset(partition, texel);
            const float target = sample_projections[texel];
            float best_error = 1e30f;
            uint32_t best_index = 0;
            for (uint32_t index = 0; index < palette_size; ++index)
            {
                const float delta = target - projections[subset][index];
                const float error = delta * delta;
                if (error < best_error)
                {
                    best_error = error;
                    best_index = index;
                }
            }
            selectors[texel] = static_cast<uint8_t>(best_index);
            total_error += length_sq(samples[texel] - palette[subset][best_index]);
        }
        return total_error;
    }

    CDM_INLINE void fix_anchors(uint32_t mode, uint32_t partition, uint8_t quantized[4][4], uint8_t pbits[4],
                                uint8_t selectors[16])
    {
        const uint32_t subsets = mode == 6 ? 1 : 2;
        const uint32_t bits = mode == 6 ? 4 : (mode == 1 ? 3 : 2);
        const uint32_t maximum = (1u << bits) - 1;
        for (uint32_t subset = 0; subset < subsets; ++subset)
        {
            const uint32_t anchor = subset == 0 ? 0 : partition_anchor(partition);
            if (selectors[anchor] < (1u << (bits - 1)))
            {
                continue;
            }
            for (uint32_t channel = 0; channel < 4; ++channel)
            {
                const uint8_t value = quantized[subset * 2][channel];
                quantized[subset * 2][channel] = quantized[subset * 2 + 1][channel];
                quantized[subset * 2 + 1][channel] = value;
            }
            const uint8_t pbit = pbits[subset * 2];
            pbits[subset * 2] = pbits[subset * 2 + 1];
            pbits[subset * 2 + 1] = pbit;
            for (uint32_t texel = 0; texel < 16; ++texel)
            {
                const uint32_t selected_subset = subsets == 1 ? 0 : partition_subset(partition, texel);
                if (selected_subset == subset)
                {
                    selectors[texel] = static_cast<uint8_t>(maximum - selectors[texel]);
                }
            }
        }
    }

    CDM_INLINE Block pack_block(uint32_t mode, uint32_t partition, const uint8_t quantized[4][4],
                                const uint8_t pbits[4], const uint8_t selectors[16])
    {
        assert(mode == 1 || mode == 6 || mode == 7);
        Block block{};
        uint32_t bit = 0;
        put_bits(block, bit, mode + 1, 1u << mode);
        if (mode != 6)
        {
            put_bits(block, bit, 6, partition);
        }
        const uint32_t endpoint_count = mode == 6 ? 2 : 4;
        const uint32_t channels = mode == 1 ? 3 : 4;
        const uint32_t precision = mode == 6 ? 7 : (mode == 1 ? 6 : 5);
        for (uint32_t channel = 0; channel < channels; ++channel)
        {
            for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
            {
                put_bits(block, bit, precision, quantized[endpoint][channel]);
            }
        }
        for (uint32_t endpoint = 0; endpoint < endpoint_count; endpoint += mode == 1 ? 2 : 1)
        {
            put_bits(block, bit, 1, pbits[endpoint]);
        }
        const uint32_t bits = mode == 6 ? 4 : (mode == 1 ? 3 : 2);
        for (uint32_t texel = 0; texel < 16; ++texel)
        {
            const bool anchor = texel == 0 || (mode != 6 && texel == partition_anchor(partition));
            put_bits(block, bit, bits - anchor, selectors[texel]);
        }
        assert(bit == 128);
        return block;
    }

    // Samples and selectors both use row-major texel order.
    inline Block encode_mode6_with_endpoints(const Float4 samples[16], const Float4 &low, const Float4 &high,
                                             const Float4 &mean, const Float4 &axis, const float sample_projections[16],
                                             float &error)
    {
        const Float4 means[2] = {mean, {}};
        const Float4 axes[2] = {axis, {}};
        uint8_t quantized[4][4]{};
        uint8_t pbits[4]{};
        uint8_t selectors[16];
        quantize_mode6_endpoint(low, quantized[0], pbits[0]);
        quantize_mode6_endpoint(high, quantized[1], pbits[1]);
        error = assign_selectors(samples, 6, 0, quantized, pbits, selectors, means, axes, sample_projections);
        fix_anchors(6, 0, quantized, pbits, selectors);
        const Block block = pack_block(6, 0, quantized, pbits, selectors);
        return block;
    }

    inline Block encode_mode6_samples(const Float4 samples[16], float &error)
    {
        Float4 mean, axis, low, high;
        float sample_projections[16];
        fit_endpoints_scalar(samples, low, high, mean, axis, sample_projections);
        return encode_mode6_with_endpoints(samples, low, high, mean, axis, sample_projections, error);
    }

    inline Block encode_partition_samples(const Float4 samples[16], uint32_t mode, uint32_t partition,
                                          const PartitionSubsetStats stats[2], const float sample_projections[16],
                                          float &error)
    {
        assert(mode == 1 || mode == 7);
        Float4 endpoints[4];
        Float4 means[2], axes[2];
        for (uint32_t subset = 0; subset < 2; ++subset)
        {
            means[subset] = stats[subset].mean;
            axes[subset] = stats[subset].axis;
            endpoints[subset * 2] = clamp01(means[subset] + axes[subset] * stats[subset].lo);
            endpoints[subset * 2 + 1] = clamp01(means[subset] + axes[subset] * stats[subset].hi);
        }
        uint8_t quantized[4][4]{};
        uint8_t pbits[4]{};
        uint8_t selectors[16];
        quantize_partition_endpoints(endpoints, mode, quantized, pbits);
        error =
            assign_selectors(samples, mode, partition, quantized, pbits, selectors, means, axes, sample_projections);
        fix_anchors(mode, partition, quantized, pbits, selectors);
        const Block block = pack_block(mode, partition, quantized, pbits, selectors);
        return block;
    }

    inline Block encode_partition_samples(const Float4 samples[16], uint32_t mode, uint32_t partition, float &error)
    {
        PartitionSubsetStats stats[2];
        float sample_projections[16];
        compute_partition_subset_stats(samples, partition, stats, has_alpha_samples(samples), sample_projections);
        return encode_partition_samples(samples, mode, partition, stats, sample_projections, error);
    }

    inline Block encode_samples_scalar(const Float4 samples[16])
    {
        // Choose RGB or RGBA fitting before computing any covariance.
        const bool has_alpha = has_alpha_samples(samples);
        Float4 mean, axis;
        SymMat4 covariance;
        compute_encoding_moments(samples, 16, has_alpha, mean, covariance, axis);
        const float variance = covariance.rr + covariance.gg + covariance.bb + covariance.aa;

        float sample_projections[16];
        float minimum, maximum;
        project_samples(samples, 16, mean, axis, sample_projections, minimum, maximum);

        Float4 low = mean;
        Float4 high = mean;
        float error;
        if (variance < 1e-10f)
        {
            return encode_mode6_with_endpoints(samples, low, high, mean, axis, sample_projections, error);
        }

        low = clamp01(mean + axis * minimum);
        high = clamp01(mean + axis * maximum);

        const uint32_t partition = choose_partition(desired_partition_mask(sample_projections));

        PartitionSubsetStats stats[2];
        float subset_projections[16];
        compute_partition_subset_stats(samples, partition, stats, has_alpha, subset_projections);
        const uint32_t mode = choose_partitioned_mode(stats, has_alpha, covariance, axis, low, high);
        if (mode == 6)
        {
            return encode_mode6_with_endpoints(samples, low, high, mean, axis, sample_projections, error);
        }

        return encode_partition_samples(samples, mode, partition, stats, subset_projections, error);
    }

    inline void prepare_child_samples(const Block &p00, const Block &p10, const Block &p01, const Block &p11,
                                      Float4 parent_means[4], uint32_t valid_width, uint32_t valid_height,
                                      uint32_t source_width, uint32_t source_height, Float4 samples[16])
    {
        assert(valid_width >= 1 && valid_width <= 4);
        assert(valid_height >= 1 && valid_height <= 4);
        const Block *parents[4] = {&p00, &p10, &p01, &p11};

        for (uint32_t parent = 0; parent < 4; ++parent)
        {
            const uint32_t px = source_width > 4 ? (parent % 2) * 4 : 0;
            const uint32_t py = source_height > 4 ? (parent / 2) * 4 : 0;
            const uint32_t width = source_width - px < 4 ? source_width - px : 4;
            const uint32_t height = source_height - py < 4 ? source_height - py : 4;

            Float4 quadrants[4];
            get_quadrant_means(*parents[parent], quadrants, width, height);
            for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
            {
                const uint32_t x = (parent % 2) * 2 + quadrant % 2;
                const uint32_t y = (parent / 2) * 2 + quadrant / 2;
                samples[y * 4 + x] = quadrants[quadrant];
            }
        }

        // Match the existing codecs' periodic padding for small output blocks.
        Float4 original[16];
        std::copy_n(samples, 16, original);
        for (uint32_t y = 0; y < 4; ++y)
        {
            for (uint32_t x = 0; x < 4; ++x)
            {
                samples[y * 4 + x] = original[(y % valid_height) * 4 + x % valid_width];
            }
        }

        // Preserve the unquantized means instead of decoding the output again.
        for (uint32_t parent = 0; parent < 4; ++parent)
        {
            const uint32_t offset = (parent / 2) * 8 + (parent % 2) * 2;
            const Float4 mean =
                (samples[offset] + samples[offset + 1] + samples[offset + 4] + samples[offset + 5]) * 0.25f;
            parent_means[parent] = mean;
        }
    }

    inline Block generate_child_block_scalar(const Block &p00, const Block &p10, const Block &p01, const Block &p11,
                                             Float4 parent_means[4], uint32_t valid_width, uint32_t valid_height,
                                             uint32_t source_width, uint32_t source_height)
    {
        Float4 samples[16];
        prepare_child_samples(p00, p10, p01, p11, parent_means, valid_width, valid_height, source_width, source_height,
                              samples);
        return encode_samples_scalar(samples);
    }

    inline Block generate_child_block_from_means_scalar(const MeanImage4 &source, uint32_t block_x, uint32_t block_y,
                                                        uint32_t valid_width, uint32_t valid_height)
    {
        Float4 samples[16];
        for (uint32_t y = 0; y < 4; ++y)
        {
            for (uint32_t x = 0; x < 4; ++x)
            {
                samples[y * 4 + x] = source.get(block_x * 4 + x % valid_width, block_y * 4 + y % valid_height);
            }
        }

        const Block block = encode_samples_scalar(samples);
        return block;
    }

} // namespace bc7
