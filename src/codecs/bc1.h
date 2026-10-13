#pragma once

#include "core/color.h"
#include "core/math.h"
#include "core/mean.h"

#include <iostream>
#include <array>
#include <cmath>
#include <algorithm>

namespace bc1
{
    // Map quadrant-grouped samples to row-major texel positions.
    constexpr uint8_t texel_map[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};

    template <uint32_t MaxValue>
    inline std::array<float, MaxValue> make_srgb_quantize_thresholds()
    {
        constexpr uint32_t bits = MaxValue == 31 ? 5 : 6;
        std::array<float, MaxValue> thresholds{};

        for (uint32_t i = 0; i < MaxValue; ++i)
        {
            const uint32_t next = i + 1;
            const uint32_t a = (i << (8 - bits)) | (i >> (2 * bits - 8));
            const uint32_t b = (next << (8 - bits)) | (next >> (2 * bits - 8));

            const float midpoint = static_cast<float>(a + b) * (1.0f / 510.0f);
            thresholds[i] = midpoint <= 0.04045f ? midpoint / 12.92f : std::pow((midpoint + 0.055f) / 1.055f, 2.4f);
        }

        return thresholds;
    }

    // Quantize a linear channel using cached sRGB thresholds.
    template <uint32_t MaxValue>
    inline uint32_t quantize_linear_channel(float value)
    {
        static const std::array<float, MaxValue> thresholds = make_srgb_quantize_thresholds<MaxValue>();
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);

        const float *begin = thresholds.data();
        const float *end = begin + MaxValue;
        const float *position = std::upper_bound(begin, end, value);

        return static_cast<uint32_t>(position - begin);
    }

    // Store one 4x4 BC1 block.
    struct Block
    {
        uint16_t c0;
        uint16_t c1;
        uint32_t indices;
    };

    // Store RGb values in the range [0, 255]
    // Use 32-bit channels for palette arithemic.
    struct Rgb8
    {
        uint32_t r;
        uint32_t g;
        uint32_t b;
    };

    static_assert(sizeof(Block) == 8, "A BC1 block must be 8 bytes");

    // Read the two-bit palette index. Both x and y must be in [0, 3]
    inline uint32_t get_selector(const Block &block, uint32_t x, uint32_t y)
    {
        const uint32_t texel_index = y * 4 + x;
        const uint32_t shift = texel_index * 2;
        return (block.indices >> shift) & 3u;
    }

    // Expand an RGB565 endpont to RGB8 using bit replication.
    CDM_INLINE Rgb8 decode_rgb565(uint16_t color)
    {
        const uint32_t r = (color >> 11) & 31u;
        const uint32_t g = (color >> 5) & 63u;
        const uint32_t b = color & 31u;

        return {(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)};
    }

    CDM_INLINE Rgb8 palette_color_rgb8(uint16_t c0, uint16_t c1, uint32_t selector, bool opaque = false)
    {
        const Rgb8 a = decode_rgb565(c0);
        const Rgb8 b = decode_rgb565(c1);

        if (selector == 0)
            return a;
        if (selector == 1)
            return b;

        // Encoding uses four colors even when the endpoint are equal
        if (opaque || c0 > c1)
        {
            // Four-color mode uses two interpolated colors.
            if (selector == 2)
            {
                return {(2 * a.r + b.r) / 3, (2 * a.g + b.g) / 3, (2 * a.b + b.b) / 3};
            }

            return {(a.r + 2 * b.r) / 3, (a.g + 2 * b.g) / 3, (a.b + 2 * b.b) / 3};
        }

        // Three-color mode uses a midpoint and transparent block.
        if (selector == 2)
        {
            return {(a.r + b.r) / 2, (a.g + b.g) / 2, (a.b + b.b) / 2};
        }

        return {0, 0, 0};
    }

    inline void get_palette(const Block &block, bool is_srgb, Float3 palette[4], bool opaque = false)
    {
        for (uint32_t selector = 0; selector < 4; selector++)
        {
            const Rgb8 color = palette_color_rgb8(block.c0, block.c1, selector, opaque);
            const float r = decode_channel(color.r, is_srgb);
            const float g = decode_channel(color.g, is_srgb);
            const float b = decode_channel(color.b, is_srgb);
            palette[selector] = {r, g, b};
        }
    }

    inline Float3 sample_texel(const Block &block, const Float3 palette[4], uint32_t x, uint32_t y, uint32_t valid_width, uint32_t valid_height)
    {
        // Repeat the last valid texel at image edges
        if (x >= valid_width)
        {
            x = valid_width - 1;
        }

        if (y >= valid_height)
        {
            y = valid_height - 1;
        }

        const uint32_t selector = get_selector(block, x, y);
        return palette[selector];
    }

    // Sum one-bit flags within each 2x2 quadrant
    CDM_INLINE uint32_t count_2x2_regions(uint32_t flags)
    {
        const uint32_t horizontal = (flags & 0x11111111u) + ((flags >> 2) & 0x11111111u);
        return (horizontal & 0x00ff00ffu) + ((horizontal >> 8) & 0x00ff00ffu);
    }

    // Count matching selectors in each quadrant
    CDM_INLINE uint32_t selector_region_counts(uint32_t indices, uint32_t selector)
    {
        uint32_t different = indices ^ (selector * 0x55555555u);
        different |= different >> 1;
        return count_2x2_regions((~different) & 0x55555555u);
    }

    // Compute linear RGB means for the four 2x2 quadrants
    inline void get_quadrant_means(const Block &block, bool is_srgb, Float3 means[4], uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        Float3 palette[4];
        get_palette(block, is_srgb, palette);
        const uint32_t shifts[4] = {0, 4, 16, 20};

        if (valid_width == 4 && valid_height == 4)
        {
            // Weight palette colors by their counts in each quadrant.
            for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
            {
                means[quadrant] = {0.0f, 0.0f, 0.0f};
            }

            for (uint32_t selector = 0; selector < 4; ++selector)
            {
                const uint32_t counts = selector_region_counts(block.indices, selector);

                for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
                {
                    const uint32_t count = (counts >> shifts[quadrant]) & 0xfu;
                    const float weight = static_cast<float>(count) * 0.25f;

                    means[quadrant] += palette[selector] * weight;
                }
            }

            return;
        }

        // Sample edge blocks with coordinates clamped to valid texels.
        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
        {
            const uint32_t x = (quadrant & 1u) * 2u;
            const uint32_t y = (quadrant >> 1u) * 2u;

            const Float3 ll = sample_texel(block, palette, x, y, valid_width, valid_height);
            const Float3 lr = sample_texel(block, palette, x + 1, y, valid_width, valid_height);
            const Float3 rl = sample_texel(block, palette, x, y + 1, valid_width, valid_height);
            const Float3 rr = sample_texel(block, palette, x + 1, y + 1, valid_width, valid_height);

            // Average the four samples in the same order as the prototype.
            means[quadrant] = (ll + lr + rl + rr) * 0.25f;
        }
    }

    // Compute the mean and covariance of 16 child colors.
    inline void compute_child_moments(const Float3 samples[16], Float3 &mean, SymMat3 &covariance)
    {
        constexpr float weight = 1.0f / 16.0f;
        mean = {0.0f, 0.0f, 0.0f};

        for (uint32_t i = 0; i < 16; ++i)
        {
            mean += samples[i];
        }

        mean = mean * weight;
        covariance = {};
        for (uint32_t i = 0; i < 16; ++i)
        {
            const Float3 deviation = samples[i] - mean;
            accumulate_outer(covariance, deviation, weight);
        }
    }

    // Quantize linear RGB and pack it into an RGB565 endpoint 
    inline uint16_t encode_rgb565(const Float3& color, bool is_srgb)
    {
        const Float3 clamped = clamp01(color);
        uint32_t r, g, b;

        if (is_srgb)
        {
            r = quantize_linear_channel<31>(clamped.r);
            g = quantize_linear_channel<63>(clamped.g);
            b = quantize_linear_channel<31>(clamped.b);
        }
        else
        {
            r = static_cast<uint32_t>(clamped.r * 31.0f + 0.5f);
            g = static_cast<uint32_t>(clamped.g * 63.0f + 0.5f);
            b = static_cast<uint32_t>(clamped.b * 31.0f + 0.5f);
        }

        return static_cast<uint16_t>((r << 11) | (g << 5) | b);
    }

    inline Block encode_samples_scalar(const Float3 samples[16], bool is_srgb)
    {
        Float3 mean;
        SymMat3 cov;
        compute_child_moments(samples, mean, cov);

        const bool flat = (cov.rr + cov.gg + cov.bb) < 1e-10f;
        const Float3 axis = compute_principal_axis(cov);

        float projections[16];
        float minimum = 1e30f;  
        float maximum = -1e30f;

        for (uint32_t i = 0; i < 16; ++i)
        {
            const float projection = dot(samples[i] - mean, axis);
            projections[i] = projection;
        
            minimum = std::min(minimum, projection);
            maximum = std::max(maximum, projection);
        }

        // Use the mean for blocks with almost no color variation.
        const Float3 p0 = flat ? mean : mean + axis * minimum;
        const Float3 p1 = flat ? mean : mean + axis * maximum;

        Block block{};
        block.c0 = encode_rgb565(p0, is_srgb);
        block.c1 = encode_rgb565(p1, is_srgb);

        if (block.c0 < block.c1) 
        {
            std::swap(block.c0, block.c1);
        }

        Float3 palette[4];
        get_palette(block, is_srgb, palette, true);

        float palette_projections[4];

        for (uint32_t selector = 0; selector < 4; ++selector)
        {
            palette_projections[selector] = dot(palette[selector] - mean, axis);
        }

        // Choose the nearest palette projection for each sample.
        for (uint32_t i = 0; i < 16; ++i)
        {
            float best_distance = 1e30f;
            uint32_t best_selector = 0;

            for (uint32_t selector = 0; selector < 4; ++selector)
            {
                const float delta = projections[i] - palette_projections[selector];
                const float distance = delta * delta;

                // Strict comparison keeps the lower selector on ties.
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_selector = selector;
                }
            }

            // Place the selector in its row-major texel position.
            block.indices |= best_selector << (2 * texel_map[i]);
        }

        return block;
    }

    // Count valid texels along one axis of a 4x4 block.
    inline uint32_t block_valid_extent(uint32_t size, uint32_t block)
    {
        const uint32_t origin = block * 4;

        if (origin >= size)
        {
            return 0;
        }

        return std::min(4u, size - origin);
    }
    
    // Reuse the first parent when a second block is missing 
    inline uint32_t parent_valid_extent(uint32_t size, uint32_t parent)
    {
        const uint32_t block = size > 4 ? parent : 0;
        return block_valid_extent(size, block);
    }

    inline void repeat_small_samples(Float3 samples[16], uint32_t width, uint32_t height)
    {
        if (width == 4 && height == 4)
        {
            return;
        }

        Float3 original[16];
        std::copy_n(samples, 16, original);

        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = texel_map[i];
            const uint32_t x = (texel & 3u) % width;
            const uint32_t y = (texel >> 2) % height;

            const uint32_t quadrant = (y >> 1) * 2 + (x >> 1);
            const uint32_t local = (y & 1u) * 2 + (x & 1u);
            const uint32_t source = quadrant * 4 + local;

            samples[i] = original[source];
        }
    }

    inline Block generate_child_block_scalar(
        const Block& p00, const Block& p10, const Block& p01, const Block& p11,
        bool is_srgb, Float3 parent_means[4],
        uint32_t valid_width, uint32_t valid_height,
        uint32_t source_width, uint32_t source_height)
    {
        const uint32_t left_width = parent_valid_extent(source_width, 0);
        const uint32_t right_width = parent_valid_extent(source_width, 1);
        const uint32_t top_height = parent_valid_extent(source_height, 0);
        const uint32_t bottom_height = parent_valid_extent(source_height, 1);

        Float3 samples[16];
        get_quadrant_means(p00, is_srgb, samples, left_width, top_height);
        get_quadrant_means(p10, is_srgb, samples + 4, right_width, top_height);
        get_quadrant_means(p01, is_srgb, samples + 8, left_width, bottom_height);
        get_quadrant_means(p11, is_srgb, samples + 12, right_width, bottom_height);

        repeat_small_samples(samples, valid_width, valid_height);

        // Preserve means before encoding for later mip levels.
        for (uint32_t parent = 0; parent < 4; ++parent)
        {
            const uint32_t offset = parent * 4;
            const Float3 ll = samples[offset];
            const Float3 lr = samples[offset + 1];
            const Float3 rl = samples[offset + 2];
            const Float3 rr = samples[offset + 3];

            parent_means[parent] = (ll + lr + rl + rr) * 0.25f;
        }

        return encode_samples_scalar(samples, is_srgb);
    }

    // Encode a block from stored means without decoding BC1 again
    inline Block generate_child_block_from_means_scalar(
        const MeanImage &source, 
        uint32_t block_x, uint32_t block_y,
        bool is_srgb,
        uint32_t valid_width, uint32_t valid_height)
    {
        Float3 samples[16];

        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = texel_map[i];
            const uint32_t x = (texel & 3u) % valid_width;
            const uint32_t y = (texel >> 2) % valid_height;
            
            samples[i] = source.get(block_x * 4 + x, block_y * 4 + y);
        }

        return encode_samples_scalar(samples, is_srgb);
    }

} // namespace bc1