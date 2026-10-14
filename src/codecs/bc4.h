#pragma once

#include "core/mean.h"

#include <cstdint>
#include <algorithm>

namespace bc4
{
    constexpr float scale = 1.0f / 255.0f;
    // Map quadrant-grouped samples to row-major texel positions.
    constexpr uint8_t texel_map[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};

    // Store one 8-byte BC4 block.
    struct Block
    {
        uint8_t e0;
        uint8_t e1;
        uint8_t selectors[6];
    };

    static_assert(sizeof(Block) == 8, "A BC4 block must be 8 bytes");

    // Both x and y must be in [0, 3]
    inline uint32_t get_selector(const Block &block, uint32_t x, uint32_t y)
    {
        const uint32_t texel_index = y * 4 + x;
        const uint32_t bit_offset = texel_index * 3;
        const uint32_t byte_index = bit_offset / 8;
        const uint32_t bit_in_byte = bit_offset % 8;

        uint32_t selector_bits = block.selectors[byte_index];

        // A selector crosses bytes when it starts at bit 6 or 7
        if (bit_in_byte > 5)
        {
            selector_bits |= static_cast<uint32_t>(block.selectors[byte_index + 1]) << 8;
        }

        return (selector_bits >> bit_in_byte) & 7u;
    }

    // Selector must be in [0, 7]
    inline float palette_value_unorm(uint8_t e0, uint8_t e1, uint32_t selector)
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

    inline void get_palette_unorm(const Block &block, float palette[8])
    {
        for (uint32_t selector = 0; selector < 8; ++selector)
        {
            palette[selector] = palette_value_unorm(block.e0, block.e1, selector);
        }
    }

    // Average one 2x2 region using selector counts
    inline float get_quadrant_mean(const Block &block, const float palette[8], uint32_t quadrant, uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        const uint32_t start_x = (quadrant % 2) * 2;
        const uint32_t start_y = (quadrant / 2) * 2;

        uint32_t counts[8] = {};

        for (uint32_t dy = 0; dy < 2; ++dy)
        {
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                uint32_t x = start_x + dx;
                uint32_t y = start_y + dy;

                // Repeat the last valid texel at image edges.
                if (x >= valid_width)
                {
                    x = valid_width - 1;
                }

                if (y >= valid_height)
                {
                    y = valid_height - 1;
                }

                const uint32_t selector = get_selector(block, x, y);
                ++counts[selector];
            }
        }

        float mean = 0.0f;

        // Each of the four texels contributes one quarter.
        for (uint32_t selector = 0; selector < 8; ++selector)
        {
            const float weight = static_cast<float>(counts[selector]) * 0.25f;
            mean += palette[selector] * weight;
        }

        return mean;
    }

    // Compute the means of the four 2x2 regions
    inline void get_quadrant_means_unorm(const Block &block, float means[4], uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        float palette[8];
        get_palette_unorm(block, palette);

        // Reuse the same palette for all four regions
        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
        {
            means[quadrant] = get_quadrant_mean(block, palette, quadrant, valid_width, valid_height);
        }
    }

    // Fit endpoints to 16 child samples
    inline void fit_endpoints_scalar(const float samples[16], float &low, float &high)
    {
        constexpr float weight = 1.0f / 16.0f;
        float mean = 0.0f;

        for (uint32_t i = 0; i < 16; ++i)
        {
            mean += samples[i];
        }

        mean *= weight;

        low = samples[0];
        high = samples[0];
        float variance = 0.0f;

        for (uint32_t i = 0; i < 16; ++i)
        {
            low = std::min(low, samples[i]);
            high = std::max(high, samples[i]);

            const float delta = samples[i] - mean;
            variance += delta * delta * weight;
        }

        if (variance < 1e-10f)
        {
            low = mean;
            high = mean;
        }
    }

    // Covert a finite value to 8-bit UNORM endpoint
    inline uint8_t quantize_unorm(float value)
    {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        // Round to the nearest integer after scaling
        return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
    }

    // Choose the palette value with the smallest error
    inline uint32_t find_best_selector(float sample, const float palette[8])
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

    // Store one selector without changing neighbor selectors
    inline void set_selector(Block &block, uint32_t x, uint32_t y, uint32_t selector)
    {
        const uint32_t texel_index = y * 4 + x;
        const uint32_t bit_offset = texel_index * 3;
        const uint32_t byte_index = bit_offset / 8;
        const uint32_t bit_in_byte = bit_offset % 8;

        const uint32_t mask = 7u << bit_in_byte;
        const uint32_t value = selector << bit_in_byte;

        // Clear the old bits, then write the new selector
        block.selectors[byte_index] = static_cast<uint8_t>((block.selectors[byte_index] & ~mask) | value);

        // Write the remaining bits when the selector crosses bytes.
        if (bit_in_byte > 5)
        {
            block.selectors[byte_index + 1] = static_cast<uint8_t>((block.selectors[byte_index + 1] & ~(mask >> 8)) | (value >> 8));
        }
    }

    // Encode 16 quadrant-grouped samples into one BC4 UNORM block.
    inline Block encode_samples_scalar(const float samples[16])
    {
        float low;
        float high;

        fit_endpoints_scalar(samples, low, high);

        // Start with all selector bits cleared
        Block block{};
        block.e0 = quantize_unorm(high);
        block.e1 = quantize_unorm(low);

        // Build the palette from the stored endpoints
        float palette[8];
        get_palette_unorm(block, palette);

        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t selector = find_best_selector(samples[i], palette);

            // Map the sample to its row-major texel position.
            const uint32_t texel = texel_map[i];
            const uint32_t x = texel % 4;
            const uint32_t y = texel / 4;

            set_selector(block, x, y, selector);
        }

        return block;
    }

    // Repeat valid samples to fill a partial 4x4 block.
    inline void repeat_small_samples(float samples[16], uint32_t valid_width, uint32_t valid_height)
    {
        if (valid_width == 4 && valid_height == 4)
        {
            return;
        }

        // Keep the source unchanged while filling repeated samples
        float original[16];
        std::copy_n(samples, 16, original);

        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = texel_map[i];
            const uint32_t x = (texel % 4) % valid_width;
            const uint32_t y = (texel / 4) % valid_height;

            // Convert texel coordinates back to the grouped sample index.
            const uint32_t quadrant = (y / 2) * 2 + (x / 2);
            const uint32_t local = (y % 2) * 2 + (x % 2);
            const uint32_t source = quadrant * 4 + local;

            samples[i] = original[source];
        }
    }

    // Count valid texels in the first or second parent block.
    inline uint32_t parent_valid_extent(uint32_t size, uint32_t parent)
    {
        // Reuse the first block when the second block is missing
        const uint32_t origin = size > 4 ? parent * 4 : 0;

        if (origin >= size)
        {
            return 0;
        }

        return std::min(4u, size - origin);
    }

    // Generate one child block and preserve means for later mip levels.
    inline Block generate_child_block_scalar(
        const Block &p00, const Block &p10, const Block &p01, const Block &p11,
        float parent_means[4], uint32_t valid_width, uint32_t valid_height, uint32_t source_width, uint32_t source_height)
    {
        const uint32_t left_width = parent_valid_extent(source_width, 0);
        const uint32_t right_width = parent_valid_extent(source_width, 1);
        const uint32_t top_height = parent_valid_extent(source_height, 0);
        const uint32_t bottom_height = parent_valid_extent(source_height, 1);

        // Each parent contributes four quadrant means.
        float samples[16];
        get_quadrant_means_unorm(p00, samples, left_width, top_height);
        get_quadrant_means_unorm(p10, samples + 4, right_width, top_height);
        get_quadrant_means_unorm(p01, samples + 8, left_width, bottom_height);
        get_quadrant_means_unorm(p11, samples + 12, right_width, bottom_height);

        repeat_small_samples(samples, valid_width, valid_height);

        // Preserve the averages before endpoint quantization.
        for (uint32_t parent = 0; parent < 4; ++parent)
        {
            const uint32_t offset = parent * 4;
            const float ll = samples[offset];
            const float lr = samples[offset + 1];
            const float rl = samples[offset + 2];
            const float rr = samples[offset + 3];

            parent_means[parent] = (ll + lr + rl + rr) * 0.25f;
        }

        return encode_samples_scalar(samples);
    }

    // Encode stored means without reading compressed blocks again.
    inline Block generate_child_block_from_means_scalar(const MeanImage &source, uint32_t block_x, uint32_t block_y, uint32_t valid_width, uint32_t valid_height)
    {
        float samples[16];

        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = texel_map[i];

            // Repeat valid samples within a partial block.
            const uint32_t x = (texel % 4) % valid_width;
            const uint32_t y = (texel / 4) % valid_height;

            // BC4 uses only the first channel of the shared mean image.
            samples[i] = source.get(block_x * 4 + x, block_y * 4 + y).r;
        }

        return encode_samples_scalar(samples);
    }

} // namespace bc4