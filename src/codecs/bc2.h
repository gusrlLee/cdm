#pragma once

#include "bc1.h"

namespace bc2
{
    struct Block
    {
        uint64_t alpha;
        bc1::Block rgb;
    };
    static_assert(sizeof(Block) == 16 && offsetof(Block, rgb) == 8);

    CDM_INLINE uint32_t quantize_alpha(float alpha)
    {
        const float value = alpha < 0.0f ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);
        return static_cast<uint32_t>(value * 15.0f + 0.5f);
    }

    CDM_INLINE float alpha_quadrant_mean(uint64_t alpha, uint32_t quadrant, uint32_t width, uint32_t height)
    {
        uint32_t sum = 0;
        for (uint32_t dy = 0; dy < 2; ++dy)
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                const uint32_t x = (quadrant & 1u) * 2 + dx;
                const uint32_t y = (quadrant >> 1u) * 2 + dy;
                const uint32_t texel = (y < height ? y : height - 1) * 4 + (x < width ? x : width - 1);
                sum += static_cast<uint32_t>((alpha >> (4 * texel)) & 15u);
            }
        return static_cast<float>(sum) * (1.0f / 60.0f);
    }

    // BC2 and BC3 always use the four-color RGB palette, regardless of endpoint order.
    inline void get_rgb_quadrant_means(const bc1::Block &block, bool srgb, Float3 means[4], uint32_t width, uint32_t height)
    {
        Float3 palette[4];
        bc1::get_palette(block, srgb, palette, true);
        for (uint32_t q = 0; q < 4; ++q)
        {
            uint32_t counts[4]{};
            for (uint32_t dy = 0; dy < 2; ++dy)
                for (uint32_t dx = 0; dx < 2; ++dx)
                {
                    const uint32_t x = (q & 1u) * 2 + dx;
                    const uint32_t y = (q >> 1u) * 2 + dy;
                    ++counts[bc1::get_selector(block, std::min(x, width - 1), std::min(y, height - 1))];
                }
            means[q] = {};
            for (uint32_t s = 0; s < 4; ++s)
                means[q] += palette[s] * (counts[s] * 0.25f);
        }
    }

    inline void get_quadrant_means(const Block &block, bool srgb, Float4 means[4], uint32_t width = 4, uint32_t height = 4)
    {
        Float3 rgb[4];
        get_rgb_quadrant_means(block.rgb, srgb, rgb, width, height);
        for (uint32_t q = 0; q < 4; ++q)
            means[q] = {rgb[q].r, rgb[q].g, rgb[q].b, alpha_quadrant_mean(block.alpha, q, width, height)};
    }

    inline bc1::Block encode_rgb(const Float4 samples[16], bool srgb)
    {
        Float3 rgb[16];
        for (uint32_t i = 0; i < 16; ++i)
            rgb[i] = {samples[i].r, samples[i].g, samples[i].b};
        return bc1::encode_samples_scalar(rgb, srgb);
    }

    inline uint64_t encode_alpha(const Float4 samples[16])
    {
        uint64_t result = 0;
        for (uint32_t i = 0; i < 16; ++i)
            result |= static_cast<uint64_t>(quantize_alpha(samples[i].a)) << (4 * bc1::texel_map[i]);
        return result;
    }

    inline Block encode_samples_scalar(const Float4 samples[16], bool srgb)
    {
        return {encode_alpha(samples), encode_rgb(samples, srgb)};
    }
}
