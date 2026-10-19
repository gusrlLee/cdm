#pragma once

#include "bc2.h"
#include "bc4.h"

namespace bc3
{
    struct Block
    {
        bc4::Block alpha;
        bc1::Block rgb;
    };

    static_assert(sizeof(Block) == 16 && offsetof(Block, rgb) == 8);

    // BC3 alpha interpolates integers; BC4 keeps fractional normalized values.
    CDM_INLINE float alpha_palette_value(uint32_t e0, uint32_t e1, uint32_t selector)
    {
        if (selector < 2)
            return static_cast<float>(selector == 0 ? e0 : e1) * (1.0f / 255.0f);
        if (e0 <= e1 && selector >= 6)
            return selector == 6 ? 0.0f : 1.0f;
        const uint32_t divisor = e0 > e1 ? 7u : 5u;
        const uint32_t weight = selector - 1;
        const uint32_t value = ((divisor - weight) * e0 + weight * e1) / divisor;
        return static_cast<float>(value) * (1.0f / 255.0f);
    }

    inline void get_quadrant_means(const Block &block, bool srgb, Float4 means[4], uint32_t width = 4,
                                   uint32_t height = 4)
    {
        Float3 rgb[4];
        bc2::get_rgb_quadrant_means(block.rgb, srgb, rgb, width, height);
        float palette[8];
        for (uint32_t s = 0; s < 8; ++s)
            palette[s] = alpha_palette_value(block.alpha.e0, block.alpha.e1, s);
        for (uint32_t q = 0; q < 4; ++q)
            means[q] = {rgb[q].r, rgb[q].g, rgb[q].b, bc4::get_quadrant_mean(block.alpha, palette, q, width, height)};
    }

    inline Block encode_samples_scalar(const Float4 samples[16], bool srgb)
    {
        float alpha[16];
        for (uint32_t i = 0; i < 16; ++i)
            alpha[i] = samples[i].a;
        float low, high;
        bc4::fit_endpoints_scalar(alpha, low, high);
        bc4::Block encoded{};
        encoded.e0 = bc4::quantize_unorm(high);
        encoded.e1 = bc4::quantize_unorm(low);
        float palette[8];
        for (uint32_t s = 0; s < 8; ++s)
            palette[s] = alpha_palette_value(encoded.e0, encoded.e1, s);
        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = bc4::texel_map[i];
            bc4::set_selector(encoded, texel % 4, texel / 4, bc4::find_best_selector(alpha[i], palette));
        }
        return {encoded, bc2::encode_rgb(samples, srgb)};
    }
}
