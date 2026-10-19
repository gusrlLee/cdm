#pragma once

#include "bc4.h"

namespace bc5
{
    struct Block
    {
        bc4::Block red;
        bc4::Block green;
    };

    static_assert(sizeof(Block) == 16 && offsetof(Block, green) == 8);

    CDM_INLINE int signed_endpoint(uint8_t byte)
    {
        const int value = byte < 128 ? byte : static_cast<int>(byte) - 256;
        return value < -127 ? -127 : value;
    }

    CDM_INLINE float signed_palette_value(uint8_t a, uint8_t b, uint32_t selector)
    {
        const int e0 = signed_endpoint(a), e1 = signed_endpoint(b);
        if (selector < 2)
            return static_cast<float>(selector == 0 ? e0 : e1) * (1.0f / 127.0f);
        if (e0 <= e1 && selector >= 6)
            return selector == 6 ? -1.0f : 1.0f;
        const int divisor = e0 > e1 ? 7 : 5;
        const int weight = static_cast<int>(selector) - 1;
        return static_cast<float>((divisor - weight) * e0 + weight * e1) * (1.0f / (127.0f * divisor));
    }

    CDM_INLINE uint8_t quantize_signed(float value)
    {
        const float clamped = value < -1.0f ? -1.0f : (value > 1.0f ? 1.0f : value);
        const float scaled = clamped * 127.0f;
        return static_cast<uint8_t>(static_cast<int>(scaled + (scaled < 0.0f ? -0.5f : 0.5f)));
    }

    inline bc4::Block encode_signed_channel(const float samples[16])
    {
        float low, high;
        bc4::fit_endpoints_scalar(samples, low, high);
        bc4::Block block{};
        block.e0 = quantize_signed(high);
        block.e1 = quantize_signed(low);
        float palette[8];
        for (uint32_t s = 0; s < 8; ++s)
            palette[s] = signed_palette_value(block.e0, block.e1, s);
        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = bc4::texel_map[i];
            bc4::set_selector(block, texel % 4, texel / 4, bc4::find_best_selector(samples[i], palette));
        }
        return block;
    }

    inline void get_quadrant_means(const Block &block, Float3 means[4], uint32_t width = 4, uint32_t height = 4,
                                   bool is_signed = false)
    {
        float red[4], green[4];
        if (is_signed)
        {
            float r[8], g[8];
            for (uint32_t s = 0; s < 8; ++s)
            {
                r[s] = signed_palette_value(block.red.e0, block.red.e1, s);
                g[s] = signed_palette_value(block.green.e0, block.green.e1, s);
            }
            for (uint32_t q = 0; q < 4; ++q)
            {
                red[q] = bc4::get_quadrant_mean(block.red, r, q, width, height);
                green[q] = bc4::get_quadrant_mean(block.green, g, q, width, height);
            }
        }
        else
        {
            bc4::get_quadrant_means_unorm(block.red, red, width, height);
            bc4::get_quadrant_means_unorm(block.green, green, width, height);
        }
        for (uint32_t q = 0; q < 4; ++q)
            means[q] = {red[q], green[q], 0.0f};
    }

    inline Block encode_samples_scalar(const Float3 samples[16], bool is_signed = false)
    {
        float red[16], green[16];
        for (uint32_t i = 0; i < 16; ++i)
        {
            red[i] = samples[i].r;
            green[i] = samples[i].g;
        }
        if (is_signed)
            return {encode_signed_channel(red), encode_signed_channel(green)};
        return {bc4::encode_samples_scalar(red), bc4::encode_samples_scalar(green)};
    }
}
