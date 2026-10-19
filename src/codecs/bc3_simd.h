#pragma once

#include "bc3.h"
#include "bc2_simd.h"
#include "bc4_simd.h"

namespace bc3
{
    inline void encode_samples_x4(const Float4 samples[4][16], bool srgb, Block output[4])
    {
        bc1::Block rgb[4];
        bc2::encode_rgb_x4(samples, srgb, rgb);
        __m128 alpha[16];
        for (uint32_t i = 0; i < 16; ++i)
            alpha[i] = _mm_setr_ps(samples[0][i].a, samples[1][i].a, samples[2][i].a, samples[3][i].a);
        __m128 low, high;
        bc4::fit_endpoints_x4(alpha, low, high);
        uint32_t e0[4], e1[4];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(e0), bc4::quantize_unorm_x4(high));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(e1), bc4::quantize_unorm_x4(low));
        __m128 palette[8];
        for (uint32_t s = 0; s < 8; ++s)
            palette[s] = _mm_setr_ps(alpha_palette_value(e0[0], e1[0], s), alpha_palette_value(e0[1], e1[1], s),
                                     alpha_palette_value(e0[2], e1[2], s), alpha_palette_value(e0[3], e1[3], s));
        for (uint32_t lane = 0; lane < 4; ++lane)
            output[lane] = {{static_cast<uint8_t>(e0[lane]), static_cast<uint8_t>(e1[lane]), {}}, rgb[lane]};
        for (uint32_t i = 0; i < 16; ++i)
        {
            uint32_t selected[4];
            _mm_storeu_si128(reinterpret_cast<__m128i *>(selected), bc4::find_best_selector_x4(alpha[i], palette));
            const uint32_t texel = bc4::texel_map[i];
            for (uint32_t lane = 0; lane < 4; ++lane)
                bc4::set_selector(output[lane].alpha, texel % 4, texel / 4, selected[lane]);
        }
    }
}
