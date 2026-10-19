#pragma once

#include "bc5.h"
#include "bc4_simd.h"
#include "core/math_simd.h"

namespace bc5
{
    inline void encode_signed_channel_x4(const __m128 samples[16], bc4::Block output[4])
    {
        __m128 low, high;
        bc4::fit_endpoints_x4(samples, low, high);
        float lows[4], highs[4];
        _mm_storeu_ps(lows, low);
        _mm_storeu_ps(highs, high);
        for (uint32_t lane = 0; lane < 4; ++lane)
            output[lane] = {quantize_signed(highs[lane]), quantize_signed(lows[lane]), {}};
        __m128 palette[8];
        for (uint32_t s = 0; s < 8; ++s)
            palette[s] = _mm_setr_ps(signed_palette_value(output[0].e0, output[0].e1, s), signed_palette_value(output[1].e0, output[1].e1, s),
                                     signed_palette_value(output[2].e0, output[2].e1, s), signed_palette_value(output[3].e0, output[3].e1, s));
        for (uint32_t i = 0; i < 16; ++i)
        {
            uint32_t selected[4];
            _mm_storeu_si128(reinterpret_cast<__m128i *>(selected), bc4::find_best_selector_x4(samples[i], palette));
            const uint32_t texel = bc4::texel_map[i];
            for (uint32_t lane = 0; lane < 4; ++lane)
                bc4::set_selector(output[lane], texel % 4, texel / 4, selected[lane]);
        }
    }

    inline void encode_samples_x4(const Float3x4 samples[16], Block output[4], bool is_signed = false)
    {
        __m128 red[16], green[16];
        for (uint32_t i = 0; i < 16; ++i)
        {
            red[i] = samples[i].r;
            green[i] = samples[i].g;
        }
        bc4::Block r[4], g[4];
        if (is_signed)
        {
            encode_signed_channel_x4(red, r);
            encode_signed_channel_x4(green, g);
        }
        else
        {
            bc4::encode_samples_x4(red, r);
            bc4::encode_samples_x4(green, g);
        }
        for (uint32_t lane = 0; lane < 4; ++lane)
            output[lane] = {r[lane], g[lane]};
    }
}
