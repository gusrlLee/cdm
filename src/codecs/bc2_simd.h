#pragma once

#include "bc2.h"
#include "bc1_simd.h"

namespace bc2
{
    inline void encode_rgb_x4(const Float4 samples[4][16], bool srgb, bc1::Block output[4])
    {
        Float3x4 rgb[16];
        for (uint32_t i = 0; i < 16; ++i)
            rgb[i] = {_mm_setr_ps(samples[0][i].r, samples[1][i].r, samples[2][i].r, samples[3][i].r),
                      _mm_setr_ps(samples[0][i].g, samples[1][i].g, samples[2][i].g, samples[3][i].g),
                      _mm_setr_ps(samples[0][i].b, samples[1][i].b, samples[2][i].b, samples[3][i].b)};
        bc1::encode_samples_x4(rgb, srgb, output);
    }

    inline void encode_samples_x4(const Float4 samples[4][16], bool srgb, Block output[4])
    {
        bc1::Block rgb[4];
        encode_rgb_x4(samples, srgb, rgb);
        for (uint32_t lane = 0; lane < 4; ++lane)
            output[lane] = {encode_alpha(samples[lane]), rgb[lane]};
    }
}
