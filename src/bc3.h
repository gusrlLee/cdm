#pragma once

#include "bc2.h"
#include "bc4.h"

namespace bc3
{
struct Block { bc4::Block alpha; Block64 rgb; };
static_assert(sizeof(Block) == 16);
static_assert(offsetof(Block, rgb) == 8);
using Color = Float4;
using MeanImage = MeanImage4;

struct Codec
{
    using Block = bc3::Block;
    using Color = bc3::Color;
    using MeanImage = bc3::MeanImage;
#if !defined(__CUDACC__)
    static void quadrant_means(const Block &block, bool srgb, Color out[4],
                               uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        Float3 rgb[4];
        bc1::get_quadrant_means_scalar<true>(block.rgb, srgb, rgb, valid_width, valid_height);
        float pal[8];
        bc4::palette<false, true>(block.alpha, pal);
        for (uint32_t q = 0; q < 4; ++q)
            out[q] = {rgb[q].r, rgb[q].g, rgb[q].b, bc4::quadrant_mean(block.alpha, pal, q, valid_width, valid_height)};
    }
    static Block encode_samples(const Color samples[16], bool srgb)
    {
        float alpha[16];
        for (uint32_t i = 0; i < 16; ++i) alpha[i] = samples[i].a;
        return {bc4::encode_channel<false, true>(alpha), bc2::encode_rgb(samples, srgb)};
    }
    static void encode_samples_x4(const Color samples[4][16], Block out[4], bool srgb)
    {
        Block64 rgb[4];
        bc2::encode_rgb_x4(samples, rgb, srgb);
        v4f alpha[16];
        for (uint32_t i = 0; i < 16; ++i)
            alpha[i] = _mm_set_ps(samples[3][i].a, samples[2][i].a, samples[1][i].a, samples[0][i].a);
        bc4::Block encoded[4];
        bc4::encode_channel_x4<false, true>(alpha, encoded);
        for (uint32_t lane = 0; lane < 4; ++lane) out[lane] = {encoded[lane], rgb[lane]};
    }
#else
    template <bool Srgb> static __device__ Color device_quadrant_mean(const Block &block, uint32_t q,
                                                                      uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        const Float3 rgb = bc1::device_quadrant_mean<Srgb, true>(block.rgb, q, valid_width, valid_height);
        return {rgb.r, rgb.g, rgb.b, bc4::device_quadrant_mean<false, true>(block.alpha, q, valid_width, valid_height)};
    }
    template <bool Srgb> static __device__ void encode_half_warp(Color sample, uint32_t lane, Block *out)
    {
        bc1::encode_half_warp<Srgb>({sample.r, sample.g, sample.b}, lane, &out->rgb);
        bc4::encode_half_warp<false, true>(sample.a, lane, &out->alpha);
    }
#endif
};
} // namespace bc3
