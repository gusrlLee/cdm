#pragma once

#include "bc1.h"

namespace bc2
{
// Explicit alpha nibbles precede an always-four-color BC1 RGB block.
struct Block { uint64_t alpha; Block64 rgb; };
static_assert(sizeof(Block) == 16);
static_assert(offsetof(Block, rgb) == 8);
using Color = Float4;
using MeanImage = MeanImage4;

CDM_INLINE float alpha_quadrant_mean(uint64_t alpha, uint32_t q,
                                     uint32_t valid_width = 4, uint32_t valid_height = 4)
{
    const uint32_t t = (q & 1u) * 2 + (q >> 1) * 8;
    const uint32_t t00 = valid_texel_index(t, valid_width, valid_height);
    const uint32_t t10 = valid_texel_index(t + 1, valid_width, valid_height);
    const uint32_t t01 = valid_texel_index(t + 4, valid_width, valid_height);
    const uint32_t t11 = valid_texel_index(t + 5, valid_width, valid_height);
    const uint32_t sum = uint32_t((alpha >> (4 * t00)) & 15u) +
                         uint32_t((alpha >> (4 * t10)) & 15u) +
                         uint32_t((alpha >> (4 * t01)) & 15u) +
                         uint32_t((alpha >> (4 * t11)) & 15u);
    return float(sum) * (1.0f / 60.0f);
}

CDM_INLINE uint32_t quantize_alpha(float alpha)
{
    return uint32_t(fmaxf(0.0f, fminf(1.0f, alpha)) * 15.0f + 0.5f);
}

#if !defined(__CUDACC__)
CDM_INLINE Block64 encode_rgb(const Color samples[16], bool srgb)
{
    Float3 rgb[16];
    for (uint32_t i = 0; i < 16; ++i) rgb[i] = {samples[i].r, samples[i].g, samples[i].b};
    return bc1::encode_samples_scalar(rgb, srgb);
}

CDM_INLINE void encode_rgb_x4(const Color samples[4][16], Block64 out[4], bool srgb)
{
    Float3x4 rgb[16];
    for (uint32_t i = 0; i < 16; ++i)
        rgb[i] = {_mm_set_ps(samples[3][i].r, samples[2][i].r, samples[1][i].r, samples[0][i].r),
                  _mm_set_ps(samples[3][i].g, samples[2][i].g, samples[1][i].g, samples[0][i].g),
                  _mm_set_ps(samples[3][i].b, samples[2][i].b, samples[1][i].b, samples[0][i].b)};
    if (srgb) bc1::encode_samples_x4<true>(rgb, out);
    else bc1::encode_samples_x4<false>(rgb, out);
}

CDM_INLINE uint64_t encode_alpha(const Color samples[16])
{
    uint64_t alpha = 0;
    for (uint32_t i = 0; i < 16; ++i)
        alpha |= uint64_t(quantize_alpha(samples[i].a)) << (4 * texel_index(i));
    return alpha;
}
#endif

struct Codec
{
    using Block = bc2::Block;
    using Color = bc2::Color;
    using MeanImage = bc2::MeanImage;
#if !defined(__CUDACC__)
    static void quadrant_means(const Block &block, bool srgb, Color out[4],
                               uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        Float3 rgb[4];
        bc1::get_quadrant_means_scalar<true>(block.rgb, srgb, rgb, valid_width, valid_height);
        for (uint32_t q = 0; q < 4; ++q)
            out[q] = {rgb[q].r, rgb[q].g, rgb[q].b, alpha_quadrant_mean(block.alpha, q, valid_width, valid_height)};
    }
    static Block encode_samples(const Color samples[16], bool srgb)
    {
        return {encode_alpha(samples), encode_rgb(samples, srgb)};
    }
    static void encode_samples_x4(const Color samples[4][16], Block out[4], bool srgb)
    {
        Block64 rgb[4];
        encode_rgb_x4(samples, rgb, srgb);
        for (uint32_t lane = 0; lane < 4; ++lane) out[lane] = {encode_alpha(samples[lane]), rgb[lane]};
    }
#else
    template <bool Srgb> static __device__ Color device_quadrant_mean(const Block &block, uint32_t q,
                                                                      uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        const Float3 rgb = bc1::device_quadrant_mean<Srgb, true>(block.rgb, q, valid_width, valid_height);
        return {rgb.r, rgb.g, rgb.b, alpha_quadrant_mean(block.alpha, q, valid_width, valid_height)};
    }
    template <bool Srgb> static __device__ void encode_half_warp(Color sample, uint32_t lane, Block *out)
    {
        bc1::encode_half_warp<Srgb>({sample.r, sample.g, sample.b}, lane, &out->rgb);
        const unsigned mask = half_warp_mask();
        const uint64_t bits = uint64_t(quantize_alpha(sample.a)) << (4 * texel_index(lane));
        const uint32_t low = or16(uint32_t(bits), mask), high = or16(uint32_t(bits >> 32), mask);
        if (lane == 0) out->alpha = uint64_t(low) | (uint64_t(high) << 32);
    }
#endif
};
} // namespace bc2
