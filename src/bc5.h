#pragma once

#include "bc4.h"

namespace bc5
{
// Red then green: independent BC4 channels, with no normal-map assumptions.
struct Block { bc4::Block red, green; };
static_assert(sizeof(Block) == 16);
static_assert(offsetof(Block, green) == 8);

template <bool Signed> struct Codec
{
    using Block = bc5::Block;
    using Color = Float3;
    using MeanImage = MeanImageChannels<2>;
#if !defined(__CUDACC__)
    static void quadrant_means(const Block &block, bool, Color out[4],
                               uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        float red[8], green[8];
        bc4::palette<Signed>(block.red, red);
        bc4::palette<Signed>(block.green, green);
        for (uint32_t q = 0; q < 4; ++q)
            out[q] = {bc4::quadrant_mean(block.red, red, q, valid_width, valid_height),
                      bc4::quadrant_mean(block.green, green, q, valid_width, valid_height), 0};
    }
    static Block encode_samples(const Color samples[16], bool)
    {
        float red[16], green[16];
        for (uint32_t i = 0; i < 16; ++i) { red[i] = samples[i].r; green[i] = samples[i].g; }
        return {bc4::encode_channel<Signed>(red), bc4::encode_channel<Signed>(green)};
    }
    static void encode_samples_x4(const Color samples[4][16], Block out[4], bool)
    {
        v4f red[16], green[16];
        for (uint32_t i = 0; i < 16; ++i)
        {
            red[i] = _mm_set_ps(samples[3][i].r, samples[2][i].r, samples[1][i].r, samples[0][i].r);
            green[i] = _mm_set_ps(samples[3][i].g, samples[2][i].g, samples[1][i].g, samples[0][i].g);
        }
        bc4::Block r[4], g[4];
        bc4::encode_channel_x4<Signed>(red, r);
        bc4::encode_channel_x4<Signed>(green, g);
        for (uint32_t lane = 0; lane < 4; ++lane) out[lane] = {r[lane], g[lane]};
    }
#else
    template <bool Srgb> static __device__ Color device_quadrant_mean(const Block &block, uint32_t q,
                                                                      uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        return {bc4::device_quadrant_mean<Signed>(block.red, q, valid_width, valid_height),
                bc4::device_quadrant_mean<Signed>(block.green, q, valid_width, valid_height), 0};
    }
    template <bool Srgb> static __device__ void encode_half_warp(Color sample, uint32_t lane, Block *out)
    {
        bc4::encode_half_warp<Signed>(sample.r, lane, &out->red);
        bc4::encode_half_warp<Signed>(sample.g, lane, &out->green);
    }
#endif
};
} // namespace bc5
