#pragma once

#include "bc1.h"

namespace bc4
{
// Two 8-bit endpoints followed by sixteen 3-bit selectors, little endian.
struct Block { uint64_t bits; };
static_assert(sizeof(Block) == 8);

template <bool Signed> CDM_INLINE int endpoint(uint32_t byte)
{
    if constexpr (Signed)
    {
        const int value = byte < 128 ? int(byte) : int(byte) - 256;
        return value < -127 ? -127 : value;
    }
    return int(byte);
}

// BC3 alpha reconstructs 8-bit integers. BC4/BC5 interpolate normalized
// endpoints without an extra 8-bit quantization (Direct3D BC4/BC5 policy).
template <bool Signed, bool Integer = false>
CDM_INLINE float palette_value(int e0, int e1, uint32_t selector)
{
    const float scale = Signed ? 1.0f / 127.0f : 1.0f / 255.0f;
    if (selector == 0) return float(e0) * scale;
    if (selector == 1) return float(e1) * scale;
    if (e0 <= e1 && selector >= 6)
        return selector == 6 ? (Signed ? -1.0f : 0.0f) : 1.0f;
    const int divisor = e0 > e1 ? 7 : 5;
    const int weight = int(selector) - 1;
    const int numerator = (divisor - weight) * e0 + weight * e1;
    if constexpr (Integer)
        return float(numerator / divisor) * scale;
    return float(numerator) * (scale / float(divisor));
}

template <bool Signed, bool Integer = false>
CDM_INLINE void palette(const Block &block, float out[8])
{
    const int e0 = endpoint<Signed>(uint32_t(block.bits & 255u));
    const int e1 = endpoint<Signed>(uint32_t((block.bits >> 8) & 255u));
    for (uint32_t s = 0; s < 8; ++s)
        out[s] = palette_value<Signed, Integer>(e0, e1, s);
}

CDM_INLINE float quadrant_mean(const Block &block, const float pal[8], uint32_t q,
                               uint32_t valid_w = 4, uint32_t valid_h = 4)
{
    float sum = 0.0f;
    for (uint32_t dy = 0; dy < 2; ++dy)
        for (uint32_t dx = 0; dx < 2; ++dx)
        {
            uint32_t x = (q & 1u) * 2 + dx, y = (q >> 1) * 2 + dy;
            x = x < valid_w ? x : valid_w - 1;
            y = y < valid_h ? y : valid_h - 1;
            sum += pal[(block.bits >> (16 + 3 * (y * 4 + x))) & 7u];
        }
    return sum * 0.25f;
}

template <bool Signed> CDM_INLINE int quantize(float value)
{
    value = fmaxf(Signed ? -1.0f : 0.0f, fminf(1.0f, value));
    const float scaled = value * (Signed ? 127.0f : 255.0f);
    return int(scaled + (scaled < 0.0f ? -0.5f : 0.5f));
}

CDM_INLINE Block pack_endpoints(int high, int low)
{
    return {uint64_t(uint8_t(high)) | (uint64_t(uint8_t(low)) << 8)};
}

template <bool Signed, bool Integer = false>
CDM_INLINE Block encode_channel(const float samples[16])
{
    // The BC1 principal-axis fit in one dimension is exactly min/max.
    float mean = 0.0f;
    for (uint32_t i = 0; i < 16; ++i) mean += samples[i];
    mean *= 1.0f / 16.0f;
    float low = samples[0], high = samples[0], variance = 0.0f;
    for (uint32_t i = 0; i < 16; ++i)
    {
        low = fminf(low, samples[i]);
        high = fmaxf(high, samples[i]);
        const float delta = samples[i] - mean;
        variance += delta * delta * (1.0f / 16.0f);
    }
    if (variance < bc1::kFlatVarianceEpsilon) low = high = mean;
    Block result = pack_endpoints(quantize<Signed>(high), quantize<Signed>(low));
    float pal[8];
    palette<Signed, Integer>(result, pal);
    for (uint32_t i = 0; i < 16; ++i)
    {
        float best = 1e30f;
        uint32_t selector = 0;
        for (uint32_t s = 0; s < 8; ++s)
        {
            const float delta = samples[i] - pal[s];
            if (delta * delta < best)
            {
                best = delta * delta;
                selector = s;
            }
        }
        result.bits |= uint64_t(selector) << (16 + 3 * texel_index(i));
    }
    return result;
}

#if !defined(__CUDACC__)
template <bool Signed, bool Integer = false>
CDM_INLINE void encode_channel_x4(const v4f samples[16], Block out[4])
{
    v4f mean(0.0f);
    for (uint32_t i = 0; i < 16; ++i) mean += samples[i];
    mean = mean * v4f(1.0f / 16.0f);
    v4f low = samples[0], high = samples[0], variance(0.0f);
    for (uint32_t i = 0; i < 16; ++i)
    {
        low = _mm_min_ps(low.v, samples[i].v);
        high = _mm_max_ps(high.v, samples[i].v);
        const v4f delta = samples[i] - mean;
        variance += delta * delta * v4f(1.0f / 16.0f);
    }
    const __m128 flat = _mm_cmplt_ps(variance.v, _mm_set1_ps(bc1::kFlatVarianceEpsilon));
    low = _mm_blendv_ps(low.v, mean.v, flat);
    high = _mm_blendv_ps(high.v, mean.v, flat);
    alignas(16) float lows[4], highs[4], pal[8][4];
    _mm_store_ps(lows, low.v);
    _mm_store_ps(highs, high.v);
    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        out[lane] = pack_endpoints(quantize<Signed>(highs[lane]), quantize<Signed>(lows[lane]));
        float values[8];
        palette<Signed, Integer>(out[lane], values);
        for (uint32_t s = 0; s < 8; ++s) pal[s][lane] = values[s];
    }
    for (uint32_t i = 0; i < 16; ++i)
    {
        v4f best(1e30f);
        __m128i selected = _mm_setzero_si128();
        for (uint32_t s = 0; s < 8; ++s)
        {
            const v4f delta = samples[i] - v4f(_mm_load_ps(pal[s]));
            const v4f error = delta * delta;
            const __m128 mask = _mm_cmplt_ps(error.v, best.v);
            best = _mm_blendv_ps(best.v, error.v, mask);
            selected = _mm_blendv_epi8(selected, _mm_set1_epi32(int(s)), _mm_castps_si128(mask));
        }
        alignas(16) uint32_t selectors[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(selectors), selected);
        for (uint32_t lane = 0; lane < 4; ++lane)
            out[lane].bits |= uint64_t(selectors[lane]) << (16 + 3 * texel_index(i));
    }
}
#endif

#if defined(__CUDACC__)
template <bool Signed, bool Integer = false>
__device__ __forceinline__ float device_quadrant_mean(const Block &block, uint32_t q,
                                                      uint32_t valid_width = 4, uint32_t valid_height = 4)
{
    float pal[8];
    palette<Signed, Integer>(block, pal);
    return quadrant_mean(block, pal, q, valid_width, valid_height);
}

template <bool Signed, bool Integer = false>
__device__ __forceinline__ void encode_half_warp(float sample, uint32_t lane, Block *output)
{
    const unsigned mask = half_warp_mask();
    const float mean = sum16(sample, mask) * (1.0f / 16.0f);
    const float delta = sample - mean;
    const float variance = sum16(delta * delta, mask) * (1.0f / 16.0f);
    float low = min16(sample, mask), high = max16(sample, mask);
    if (variance < bc1::kFlatVarianceEpsilon) low = high = mean;
    uint32_t endpoints = 0;
    if (lane == 0)
        endpoints = uint32_t(pack_endpoints(quantize<Signed>(high), quantize<Signed>(low)).bits);
    endpoints = __shfl_sync(mask, endpoints, 0, 16);
    const int e0 = endpoint<Signed>(endpoints & 255u), e1 = endpoint<Signed>((endpoints >> 8) & 255u);
    const float owned = lane < 8 ? palette_value<Signed, Integer>(e0, e1, lane) : 0.0f;
    float best = 1e30f;
    uint32_t selector = 0;
    for (uint32_t s = 0; s < 8; ++s)
    {
        const float delta = sample - __shfl_sync(mask, owned, s, 16);
        if (delta * delta < best)
        {
            best = delta * delta;
            selector = s;
        }
    }
    const uint64_t bits = uint64_t(selector) << (16 + 3 * texel_index(lane));
    const uint32_t lo = or16(uint32_t(bits), mask), hi = or16(uint32_t(bits >> 32), mask);
    if (lane == 0) output->bits = uint64_t(endpoints | lo) | (uint64_t(hi) << 32);
}
#endif

template <bool Signed> struct Codec
{
    using Block = bc4::Block;
    using Color = Float3;
    using MeanImage = MeanImageChannels<1>;
#if !defined(__CUDACC__)
    static void quadrant_means(const Block &block, bool, Color out[4],
                               uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        float pal[8];
        palette<Signed>(block, pal);
        for (uint32_t q = 0; q < 4; ++q) out[q] = {quadrant_mean(block, pal, q, valid_width, valid_height), 0, 0};
    }
    static Block encode_samples(const Color samples[16], bool)
    {
        float values[16];
        for (uint32_t i = 0; i < 16; ++i) values[i] = samples[i].r;
        return encode_channel<Signed>(values);
    }
    static void encode_samples_x4(const Color samples[4][16], Block out[4], bool)
    {
        v4f values[16];
        for (uint32_t i = 0; i < 16; ++i)
            values[i] = _mm_set_ps(samples[3][i].r, samples[2][i].r, samples[1][i].r, samples[0][i].r);
        encode_channel_x4<Signed>(values, out);
    }
#else
    template <bool Srgb> static __device__ Color device_quadrant_mean(const Block &block, uint32_t q,
                                                                      uint32_t valid_width = 4, uint32_t valid_height = 4)
    {
        return {bc4::device_quadrant_mean<Signed>(block, q, valid_width, valid_height), 0, 0};
    }
    template <bool Srgb> static __device__ void encode_half_warp(Color sample, uint32_t lane, Block *out)
    {
        bc4::encode_half_warp<Signed>(sample.r, lane, out);
    }
#endif
};
} // namespace bc4
