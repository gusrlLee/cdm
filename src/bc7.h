#pragma once

#include "bc1.h"
#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <array>

namespace bc7
{
constexpr float kFlatVarianceEpsilon = 1e-10f;

template <typename T> CDM_INLINE T flat_variance(const SymMat4T<T> &cov)
{
    return cov.rr + cov.gg + cov.bb + cov.aa;
}

// cov is normalized by the number of samples in this group 
CDM_INLINE float estimate_rgb_line_sse(const SymMat4 &cov, const Float4 &axis, uint32_t sample_count)
{
    const Float4 rgb_axis = {axis.r, axis.g, axis.b, 0.0f};
    const float total = cov.rr + cov.gg + cov.bb;
    const float along_axis = dot(rgb_axis, cov.multiply(rgb_axis));
    const float residual = total - along_axis;
    return sample_count * (residual > 0.0f ? residual : 0.0f);
}

CDM_INLINE float estimate_rgb_index_sse(const Float4 &low, const Float4 &high, uint32_t sample_count, uint32_t index_levels)
{
    const Float4 span = high - low;
    const float span_sq = span.r * span.r + span.g * span.g + span.b * span.b;
    const float  steps = float(index_levels - 1);

    return float(sample_count) * span_sq / (12.0f * steps * steps);
}

// Raw BC7 block. All backends output Modes 1/6/7.
struct Block
{
    uint64_t low, high;
};
static_assert(sizeof(Block) == 16);

using MeanImage = MeanImage4;

#include "bc7_tables.inl"

// ---------------------------------------------------------------------
// Mode 6 bit layout and palette reconstruction
// ---------------------------------------------------------------------

CDM_INLINE bool is_mode6(const Block &b)
{
    return (b.low & 0x7fu) == 0x40u;
}

CDM_INLINE uint64_t get_bits(const Block &b, uint32_t bit, uint32_t count)
{
    if (!count)
        return 0;
    if (bit < 64)
    {
        uint64_t value = b.low >> bit;
        if (bit + count > 64)
            value |= b.high << (64 - bit);
        return count == 64 ? value : value & ((uint64_t(1) << count) - 1);
    }
    return (b.high >> (bit - 64)) & ((uint64_t(1) << count) - 1);
}

struct SymbolicBC7
{
    uint8_t endpoints[6][4];
    uint8_t subset[16];
    uint8_t color_index[16];
    uint8_t alpha_index[16];
    uint8_t mode;
    uint8_t subsets;
    uint8_t color_bits;
    uint8_t alpha_bits;
    uint8_t rotation;
    bool valid;
};

// Parse all BC7 modes without materializing an RGBA texel block.  Endpoint
// expansion and fix-up handling follow the BPTC reference reconstruction.
CDM_INLINE SymbolicBC7 parse_symbolic_block(const Block &block)
{
    constexpr uint8_t color_precision[8] = {4, 6, 5, 7, 5, 7, 7, 5};
    constexpr uint8_t alpha_precision[8] = {0, 0, 0, 0, 6, 8, 7, 5};
    constexpr uint8_t modes_with_pbits = 0xcb;
    SymbolicBC7 out{};
    uint32_t bit = 0;
    for (; out.mode < 8 && get_bits(block, bit++, 1) == 0; ++out.mode)
    {
    }
    if (out.mode >= 8)
        return out;

    out.valid = true;
    out.subsets = (out.mode == 0 || out.mode == 2) ? 3 :
                  (out.mode == 1 || out.mode == 3 || out.mode == 7) ? 2 : 1;
    uint32_t partition = 0;
    if (out.subsets > 1)
    {
        const uint32_t partition_bits = out.mode == 0 ? 4 : 6;
        partition = uint32_t(get_bits(block, bit, partition_bits));
        bit += partition_bits;
    }
    if (out.mode == 4 || out.mode == 5)
    {
        out.rotation = uint8_t(get_bits(block, bit, 2));
        bit += 2;
    }
    uint32_t index_selection = 0;
    if (out.mode == 4)
        index_selection = uint32_t(get_bits(block, bit++, 1));

    const uint32_t endpoint_count = out.subsets * 2;
    for (uint32_t channel = 0; channel < 3; ++channel)
        for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
        {
            out.endpoints[endpoint][channel] = uint8_t(get_bits(block, bit, color_precision[out.mode]));
            bit += color_precision[out.mode];
        }
    if (alpha_precision[out.mode])
        for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
        {
            out.endpoints[endpoint][3] = uint8_t(get_bits(block, bit, alpha_precision[out.mode]));
            bit += alpha_precision[out.mode];
        }

    const bool has_pbits = (modes_with_pbits & (1u << out.mode)) != 0;
    if (out.mode == 0 || out.mode == 1 || out.mode == 3 || out.mode == 6 || out.mode == 7)
    {
        for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
            for (uint32_t channel = 0; channel < 4; ++channel)
                out.endpoints[endpoint][channel] <<= 1;
        if (out.mode == 1)
        {
            const uint32_t p0 = uint32_t(get_bits(block, bit++, 1));
            const uint32_t p1 = uint32_t(get_bits(block, bit++, 1));
            for (uint32_t channel = 0; channel < 3; ++channel)
            {
                out.endpoints[0][channel] |= uint8_t(p0);
                out.endpoints[1][channel] |= uint8_t(p0);
                out.endpoints[2][channel] |= uint8_t(p1);
                out.endpoints[3][channel] |= uint8_t(p1);
            }
        }
        else if (has_pbits)
        {
            for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
            {
                const uint8_t p = uint8_t(get_bits(block, bit++, 1));
                for (uint32_t channel = 0; channel < 4; ++channel)
                    out.endpoints[endpoint][channel] |= p;
            }
        }
    }

    const uint32_t color_precision_with_p = color_precision[out.mode] + uint32_t(has_pbits);
    const uint32_t alpha_precision_with_p = alpha_precision[out.mode] + uint32_t(has_pbits);
    for (uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint)
    {
        for (uint32_t channel = 0; channel < 3; ++channel)
        {
            uint32_t value = uint32_t(out.endpoints[endpoint][channel]) << (8 - color_precision_with_p);
            out.endpoints[endpoint][channel] = uint8_t(value | (value >> color_precision_with_p));
        }
        if (alpha_precision[out.mode])
        {
            uint32_t value = uint32_t(out.endpoints[endpoint][3]) << (8 - alpha_precision_with_p);
            out.endpoints[endpoint][3] = uint8_t(value | (value >> alpha_precision_with_p));
        }
        else
            out.endpoints[endpoint][3] = 255;
    }

    out.color_bits = (out.mode == 0 || out.mode == 1) ? 3 : (out.mode == 6 ? 4 : 2);
    out.alpha_bits = out.mode == 4 ? 3 : (out.mode == 5 ? 2 : 0);
    for (uint32_t texel = 0; texel < 16; ++texel)
    {
        uint8_t partition_entry = uint8_t(texel ? 0 : 128);
        if (out.subsets > 1)
        {
            partition_entry = kPartitionSets[out.subsets - 2][partition][texel >> 2][texel & 3];
        }
        out.subset[texel] = partition_entry & 3;
        const uint32_t count = out.color_bits - ((partition_entry & 0x80) != 0);
        out.color_index[texel] = uint8_t(get_bits(block, bit, count));
        bit += count;
    }
    if (out.alpha_bits)
        for (uint32_t texel = 0; texel < 16; ++texel)
        {
            const uint32_t count = out.alpha_bits - (texel == 0);
            out.alpha_index[texel] = uint8_t(get_bits(block, bit, count));
            bit += count;
        }

    // Normalize the two streams so color_index always drives RGB and
    // alpha_index always drives alpha.
    if (out.alpha_bits && index_selection)
    {
        for (uint32_t texel = 0; texel < 16; ++texel)
        {
            const uint8_t primary = out.color_index[texel];
            out.color_index[texel] = out.alpha_index[texel];
            out.alpha_index[texel] = primary;
        }
        const uint8_t primary_bits = out.color_bits;
        out.color_bits = out.alpha_bits;
        out.alpha_bits = primary_bits;
    }
    return out;
}

CDM_INLINE void put_bits(Block &b, uint32_t bit, uint32_t count, uint64_t value)
{
    if (bit < 64)
    {
        b.low |= value << bit;
        if (bit + count > 64)
            b.high |= value >> (64 - bit);
    }
    else
        b.high |= value << (bit - 64);
}

CDM_INLINE void unpack(const Block &b, uint8_t ep[2][4], uint8_t selector[16])
{
    uint32_t bit = 7;
    for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t e = 0; e < 2; ++e)
        {
            ep[e][c] = uint8_t(get_bits(b, bit, 7));
            bit += 7;
        }
    uint32_t p0 = uint32_t(get_bits(b, 63, 1)), p1 = uint32_t(get_bits(b, 64, 1));
    for (uint32_t c = 0; c < 4; ++c)
    {
        ep[0][c] = uint8_t((ep[0][c] << 1) | p0);
        ep[1][c] = uint8_t((ep[1][c] << 1) | p1);
    }
    bit = 65;
    selector[0] = uint8_t(get_bits(b, bit, 3));
    bit += 3;
    for (uint32_t i = 1; i < 16; ++i)
    {
        selector[i] = uint8_t(get_bits(b, bit, 4));
        bit += 4;
    }
}

CDM_INLINE uint8_t interpolate(uint32_t a, uint32_t b, uint32_t s)
{
    uint32_t w;
    switch (s)
    {
    case 0:
        w = 0;
        break;
    case 1:
        w = 4;
        break;
    case 2:
        w = 9;
        break;
    case 3:
        w = 13;
        break;
    case 4:
        w = 17;
        break;
    case 5:
        w = 21;
        break;
    case 6:
        w = 26;
        break;
    case 7:
        w = 30;
        break;
    case 8:
        w = 34;
        break;
    case 9:
        w = 38;
        break;
    case 10:
        w = 43;
        break;
    case 11:
        w = 47;
        break;
    case 12:
        w = 51;
        break;
    case 13:
        w = 55;
        break;
    case 14:
        w = 60;
        break;
    default:
        w = 64;
        break;
    }
    return uint8_t(((64 - w) * a + w * b + 32) >> 6);
}

CDM_INLINE uint8_t interpolate_with_bits(uint32_t a, uint32_t b, uint32_t selector, uint32_t bits)
{
    if (bits == 2)
    {
        constexpr uint8_t weights[4] = {0, 21, 43, 64};
        return uint8_t(((64 - weights[selector]) * a + weights[selector] * b + 32) >> 6);
    }
    if (bits == 3)
    {
        constexpr uint8_t weights[8] = {0, 9, 18, 27, 37, 46, 55, 64};
        return uint8_t(((64 - weights[selector]) * a + weights[selector] * b + 32) >> 6);
    }
    return interpolate(a, b, selector);
}

CDM_INLINE Block pack_block(const uint8_t ep8[2][4], const uint8_t selectors[16])
{
    Block out = {0x40u, 0};
    uint32_t bit = 7;
    for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t e = 0; e < 2; ++e)
        {
            put_bits(out, bit, 7, ep8[e][c] >> 1);
            bit += 7;
        }
    put_bits(out, 63, 1, ep8[0][0] & 1);
    put_bits(out, 64, 1, ep8[1][0] & 1);
    bit = 65;
    put_bits(out, bit, 3, selectors[0]);
    bit += 3;
    for (uint32_t i = 1; i < 16; ++i)
    {
        put_bits(out, bit, 4, selectors[i]);
        bit += 4;
    }
    return out;
}

template <bool Srgb> CDM_INLINE float channel_to_linear(uint32_t value)
{
#if defined(__CUDA_ARCH__)
    return decode_channel<Srgb>(value);
#elif !defined(__CUDACC__)
    return Srgb ? bc1::c_srgb_to_linear[value] : float(value) * (1.0f / 255.0f);
#else
    const float s = float(value) * (1.0f / 255.0f);
    return Srgb ? (s <= 0.04045f ? s / 12.92f : powf((s + 0.055f) / 1.055f, 2.4f)) : s;
#endif
}

template <bool Srgb> CDM_INLINE Float4 rgba8_to_linear(const uint8_t v[4])
{
    return {channel_to_linear<Srgb>(v[0]), channel_to_linear<Srgb>(v[1]),
            channel_to_linear<Srgb>(v[2]), v[3] * (1.0f / 255.0f)};
}

template <bool Srgb> CDM_INLINE void palette(const uint8_t ep[2][4], Float4 out[16])
{
    for (uint32_t s = 0; s < 16; ++s)
    {
        uint8_t v[4];
        for (uint32_t c = 0; c < 4; ++c)
            v[c] = interpolate(ep[0][c], ep[1][c], s);
        out[s] = rgba8_to_linear<Srgb>(v);
    }
}

#if !defined(__CUDACC__)
template <bool Srgb> CDM_INLINE void quadrant_means(const Block &block, Float4 out[4])
{
    const SymbolicBC7 symbolic = parse_symbolic_block(block);
    for (uint32_t q = 0; q < 4; ++q)
        out[q] = Float4::zero();
    if (!symbolic.valid)
        return;

    struct HistogramEntry
    {
        uint16_t key;
        uint16_t quadrant_counts;
    } histogram[16];
    uint32_t entry_count = 0;

    for (uint32_t texel = 0; texel < 16; ++texel)
    {
        const uint32_t x = texel & 3, y = texel >> 2;
        const uint32_t quadrant = (x >> 1) | ((y >> 1) << 1);
        const uint16_t key = uint16_t(symbolic.subset[texel] | (symbolic.color_index[texel] << 2) |
                                      (symbolic.alpha_index[texel] << 6));
        uint32_t entry = 0;
        while (entry < entry_count && histogram[entry].key != key)
            ++entry;
        if (entry == entry_count)
            histogram[entry_count++] = {key, 0};
        histogram[entry].quadrant_counts += uint16_t(1u << (quadrant * 3));
    }

    for (uint32_t entry = 0; entry < entry_count; ++entry)
    {
        const uint32_t subset = histogram[entry].key & 3;
        const uint32_t color_index = (histogram[entry].key >> 2) & 15;
        const uint32_t alpha_index = (histogram[entry].key >> 6) & 15;
        const uint8_t *a = symbolic.endpoints[subset * 2];
        const uint8_t *b = symbolic.endpoints[subset * 2 + 1];
        uint8_t rgba[4];
        for (uint32_t channel = 0; channel < 3; ++channel)
            rgba[channel] = interpolate_with_bits(a[channel], b[channel], color_index, symbolic.color_bits);
        rgba[3] = symbolic.alpha_bits
                      ? interpolate_with_bits(a[3], b[3], alpha_index, symbolic.alpha_bits)
                      : interpolate_with_bits(a[3], b[3], color_index, symbolic.color_bits);
        if (symbolic.rotation)
            std::swap(rgba[3], rgba[symbolic.rotation - 1]);
        const Float4 value = rgba8_to_linear<Srgb>(rgba);
        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
            out[quadrant] += value * (float((histogram[entry].quadrant_counts >> (quadrant * 3)) & 7) * 0.25f);
    }
}

// ---------------------------------------------------------------------
// Scalar endpoint estimation, quantization, and selector assignment
// ---------------------------------------------------------------------
#endif

CDM_INLINE Float4 clamp4(Float4 v)
{
    auto c = [](float x) { return fmaxf(0.0f, fminf(1.0f, x)); };
    return {c(v.r), c(v.g), c(v.b), c(v.a)};
}

template <bool Srgb> CDM_INLINE uint8_t quantize_component(float target, uint32_t pbit, bool alpha)
{
    uint32_t best = pbit;
    float error = 1e30f;
    for (uint32_t q = 0; q < 128; ++q)
    {
        uint32_t code = (q << 1) | pbit;
        float value = !alpha ? channel_to_linear<Srgb>(code) : code * (1.0f / 255.0f);
        float e = (target - value) * (target - value);
        if (e < error)
        {
            error = e;
            best = code;
        }
    }
    return uint8_t(best);
}

template <bool Srgb> CDM_INLINE void quantize_endpoint(const Float4 &v, uint8_t out[4])
{
    float best = 1e30f;
    for (uint32_t p = 0; p < 2; ++p)
    {
        uint8_t q[4] = {quantize_component<Srgb>(v.r, p, false), quantize_component<Srgb>(v.g, p, false),
                        quantize_component<Srgb>(v.b, p, false), quantize_component<Srgb>(v.a, p, true)};
        Float4 d = rgba8_to_linear<Srgb>(q) - v;
        float e = length_sq(d);
        if (e < best)
        {
            best = e;
            for (int c = 0; c < 4; ++c)
                out[c] = q[c];
        }
    }
}

CDM_INLINE uint16_t desired_partition_mask(const Float4 samples[16], const Float4 &mean, const Float4 &axis,
                                          bool alpha = false)
{
    uint16_t mask = 0;
    for (uint32_t i = 0; i < 16; i++)
    {
        const Float4 delta = samples[i] - mean;
        const float projection = delta.r * axis.r + delta.g * axis.g + delta.b * axis.b +
                                 (alpha ? delta.a * axis.a : 0.0f);
        mask |= uint16_t(projection > 0.0f) << texel_index(i);
    }

    return mask;
}

#if !defined(__CUDA_ARCH__)
static constexpr std::array<uint16_t, 64> kPartitionMasks2 = []{
    std::array<uint16_t, 64> masks{};
    for (uint32_t p = 0; p < 64; p++)
    {
        for (uint32_t t = 0; t < 16; t++)
        {
            masks[p] |= uint16_t(kPartitionSets[0][p][t >> 2][t & 3] & 1u) << t;
        }
    }

    return masks;
}();

#endif

CDM_INLINE uint16_t partition_mask(uint32_t partition)
{
#if defined(__CUDA_ARCH__)
    uint16_t mask = 0;
    for (uint32_t t = 0; t < 16; ++t)
        mask |= uint16_t(kPartitionSets[0][partition][t >> 2][t & 3] & 1u) << t;
    return mask;
#else
    return kPartitionMasks2[partition];
#endif
}

CDM_INLINE uint32_t popcount16(uint32_t x)
{
    x -= (x >> 1) & 0x5555u;
    x = (x & 0x3333u) + ((x >> 2) & 0x3333u);
    x = (x + (x >> 4)) & 0x0f0fu;
    return (x + (x >> 8)) & 0x1fu;
}

CDM_INLINE uint32_t choose_partition(uint16_t desired)
{
    uint32_t best_partition = 0;
    uint32_t best_distance = 17;
    for (uint32_t p = 0; p < 64; ++p)
    {
        const uint32_t distance = popcount16(desired ^ partition_mask(p));
        const uint32_t match = (distance < 16u - distance ? distance : 16u - distance);
        if (match < best_distance)
        {
            best_distance = match;
            best_partition = p;
            if (match == 0)
                break;
        }
    }
    return best_partition;
}

struct PartitionSubsetStats
{
    uint32_t count;
    Float4 mean;
    SymMat4 cov;
    Float4 axis;
    float lo, hi;
};

CDM_INLINE void compute_partition_subset_stats(const Float4 samples[16], uint32_t partition, PartitionSubsetStats out[2])
{
    out[0] = {};
    out[1] = {};
    const uint16_t mask = partition_mask(partition);

    for (uint32_t i = 0; i < 16; ++i)
    {
        const uint32_t subset = (mask >> texel_index(i)) & 1u;
        out[subset].mean += samples[i];
        ++out[subset].count;
    }
    for (uint32_t s = 0; s < 2; ++s)
        out[s].mean = out[s].mean * (1.0f / float(out[s].count));

    for (uint32_t i = 0; i < 16; ++i)
    {
        const uint32_t subset = (mask >> texel_index(i)) & 1u;
        out[subset].cov.accumulate_outer(
            samples[i] - out[subset].mean, 1.0f / float(out[subset].count));
    }

    for (uint32_t s = 0; s < 2; ++s)
    {
        out[s].axis = compute_principal_axis(out[s].cov);
        out[s].lo = 1e30f;
        out[s].hi = -1e30f;
    }

    for (uint32_t i = 0; i < 16; ++i)
    {
        const uint32_t subset = (mask >> texel_index(i)) & 1u;
        const float p = dot(samples[i] - out[subset].mean, out[subset].axis);
        out[subset].lo = fminf(out[subset].lo, p);
        out[subset].hi = fmaxf(out[subset].hi, p);
    }
}

CDM_INLINE float estimate_rgb_endpoint_sse(uint32_t sample_count, uint32_t endpoint_levels, uint32_t index_levels)
{
    const float step = 1.0f / float(endpoint_levels - 1);
    const float levels = float(index_levels);
    const float endpoint_weight = (2.0f * levels - 1.0f) / (3.0f * (levels - 1.0f));

    return float(sample_count) * 3.0f * step * step * endpoint_weight / 12.0f;
}

CDM_INLINE float estimate_mode1_sse(const PartitionSubsetStats stats[2])
{
    float error = 0.0f;
    for (uint32_t s = 0; s < 2; ++s)
    {
        const Float4 low = clamp4(stats[s].mean + stats[s].axis * stats[s].lo);
        const Float4 high = clamp4(stats[s].mean + stats[s].axis * stats[s].hi);

        error += estimate_rgb_line_sse(stats[s].cov, stats[s].axis, stats[s].count);
        error += estimate_rgb_index_sse(low, high, stats[s].count, 8);
        error += estimate_rgb_endpoint_sse(stats[s].count, 64, 8);
    }

    return error;
}

CDM_INLINE float estimate_mode6_sse(const SymMat4 &cov, const Float4 &axis, const Float4 &low, const Float4 &high)
{
    return estimate_rgb_line_sse(cov, axis, 16) + estimate_rgb_index_sse(low, high, 16, 16) + estimate_rgb_endpoint_sse(16, 128, 16);
}

// ep6 order: subset 0 low/high, subset 1 low/high.
// Fix-up texels must have selectors below 4.
CDM_INLINE Block pack_mode1(uint32_t partition, const uint8_t ep6[4][3], const uint8_t pbit[2], const uint8_t selector[16])
{
    Block out = {0x2u, 0}; // BC7 mode 1: binary prefix 10
    uint32_t bit = 2;

    put_bits(out, bit, 6, partition);
    bit += 6;

    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t e = 0; e < 4; ++e)
        {
            put_bits(out, bit, 6, ep6[e][c]);
            bit += 6;
        }

    put_bits(out, bit++, 1, pbit[0]);
    put_bits(out, bit++, 1, pbit[1]);

    for (uint32_t t = 0; t < 16; ++t)
    {
        const uint8_t entry = kPartitionSets[0][partition][t >> 2][t & 3];
        const uint32_t count = 3u - uint32_t((entry & 0x80u) != 0);
        put_bits(out, bit, count, selector[t]);
        bit += count;
    }
    return out;
}

CDM_INLINE void swap_bytes(uint8_t &a, uint8_t &b)
{
    const uint8_t value = a;
    a = b;
    b = value;
}

CDM_INLINE void fix_mode1_anchors(uint32_t partition, uint8_t ep6[4][3], uint8_t selector[16])
{
    const uint16_t mask = partition_mask(partition);

    for (uint32_t t = 0; t < 16; ++t)
    {
        const uint8_t entry = kPartitionSets[0][partition][t >> 2][t & 3];
        if (!(entry & 0x80u) || selector[t] < 4)
            continue;

        const uint32_t subset = entry & 1u;
        for (uint32_t c = 0; c < 3; ++c)
            swap_bytes(ep6[subset * 2][c], ep6[subset * 2 + 1][c]);

        for (uint32_t q = 0; q < 16; ++q)
            if (((mask >> q) & 1u) == subset)
                selector[q] = uint8_t(7 - selector[q]);
    }
}

template <bool Srgb>
CDM_INLINE float mode1_endpoint_value(uint32_t q, uint32_t pbit)
{
    const uint32_t code7 = (q << 1) | pbit;
    const uint32_t code8 = (code7 << 1) | (code7 >> 6);
    return channel_to_linear<Srgb>(code8);
}

template <bool Srgb>
CDM_INLINE uint8_t quantize_mode1_component(float target, uint32_t pbit)
{
    uint32_t left = 0, right = 64;
    while (left < right)
    {
        const uint32_t mid = (left + right) >> 1;
        if (mode1_endpoint_value<Srgb>(mid, pbit) < target)
            left = mid + 1;
        else
            right = mid;
    }

    uint32_t q = left < 64 ? left : 63;
    if (q > 0)
    {
        const float upper = mode1_endpoint_value<Srgb>(q, pbit) - target;
        const float lower = mode1_endpoint_value<Srgb>(q - 1, pbit) - target;
        if (lower * lower < upper * upper)
            --q;
    }
    return uint8_t(q);
}

template <bool Srgb>
CDM_INLINE void quantize_mode1_subset(const Float4 &low, const Float4 &high, uint8_t ep6[2][3], uint8_t &pbit)
{
    const float target[2][3] = {
        {low.r, low.g, low.b},
        {high.r, high.g, high.b}
    };

    float best_error = FLT_MAX;
    for (uint32_t p = 0; p < 2; ++p)
    {
        uint8_t candidate[2][3];
        float error = 0.0f;

        for (uint32_t e = 0; e < 2; ++e)
        {
            for (uint32_t c = 0; c < 3; ++c)
            {
                candidate[e][c] =
                    quantize_mode1_component<Srgb>(target[e][c], p);
                const float value =
                    mode1_endpoint_value<Srgb>(candidate[e][c], p);
                const float delta = value - target[e][c];
                error += delta * delta;
            }
        }

        if (error < best_error)
        {
            best_error = error;
            pbit = uint8_t(p);
            for (uint32_t e = 0; e < 2; ++e)
                for (uint32_t c = 0; c < 3; ++c)
                    ep6[e][c] = candidate[e][c];
        }
    }
}

template <bool Srgb>
CDM_INLINE void quantize_mode1_endpoints(const PartitionSubsetStats stats[2], uint8_t ep6[4][3], uint8_t pbit[2])
{
    for (uint32_t s = 0; s < 2; ++s)
    {
        const Float4 low = clamp4(stats[s].mean + stats[s].axis * stats[s].lo);
        const Float4 high = clamp4(stats[s].mean + stats[s].axis * stats[s].hi);

        quantize_mode1_subset<Srgb>(low, high, ep6 + s * 2, pbit[s]);
    }
}

template <bool Srgb>
CDM_INLINE void assign_mode1_selectors(
    const Float4 samples[16], uint32_t partition,
    const PartitionSubsetStats stats[2],
    const uint8_t ep6[4][3], const uint8_t pbit[2],
    uint8_t selector[16])
{
    uint8_t ep8[4][3];
    for (uint32_t e = 0; e < 4; ++e)
    {
        for (uint32_t c = 0; c < 3; ++c)
        {
            const uint32_t code7 = (uint32_t(ep6[e][c]) << 1) | pbit[e >> 1];
            ep8[e][c] = uint8_t((code7 << 1) | (code7 >> 6));
        }
    }

    float projection[2][8];
    for (uint32_t s = 0; s < 2; ++s)
    {
        for (uint32_t index = 0; index < 8; ++index)
        {
            uint8_t rgba[4] = {0, 0, 0, 255};
            for (uint32_t c = 0; c < 3; ++c)
                rgba[c] = interpolate_with_bits(ep8[s * 2][c], ep8[s * 2 + 1][c], index, 3);

            const Float4 color = rgba8_to_linear<Srgb>(rgba);
            projection[s][index] = dot(color - stats[s].mean, stats[s].axis);
        }
    }

    const uint16_t mask = partition_mask(partition);
    for (uint32_t i = 0; i < 16; ++i)
    {
        const uint32_t texel = texel_index(i);
        const uint32_t subset = (mask >> texel) & 1u;
        const float target = dot(samples[i] - stats[subset].mean, stats[subset].axis);

        float best_error = FLT_MAX;
        uint32_t best = 0;
        for (uint32_t index = 0; index < 8; ++index)
        {
            const float delta = target - projection[subset][index];
            const float error = delta * delta;
            if (error < best_error)
            {
                best_error = error;
                best = index;
            }
        }
        selector[texel] = uint8_t(best);
    }
}

template <bool Srgb>
CDM_INLINE Block encode_mode1_samples(const Float4 samples[16], uint32_t partition, const PartitionSubsetStats stats[2])
{
    uint8_t ep6[4][3];
    uint8_t pbit[2];
    uint8_t selector[16];

    quantize_mode1_endpoints<Srgb>(stats, ep6, pbit);
    assign_mode1_selectors<Srgb>(samples, partition, stats, ep6, pbit, selector);
    fix_mode1_anchors(partition, ep6, selector);
    return pack_mode1(partition, ep6, pbit, selector);
}


CDM_INLINE float estimate_rgba_sse(const SymMat4 &cov, const Float4 &axis,
                                   const Float4 &low, const Float4 &high, uint32_t count,
                                   uint32_t endpoint_levels, uint32_t index_levels)
{
    const float residual = flat_variance(cov) - dot(axis, cov.multiply(axis));
    const float steps = float(index_levels - 1);
    return float(count) * fmaxf(0.0f, residual) +
           float(count) * length_sq(high - low) / (12.0f * steps * steps) +
           estimate_rgb_endpoint_sse(count, endpoint_levels, index_levels) * (4.0f / 3.0f);
}

template <bool Srgb>
CDM_INLINE float mode7_endpoint_value(uint32_t q, uint32_t pbit, bool alpha)
{
    const uint32_t code6 = (q << 1) | pbit;
    const uint32_t code8 = (code6 << 2) | (code6 >> 4);
    return !alpha ? channel_to_linear<Srgb>(code8) : float(code8) * (1.0f / 255.0f);
}

template <bool Srgb>
CDM_INLINE void quantize_mode7_endpoints(const PartitionSubsetStats stats[2],
                                        uint8_t ep5[4][4], uint8_t ep8[4][4], uint8_t pbit[4])
{
    for (uint32_t e = 0; e < 4; ++e)
    {
        const PartitionSubsetStats &s = stats[e >> 1];
        const Float4 endpoint = clamp4(s.mean + s.axis * ((e & 1u) ? s.hi : s.lo));
        const float target[4] = {endpoint.r, endpoint.g, endpoint.b, endpoint.a};
        float best_error = FLT_MAX;
        for (uint32_t p = 0; p < 2; ++p)
        {
            uint8_t candidate[4];
            float error = 0.0f;
            for (uint32_t c = 0; c < 4; ++c)
            {
                float best_component_error = FLT_MAX;
                for (uint32_t q = 0; q < 32; ++q)
                {
                    const float delta = target[c] - mode7_endpoint_value<Srgb>(q, p, c == 3);
                    if (delta * delta < best_component_error)
                    {
                        best_component_error = delta * delta;
                        candidate[c] = uint8_t(q);
                    }
                }
                error += best_component_error;
            }
            if (error < best_error)
            {
                best_error = error;
                pbit[e] = uint8_t(p);
                for (uint32_t c = 0; c < 4; ++c)
                    ep5[e][c] = candidate[c];
            }
        }
        for (uint32_t c = 0; c < 4; ++c)
        {
            const uint32_t code6 = (uint32_t(ep5[e][c]) << 1) | pbit[e];
            ep8[e][c] = uint8_t((code6 << 2) | (code6 >> 4));
        }
    }
}

CDM_INLINE void fix_mode7_anchors(uint32_t partition, uint8_t ep5[4][4], uint8_t pbit[4], uint8_t selector[16])
{
    const uint16_t mask = partition_mask(partition);
    for (uint32_t t = 0; t < 16; ++t)
    {
        const uint8_t entry = kPartitionSets[0][partition][t >> 2][t & 3];
        if (!(entry & 0x80u) || selector[t] < 2)
            continue;
        const uint32_t subset = entry & 1u;
        for (uint32_t c = 0; c < 4; ++c)
            swap_bytes(ep5[subset * 2][c], ep5[subset * 2 + 1][c]);
        swap_bytes(pbit[subset * 2], pbit[subset * 2 + 1]);
        for (uint32_t q = 0; q < 16; ++q)
            if (((mask >> q) & 1u) == subset)
                selector[q] = uint8_t(3 - selector[q]);
    }
}

CDM_INLINE Block pack_mode7(uint32_t partition, const uint8_t ep5[4][4],
                            const uint8_t pbit[4], const uint8_t selector[16])
{
    Block out = {0x80u, 0};
    uint32_t bit = 8;
    put_bits(out, bit, 6, partition);
    bit += 6;
    for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t e = 0; e < 4; ++e)
        {
            put_bits(out, bit, 5, ep5[e][c]);
            bit += 5;
        }
    for (uint32_t e = 0; e < 4; ++e)
        put_bits(out, bit++, 1, pbit[e]);
    for (uint32_t t = 0; t < 16; ++t)
    {
        const uint8_t entry = kPartitionSets[0][partition][t >> 2][t & 3];
        const uint32_t count = 2u - uint32_t((entry & 0x80u) != 0);
        put_bits(out, bit, count, selector[t]);
        bit += count;
    }
    return out;
}

template <bool Srgb>
CDM_INLINE Block encode_mode7_samples(const Float4 samples[16], uint32_t partition,
                                      const PartitionSubsetStats stats[2])
{
    uint8_t ep5[4][4], ep8[4][4], pbit[4];
    quantize_mode7_endpoints<Srgb>(stats, ep5, ep8, pbit);

    float projection[2][4];
    for (uint32_t s = 0; s < 2; ++s)
        for (uint32_t index = 0; index < 4; ++index)
        {
            uint8_t rgba[4];
            for (uint32_t c = 0; c < 4; ++c)
                rgba[c] = interpolate_with_bits(ep8[s * 2][c], ep8[s * 2 + 1][c], index, 2);
            projection[s][index] = dot(rgba8_to_linear<Srgb>(rgba) - stats[s].mean, stats[s].axis);
        }

    uint8_t selector[16];
    const uint16_t mask = partition_mask(partition);
    for (uint32_t i = 0; i < 16; ++i)
    {
        const uint32_t texel = texel_index(i), subset = (mask >> texel) & 1u;
        const float target = dot(samples[i] - stats[subset].mean, stats[subset].axis);
        float best_error = FLT_MAX;
        for (uint32_t index = 0; index < 4; ++index)
        {
            const float delta = target - projection[subset][index];
            if (delta * delta < best_error)
            {
                best_error = delta * delta;
                selector[texel] = uint8_t(index);
            }
        }
    }

    fix_mode7_anchors(partition, ep5, pbit, selector);
    return pack_mode7(partition, ep5, pbit, selector);
}

CDM_INLINE uint32_t choose_partitioned_mode(const PartitionSubsetStats stats[2], bool has_alpha,
                                           const SymMat4 &cov, const Float4 &axis,
                                           const Float4 &low, const Float4 &high)
{
    if (!has_alpha)
        return estimate_mode1_sse(stats) < estimate_mode6_sse(cov, axis, low, high) ? 1u : 6u;
    float error = 0.0f;
    for (uint32_t s = 0; s < 2; ++s)
        error += estimate_rgba_sse(stats[s].cov, stats[s].axis,
                                  clamp4(stats[s].mean + stats[s].axis * stats[s].lo),
                                  clamp4(stats[s].mean + stats[s].axis * stats[s].hi), stats[s].count, 32, 4);
    return error < estimate_rgba_sse(cov, axis, low, high, 16, 128, 16) ? 7u : 6u;
}

template <bool Srgb>
CDM_INLINE bool encode_partitioned_samples(const Float4 samples[16], const Float4 &mean,
                                          const SymMat4 &cov, const Float4 &axis,
                                          const Float4 &low, const Float4 &high, Block &out)
{
    if (flat_variance(cov) < kFlatVarianceEpsilon)
        return false;
    bool has_alpha = false;
    for (uint32_t i = 0; i < 16; ++i)
        has_alpha |= samples[i].a < 1.0f;
    const uint32_t partition = choose_partition(desired_partition_mask(samples, mean, axis, has_alpha));
    PartitionSubsetStats stats[2];
    compute_partition_subset_stats(samples, partition, stats);
    const uint32_t mode = choose_partitioned_mode(stats, has_alpha, cov, axis, low, high);
    if (mode == 6)
        return false;
    out = mode == 1 ? encode_mode1_samples<Srgb>(samples, partition, stats)
                    : encode_mode7_samples<Srgb>(samples, partition, stats);
    return true;
}

// Mode 6 endpoint fitting and selector assignment are shared by scalar and SIMD.
template <bool Srgb>
CDM_INLINE Block encode_with_endpoints(const Float4 samples[16], Float4 p0, Float4 p1, Float4 mean, Float4 axis)
{
    uint8_t ep[2][4];
    quantize_endpoint<Srgb>(clamp4(p0), ep[0]);
    quantize_endpoint<Srgb>(clamp4(p1), ep[1]);
    Float4 pal[16];
    palette<Srgb>(ep, pal);
    float palette_projection[16];
    for (uint32_t s = 0; s < 16; ++s)
        palette_projection[s] = dot(pal[s] - mean, axis);
    uint8_t selector[16] = {};
    for (uint32_t i = 0; i < 16; ++i)
    {
        uint32_t best = 0;
        float error = 1e30f;
        const float sample_projection = dot(samples[i] - mean, axis);
        for (uint32_t s = 0; s < 16; ++s)
        {
            const float delta = sample_projection - palette_projection[s];
            const float e = delta * delta;
            if (e < error)
            {
                error = e;
                best = s;
            }
        }
        selector[texel_index(i)] = uint8_t(best);
    }

    if (selector[0] >= 8)
    {
        for (int c = 0; c < 4; ++c)
            swap_bytes(ep[0][c], ep[1][c]);
            
        for (auto &s : selector)
            s = uint8_t(15 - s);
    }

    return pack_block(ep, selector);
}

template <bool Srgb> CDM_INLINE Block encode_samples(const Float4 samples[16])
{
    Float4 mean = Float4::zero();
    for (int i = 0; i < 16; ++i)
        mean += samples[i];
    
    mean = mean * (1.0f / 16.0f);
    SymMat4 cov = SymMat4::zero();
    for (int i = 0; i < 16; ++i)
        cov.accumulate_outer(samples[i] - mean, 1.0f / 16.0f);
    
    const bool flat = flat_variance(cov) < kFlatVarianceEpsilon;
    Float4 axis = compute_principal_axis(cov);
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < 16; ++i)
    {
        float p = dot(samples[i] - mean, axis);
        lo = fminf(lo, p);
        hi = fmaxf(hi, p);
    }
    
    Float4 endpoints[2] = {mean, mean};
    if (!flat)
    {
        endpoints[0] = mean + axis * lo;
        endpoints[1] = mean + axis * hi;
    }
    
    endpoints[0] = clamp4(endpoints[0]);
    endpoints[1] = clamp4(endpoints[1]);
    
    Block partitioned;
    if (encode_partitioned_samples<Srgb>(samples, mean, cov, axis, endpoints[0], endpoints[1], partitioned))
        return partitioned;
    return encode_with_endpoints<Srgb>(samples, endpoints[0], endpoints[1], mean, axis);
}

#if !defined(__CUDACC__)

CDM_INLINE Block encode_samples(const Float4 s[16], bool srgb)
{
    return srgb ? encode_samples<true>(s) : encode_samples<false>(s);
}

CDM_INLINE Block generate_child(const Block &p00, const Block &p10, const Block &p01, const Block &p11, bool srgb,
                                Float4 parent_means[4], uint32_t valid_width, uint32_t valid_height)
{
    Float4 s[16];
    if (srgb)
    {
        quadrant_means<true>(p00, s);
        quadrant_means<true>(p10, s + 4);
        quadrant_means<true>(p01, s + 8);
        quadrant_means<true>(p11, s + 12);
    }
    else
    {
        quadrant_means<false>(p00, s);
        quadrant_means<false>(p10, s + 4);
        quadrant_means<false>(p01, s + 8);
        quadrant_means<false>(p11, s + 12);
    }
    for (int p = 0; p < 4; ++p)
        parent_means[p] = (s[p * 4] + s[p * 4 + 1] + s[p * 4 + 2] + s[p * 4 + 3]) * 0.25f;
    if (valid_width < 4 || valid_height < 4)
    {
        Float4 copy[16];
        for (int i = 0; i < 16; ++i)
            copy[i] = s[i];
        for (uint32_t i = 0; i < 16; ++i)
            s[i] = copy[repeat_small_sample(i, valid_width, valid_height)];
    }
    return encode_samples(s, srgb);
}

CDM_INLINE Block generate_from_means(const MeanImage &source, uint32_t bx, uint32_t by, bool srgb, uint32_t vw,
                                     uint32_t vh)
{
    Float4 s[16];
    for (uint32_t i = 0; i < 16; ++i)
    {
        uint32_t t = texel_index(i);
        s[i] =
            source.get(repeat_small_coordinate(bx * 4 + (t & 3), vw), repeat_small_coordinate(by * 4 + (t >> 2), vh));
    }
    return encode_samples(s, srgb);
}

// ---------------------------------------------------------------------
// Four-block SSE path
// ---------------------------------------------------------------------

using Float4x4 = Vec4T<v4f>;
using SymMat4x4 = SymMat4T<v4f>;

CDM_INLINE Float4x4 principal_axis_x4(const SymMat4x4 &cov)
{
    const v4f matrix_norm_sq = cov.rr * cov.rr + cov.gg * cov.gg + cov.bb * cov.bb + cov.aa * cov.aa +
                               (cov.rg * cov.rg + cov.rb * cov.rb + cov.ra * cov.ra + cov.gb * cov.gb +
                                cov.ga * cov.ga + cov.ba * cov.ba) *
                                   v4f(2.0f);
    const v4f threshold = matrix_norm_sq * v4f(1e-6f);
    Float4x4 next = cov.multiply({v4f(0.5f), v4f(0.5f), v4f(0.5f), v4f(0.5f)});
    v4f lsq = length_sq(next);
    if (_mm_movemask_ps(_mm_cmple_ps(lsq.v, threshold.v)) == 0)
        return next * v4f(_mm_rsqrt_ps(lsq.v));
    const v4f h(0.70710678f), nh(-0.70710678f), zero(0.0f);

    auto use_if_degenerate = [&](const Float4x4 &candidate) {
        const __m128 mask = _mm_cmple_ps(lsq.v, threshold.v);
        const v4f candidate_lsq = length_sq(candidate);
        next.r = _mm_blendv_ps(next.r.v, candidate.r.v, mask);
        next.g = _mm_blendv_ps(next.g.v, candidate.g.v, mask);
        next.b = _mm_blendv_ps(next.b.v, candidate.b.v, mask);
        next.a = _mm_blendv_ps(next.a.v, candidate.a.v, mask);
        lsq = _mm_blendv_ps(lsq.v, candidate_lsq.v, mask);
    };
    use_if_degenerate(cov.multiply({h, nh, zero, zero}));
    use_if_degenerate(cov.multiply({zero, h, nh, zero}));
    use_if_degenerate(cov.multiply({zero, zero, h, nh}));

    const __m128 flat = _mm_cmple_ps(lsq.v, threshold.v);
    const v4f inverse_length = _mm_rsqrt_ps(_mm_blendv_ps(lsq.v, _mm_set1_ps(1.0f), flat));
    Float4x4 result = next * inverse_length;
    result.r = _mm_blendv_ps(result.r.v, _mm_set1_ps(1.0f), flat);
    result.g = _mm_andnot_ps(flat, result.g.v);
    result.b = _mm_andnot_ps(flat, result.b.v);
    result.a = _mm_andnot_ps(flat, result.a.v);
    return result;
}

CDM_INLINE Float4 extract_lane(const Float4x4 &v, uint32_t lane)
{
    alignas(16) float values[4][4];
    _mm_store_ps(values[0], v.r.v);
    _mm_store_ps(values[1], v.g.v);
    _mm_store_ps(values[2], v.b.v);
    _mm_store_ps(values[3], v.a.v);
    return {values[0][lane], values[1][lane], values[2][lane], values[3][lane]};
}

CDM_INLINE SymMat4 extract_lane(const SymMat4x4 &v, uint32_t lane)
{
    alignas(16) float values[10][4];
    const v4f components[10] = {v.rr, v.gg, v.bb, v.aa, v.rg, v.rb, v.ra, v.gb, v.ga, v.ba};
    for (uint32_t c = 0; c < 10; ++c)
        _mm_store_ps(values[c], components[c].v);
    return {values[0][lane], values[1][lane], values[2][lane], values[3][lane], values[4][lane],
            values[5][lane], values[6][lane], values[7][lane], values[8][lane], values[9][lane]};
}

CDM_INLINE void compute_partition_subset_stats_x4(const Float4x4 samples[16], const uint32_t partition[4],
                                                 PartitionSubsetStats out[4][2])
{
    const uint16_t masks[4] = {partition_mask(partition[0]), partition_mask(partition[1]),
                               partition_mask(partition[2]), partition_mask(partition[3])};
    for (uint32_t subset = 0; subset < 2; ++subset)
    {
        v4f weights[16], count(0.0f);
        Float4x4 mean = Float4x4::zero();
        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t t = texel_index(i);
            weights[i] = _mm_set_ps(float(((masks[3] >> t) & 1u) == subset),
                                    float(((masks[2] >> t) & 1u) == subset),
                                    float(((masks[1] >> t) & 1u) == subset),
                                    float(((masks[0] >> t) & 1u) == subset));
            mean += samples[i] * weights[i];
            count = count + weights[i];
        }
        const v4f inverse_count = _mm_div_ps(_mm_set1_ps(1.0f), count.v);
        mean = mean * inverse_count;
        SymMat4x4 cov = SymMat4x4::zero();
        for (uint32_t i = 0; i < 16; ++i)
            cov.accumulate_outer(samples[i] - mean, weights[i] * inverse_count);
        const Float4x4 axis = principal_axis_x4(cov);
        v4f lo(FLT_MAX), hi(-FLT_MAX);
        for (uint32_t i = 0; i < 16; ++i)
        {
            const v4f projection = dot(samples[i] - mean, axis);
            const __m128 included = _mm_cmpgt_ps(weights[i].v, _mm_setzero_ps());
            lo = _mm_min_ps(lo.v, _mm_blendv_ps(_mm_set1_ps(FLT_MAX), projection.v, included));
            hi = _mm_max_ps(hi.v, _mm_blendv_ps(_mm_set1_ps(-FLT_MAX), projection.v, included));
        }
        alignas(16) float counts[4], lows[4], highs[4];
        _mm_store_ps(counts, count.v);
        _mm_store_ps(lows, lo.v);
        _mm_store_ps(highs, hi.v);
        for (uint32_t lane = 0; lane < 4; ++lane)
            out[lane][subset] = {uint32_t(counts[lane]), extract_lane(mean, lane), extract_lane(cov, lane),
                                 extract_lane(axis, lane), lows[lane], highs[lane]};
    }
}

template <bool Srgb> CDM_INLINE void encode_samples_x4(const Float4 scalar[4][16], Block out[4])
{
    Float4x4 samples[16];
    for (int i = 0; i < 16; ++i)
    {
        samples[i].r = _mm_set_ps(scalar[3][i].r, scalar[2][i].r, scalar[1][i].r, scalar[0][i].r);
        samples[i].g = _mm_set_ps(scalar[3][i].g, scalar[2][i].g, scalar[1][i].g, scalar[0][i].g);
        samples[i].b = _mm_set_ps(scalar[3][i].b, scalar[2][i].b, scalar[1][i].b, scalar[0][i].b);
        samples[i].a = _mm_set_ps(scalar[3][i].a, scalar[2][i].a, scalar[1][i].a, scalar[0][i].a);
    }
    Float4x4 mean = Float4x4::zero();
    for (auto &s : samples)
        mean += s;
    mean = mean * v4f(1.0f / 16);
    SymMat4x4 cov = SymMat4x4::zero();
    for (auto &s : samples)
        cov.accumulate_outer(s - mean, v4f(1.0f / 16));
    const __m128 flat = _mm_cmplt_ps(flat_variance(cov).v, _mm_set1_ps(kFlatVarianceEpsilon));
    Float4x4 axis = principal_axis_x4(cov);
    v4f lo(1e30f), hi(-1e30f);
    for (auto &s : samples)
    {
        v4f x = dot(s - mean, axis);
        lo = _mm_min_ps(lo.v, x.v);
        hi = _mm_max_ps(hi.v, x.v);
    }
    Float4x4 p0 = mean + axis * lo, p1 = mean + axis * hi;
    p0.r = _mm_blendv_ps(p0.r.v, mean.r.v, flat);
    p0.g = _mm_blendv_ps(p0.g.v, mean.g.v, flat);
    p0.b = _mm_blendv_ps(p0.b.v, mean.b.v, flat);
    p0.a = _mm_blendv_ps(p0.a.v, mean.a.v, flat);
    p1.r = _mm_blendv_ps(p1.r.v, mean.r.v, flat);
    p1.g = _mm_blendv_ps(p1.g.v, mean.g.v, flat);
    p1.b = _mm_blendv_ps(p1.b.v, mean.b.v, flat);
    p1.a = _mm_blendv_ps(p1.a.v, mean.a.v, flat);
    uint32_t partitions[4];
    bool has_alpha[4] = {};
    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        for (uint32_t i = 0; i < 16; ++i)
            has_alpha[lane] |= scalar[lane][i].a < 1.0f;
        partitions[lane] = choose_partition(desired_partition_mask(
            scalar[lane], extract_lane(mean, lane), extract_lane(axis, lane), has_alpha[lane]));
    }
    PartitionSubsetStats stats[4][2];
    const uint32_t flat_mask = uint32_t(_mm_movemask_ps(flat));
    if (flat_mask != 15u)
        compute_partition_subset_stats_x4(samples, partitions, stats);
    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        const Float4 low = clamp4(extract_lane(p0, lane)), high = clamp4(extract_lane(p1, lane));
        const Float4 lane_mean = extract_lane(mean, lane), lane_axis = extract_lane(axis, lane);
        const uint32_t mode = (flat_mask & (1u << lane)) ? 6u : choose_partitioned_mode(
            stats[lane], has_alpha[lane], extract_lane(cov, lane), lane_axis, low, high);
        if (mode == 1)
            out[lane] = encode_mode1_samples<Srgb>(scalar[lane], partitions[lane], stats[lane]);
        else if (mode == 7)
            out[lane] = encode_mode7_samples<Srgb>(scalar[lane], partitions[lane], stats[lane]);
        else
            out[lane] = encode_with_endpoints<Srgb>(scalar[lane], low, high, lane_mean, lane_axis);
    }
}

CDM_INLINE void generate_from_means_x4(const MeanImage &m, uint32_t bx, uint32_t by, uint32_t lanes, Block *out,
                                       bool srgb, uint32_t vw, uint32_t vh)
{
    Float4 s[4][16] = {};
    for (uint32_t lane = 0; lane < lanes; ++lane)
        for (uint32_t i = 0; i < 16; ++i)
        {
            uint32_t t = texel_index(i);
            s[lane][i] = m.get(repeat_small_coordinate((bx + lane) * 4 + (t & 3), vw),
                               repeat_small_coordinate(by * 4 + (t >> 2), vh));
        }
    for (uint32_t lane = lanes; lane < 4; ++lane)
        for (int i = 0; i < 16; ++i)
            s[lane][i] = s[0][i];
    
    Block encoded[4];
    if (srgb)
        encode_samples_x4<true>(s, encoded);
    else
        encode_samples_x4<false>(s, encoded);
    for (uint32_t lane = 0; lane < lanes; ++lane)
        out[lane] = encoded[lane];
}

CDM_INLINE void generate_child_x4(const Block *row0, const Block *row1, Block *out, const MeanImage &means,
                                  uint32_t source_x, uint32_t y0, uint32_t y1, bool srgb, uint32_t vw, uint32_t vh)
{
    Float4 s[4][16];
    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        if (srgb)
        {
            quadrant_means<true>(row0[lane * 2], s[lane]);
            quadrant_means<true>(row0[lane * 2 + 1], s[lane] + 4);
            quadrant_means<true>(row1[lane * 2], s[lane] + 8);
            quadrant_means<true>(row1[lane * 2 + 1], s[lane] + 12);
        }
        else
        {
            quadrant_means<false>(row0[lane * 2], s[lane]);
            quadrant_means<false>(row0[lane * 2 + 1], s[lane] + 4);
            quadrant_means<false>(row1[lane * 2], s[lane] + 8);
            quadrant_means<false>(row1[lane * 2 + 1], s[lane] + 12);
        }
        for (uint32_t p = 0; p < 4; ++p)
        {
            Float4 mean = (s[lane][p * 4] + s[lane][p * 4 + 1] + s[lane][p * 4 + 2] + s[lane][p * 4 + 3]) * .25f;
            means.set(source_x + lane * 2 + (p & 1), p & 2 ? y1 : y0, mean);
        }
        if (vw < 4 || vh < 4)
        {
            Float4 copy[16];
            for (int i = 0; i < 16; ++i)
                copy[i] = s[lane][i];
            for (uint32_t i = 0; i < 16; ++i)
                s[lane][i] = copy[repeat_small_sample(i, vw, vh)];
        }
    }
    if (srgb)
        encode_samples_x4<true>(s, out);
    else
        encode_samples_x4<false>(s, out);
}
#endif

#if defined(__CUDACC__)
// CUDA uses one half-warp per block: each lane owns one of the sixteen samples.
template <bool Srgb> __device__ __forceinline__ Float4 device_color(const uint8_t endpoints[2][4], uint32_t selector)
{
    uint8_t value[4];
    for (int channel = 0; channel < 4; ++channel)
        value[channel] = interpolate(endpoints[0][channel], endpoints[1][channel], selector);
    return rgba8_to_linear<Srgb>(value);
}

template <bool Srgb> __device__ __forceinline__ Float4 device_quadrant_mean(const Block &block, uint32_t quadrant)
{
    const SymbolicBC7 symbolic = parse_symbolic_block(block);
    if (!symbolic.valid)
        return Float4::zero();
    uint16_t keys[4] = {};
    uint8_t counts[4] = {};
    uint32_t entries = 0;
    const uint32_t x0 = (quadrant & 1u) * 2, y0 = (quadrant >> 1u) * 2;
    for (uint32_t dy = 0; dy < 2; ++dy)
        for (uint32_t dx = 0; dx < 2; ++dx)
        {
            const uint32_t texel = (y0 + dy) * 4 + x0 + dx;
            const uint16_t key = uint16_t(symbolic.subset[texel] | (symbolic.color_index[texel] << 2) |
                                          (symbolic.alpha_index[texel] << 6));
            uint32_t entry = 0;
            while (entry < entries && keys[entry] != key)
                ++entry;
            if (entry == entries)
                keys[entries++] = key;
            ++counts[entry];
        }
    Float4 result = Float4::zero();
    for (uint32_t entry = 0; entry < entries; ++entry)
    {
        const uint32_t subset = keys[entry] & 3, color_index = (keys[entry] >> 2) & 15;
        const uint32_t alpha_index = (keys[entry] >> 6) & 15;
        const uint8_t *a = symbolic.endpoints[subset * 2], *b = symbolic.endpoints[subset * 2 + 1];
        uint8_t rgba[4];
        for (uint32_t channel = 0; channel < 3; ++channel)
            rgba[channel] = interpolate_with_bits(a[channel], b[channel], color_index, symbolic.color_bits);
        rgba[3] = symbolic.alpha_bits
                      ? interpolate_with_bits(a[3], b[3], alpha_index, symbolic.alpha_bits)
                      : interpolate_with_bits(a[3], b[3], color_index, symbolic.color_bits);
        if (symbolic.rotation)
        {
            const uint8_t tmp = rgba[3];
            rgba[3] = rgba[symbolic.rotation - 1];
            rgba[symbolic.rotation - 1] = tmp;
        }
        const Float4 value = rgba8_to_linear<Srgb>(rgba);
        result += value * (float(counts[entry]) * 0.25f);
    }
    return result;
}

__device__ __forceinline__ PartitionSubsetStats device_subset_stats(Float4 sample, bool included, unsigned mask)
{
    PartitionSubsetStats stats;
    const float weight = included ? 1.0f : 0.0f;
    stats.count = uint32_t(sum16(weight, mask));
    const float inverse_count = 1.0f / float(stats.count);
    const Float4 weighted = sample * weight;
    stats.mean = {sum16(weighted.r, mask) * inverse_count, sum16(weighted.g, mask) * inverse_count,
                  sum16(weighted.b, mask) * inverse_count, sum16(weighted.a, mask) * inverse_count};
    const Float4 d = (sample - stats.mean) * weight;
    stats.cov = {sum16(d.r * d.r, mask) * inverse_count, sum16(d.g * d.g, mask) * inverse_count,
                 sum16(d.b * d.b, mask) * inverse_count, sum16(d.a * d.a, mask) * inverse_count,
                 sum16(d.r * d.g, mask) * inverse_count, sum16(d.r * d.b, mask) * inverse_count,
                 sum16(d.r * d.a, mask) * inverse_count, sum16(d.g * d.b, mask) * inverse_count,
                 sum16(d.g * d.a, mask) * inverse_count, sum16(d.b * d.a, mask) * inverse_count};
    Float4 axis = {1, 0, 0, 0};
    if ((threadIdx.x & 15u) == 0)
        axis = compute_principal_axis(stats.cov);
    stats.axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16),
                  __shfl_sync(mask, axis.b, 0, 16), __shfl_sync(mask, axis.a, 0, 16)};
    const float projection = dot(sample - stats.mean, stats.axis);
    stats.lo = min16(included ? projection : FLT_MAX, mask);
    stats.hi = max16(included ? projection : -FLT_MAX, mask);
    return stats;
}

template <bool Srgb>
__device__ __forceinline__ bool device_encode_partitioned(Float4 sample, uint32_t lane, unsigned mask,
                                                          const Float4 &mean, const SymMat4 &cov,
                                                          const Float4 &axis, const Float4 &low,
                                                          const Float4 &high, Block *output)
{
    if (flat_variance(cov) < kFlatVarianceEpsilon)
        return false;
    const bool has_alpha = or16(uint32_t(sample.a < 1.0f), mask) != 0;
    const Float4 delta = sample - mean;
    const float projection = delta.r * axis.r + delta.g * axis.g + delta.b * axis.b +
                             (has_alpha ? delta.a * axis.a : 0.0f);
    const uint16_t desired = uint16_t(or16(uint32_t(projection > 0.0f) << texel_index(lane), mask));
    uint32_t partition = 0;
    if (lane == 0)
        partition = choose_partition(desired);
    partition = __shfl_sync(mask, partition, 0, 16);
    const uint32_t subset = (partition_mask(partition) >> texel_index(lane)) & 1u;
    PartitionSubsetStats stats[2];
    for (uint32_t s = 0; s < 2; ++s)
        stats[s] = device_subset_stats(sample, subset == s, mask);
    uint32_t mode = 6;
    if (lane == 0)
        mode = choose_partitioned_mode(stats, has_alpha, cov, axis, low, high);
    mode = __shfl_sync(mask, mode, 0, 16);
    if (mode == 6)
        return false;

    uint8_t ep6[4][3] = {}, ep5[4][4] = {}, ep8[4][4] = {}, pbit[4] = {};
    if (lane == 0)
    {
        if (mode == 1)
        {
            quantize_mode1_endpoints<Srgb>(stats, ep6, pbit);
            for (uint32_t e = 0; e < 4; ++e)
            {
                for (uint32_t c = 0; c < 3; ++c)
                {
                    const uint32_t code7 = (uint32_t(ep6[e][c]) << 1) | pbit[e >> 1];
                    ep8[e][c] = uint8_t((code7 << 1) | (code7 >> 6));
                }
                ep8[e][3] = 255;
            }
        }
        else
            quantize_mode7_endpoints<Srgb>(stats, ep5, ep8, pbit);
    }
    for (uint32_t e = 0; e < 4; ++e)
        for (uint32_t c = 0; c < 4; ++c)
            ep8[e][c] = uint8_t(__shfl_sync(mask, uint32_t(ep8[e][c]), 0, 16));
    const uint32_t bits = mode == 1 ? 3u : 2u;
    const float target = dot(sample - stats[subset].mean, stats[subset].axis);
    float best_error = FLT_MAX;
    uint32_t selector = 0;
    for (uint32_t index = 0; index < (1u << bits); ++index)
    {
        uint8_t rgba[4];
        for (uint32_t c = 0; c < 4; ++c)
            rgba[c] = interpolate_with_bits(ep8[subset * 2][c], ep8[subset * 2 + 1][c], index, bits);
        const float delta = target - dot(rgba8_to_linear<Srgb>(rgba) - stats[subset].mean, stats[subset].axis);
        if (delta * delta < best_error)
        {
            best_error = delta * delta;
            selector = index;
        }
    }
    uint8_t selectors[16];
    for (uint32_t i = 0; i < 16; ++i)
    {
        const uint32_t selected = __shfl_sync(mask, selector, i, 16);
        if (lane == 0)
            selectors[texel_index(i)] = uint8_t(selected);
    }
    if (lane == 0)
    {
        if (mode == 1)
        {
            fix_mode1_anchors(partition, ep6, selectors);
            *output = pack_mode1(partition, ep6, pbit, selectors);
        }
        else
        {
            fix_mode7_anchors(partition, ep5, pbit, selectors);
            *output = pack_mode7(partition, ep5, pbit, selectors);
        }
    }
    return true;
}

template <bool Srgb> __device__ __forceinline__ void encode_half_warp(Float4 sample, uint32_t lane, Block *output)
{
    const unsigned mask = half_warp_mask();
    Float4 mean = {sum16(sample.r, mask) / 16, sum16(sample.g, mask) / 16, sum16(sample.b, mask) / 16,
                   sum16(sample.a, mask) / 16};
    Float4 delta = sample - mean;
    SymMat4 covariance = {sum16(delta.r * delta.r, mask) / 16, sum16(delta.g * delta.g, mask) / 16,
                          sum16(delta.b * delta.b, mask) / 16, sum16(delta.a * delta.a, mask) / 16,
                          sum16(delta.r * delta.g, mask) / 16, sum16(delta.r * delta.b, mask) / 16,
                          sum16(delta.r * delta.a, mask) / 16, sum16(delta.g * delta.b, mask) / 16,
                          sum16(delta.g * delta.a, mask) / 16, sum16(delta.b * delta.a, mask) / 16};
    const bool flat = flat_variance(covariance) < kFlatVarianceEpsilon;
    Float4 axis = {1, 0, 0, 0};
    if (lane == 0)
        axis = compute_principal_axis(covariance);
    axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16), __shfl_sync(mask, axis.b, 0, 16),
            __shfl_sync(mask, axis.a, 0, 16)};
    const float projection = dot(delta, axis);
    const float low = min16(projection, mask), high = max16(projection, mask);
    Float4 endpoints_float[2] = {mean + axis * low, mean + axis * high};
    if (flat)
        endpoints_float[0] = endpoints_float[1] = mean;
    if (device_encode_partitioned<Srgb>(sample, lane, mask, mean, covariance, axis,
                                        clamp4(endpoints_float[0]), clamp4(endpoints_float[1]), output))
        return;
    uint32_t endpoints[2][4] = {};
    if (lane == 0)
    {
        for (Float4 &v : endpoints_float)
            v = clamp4(v);
        uint8_t quantized[2][4];
        quantize_endpoint<Srgb>(endpoints_float[0], quantized[0]);
        quantize_endpoint<Srgb>(endpoints_float[1], quantized[1]);
        for (int e = 0; e < 2; ++e)
            for (int c = 0; c < 4; ++c)
                endpoints[e][c] = quantized[e][c];
    }
    for (int e = 0; e < 2; ++e)
        for (int c = 0; c < 4; ++c)
            endpoints[e][c] = __shfl_sync(mask, endpoints[e][c], 0, 16);
    uint8_t ep[2][4];
    for (int e = 0; e < 2; ++e)
        for (int c = 0; c < 4; ++c)
            ep[e][c] = uint8_t(endpoints[e][c]);
    const Float4 owned = device_color<Srgb>(ep, lane);
    const float owned_projection = dot(owned - mean, axis);
    float best = 1e30f;
    uint32_t selector = 0;
    for (uint32_t s = 0; s < 16; ++s)
    {
        const float delta = projection - __shfl_sync(mask, owned_projection, s, 16);
        const float error = delta * delta;
        if (error < best)
        {
            best = error;
            selector = s;
        }
    }
    const bool reverse = __shfl_sync(mask, selector >= 8, 0, 16);
    if (reverse)
    {
        selector = 15 - selector;
        for (int c = 0; c < 4; ++c)
        {
            uint8_t t = ep[0][c];
            ep[0][c] = ep[1][c];
            ep[1][c] = t;
        }
    }

    uint8_t selectors[16];

    for (int i = 0; i < 16; ++i) 
    {
        const uint32_t selected = __shfl_sync(mask, selector, i, 16);
        if (lane == 0) 
        {
            selectors[texel_index(i)] = uint8_t(selected);
        }
    }
    if (lane == 0)
    {
        *output = pack_block(ep, selectors);
    }

}
#endif
} // namespace bc7
