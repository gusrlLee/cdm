#pragma once

#include "bc1.h"
#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace bc7
{
constexpr float kFlatVarianceEpsilon = 1e-10f;

template <typename T> CDM_INLINE T flat_variance(const SymMat4T<T> &cov)
{
    return cov.rr + cov.gg + cov.bb + cov.aa;
}
// Raw BC7 block. Input accepts all eight modes; generated child blocks use Mode 6.
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
        uint8_t partition_entry = out.subsets == 1 ? uint8_t(texel ? 0 : 128)
                                                    : kPartitionSets[out.subsets - 2][partition][texel >> 2][texel & 3];
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

#if !defined(__CUDACC__)
template <bool Srgb> CDM_INLINE Float4 rgba8_to_linear(const uint8_t v[4])
{
    if constexpr (Srgb)
        return {bc1::c_srgb_to_linear[v[0]], bc1::c_srgb_to_linear[v[1]], bc1::c_srgb_to_linear[v[2]],
                v[3] * (1.0f / 255.0f)};
    return {v[0] * (1.0f / 255.0f), v[1] * (1.0f / 255.0f), v[2] * (1.0f / 255.0f), v[3] * (1.0f / 255.0f)};
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

CDM_INLINE Float4 clamp4(Float4 v)
{
    auto c = [](float x) { return std::max(0.0f, std::min(1.0f, x)); };
    return {c(v.r), c(v.g), c(v.b), c(v.a)};
}

template <bool Srgb> CDM_INLINE uint8_t quantize_component(float target, uint32_t pbit, bool alpha)
{
    uint32_t best = pbit;
    float error = 1e30f;
    for (uint32_t q = 0; q < 128; ++q)
    {
        uint32_t code = (q << 1) | pbit;
        float value = (Srgb && !alpha) ? bc1::c_srgb_to_linear[code] : code * (1.0f / 255.0f);
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
        lo = std::min(lo, p);
        hi = std::max(hi, p);
    }
    Float4 endpoints[2] = {mean, mean};
    if (!flat)
    {
        endpoints[0] = mean + axis * lo;
        endpoints[1] = mean + axis * hi;
    }
    endpoints[0] = clamp4(endpoints[0]);
    endpoints[1] = clamp4(endpoints[1]);
    uint8_t ep[2][4];
    quantize_endpoint<Srgb>(endpoints[0], ep[0]);
    quantize_endpoint<Srgb>(endpoints[1], ep[1]);
    Float4 pal[16];
    palette<Srgb>(ep, pal);
    float palette_projection[16];
    for (uint32_t s = 0; s < 16; ++s)
        palette_projection[s] = dot(pal[s] - mean, axis);
    uint8_t selector[16] = {};
    for (uint32_t i = 0; i < 16; ++i)
    {
        uint32_t texel = texel_index(i), best = 0;
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
        selector[texel] = uint8_t(best);
    }
    if (selector[0] >= 8)
    {
        for (int c = 0; c < 4; ++c)
            std::swap(ep[0][c], ep[1][c]);
        for (auto &s : selector)
            s = uint8_t(15 - s);
    }
    return pack_block(ep, selector);
}

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
            std::swap(ep[0][c], ep[1][c]);
        for (auto &s : selector)
            s = uint8_t(15 - s);
    }
    return pack_block(ep, selector);
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
    alignas(16) float values[8][4];
    _mm_store_ps(values[0], p0.r.v);
    _mm_store_ps(values[1], p0.g.v);
    _mm_store_ps(values[2], p0.b.v);
    _mm_store_ps(values[3], p0.a.v);
    _mm_store_ps(values[4], p1.r.v);
    _mm_store_ps(values[5], p1.g.v);
    _mm_store_ps(values[6], p1.b.v);
    _mm_store_ps(values[7], p1.a.v);
    alignas(16) float means[4][4], axes[4][4];
    _mm_store_ps(means[0], mean.r.v);
    _mm_store_ps(means[1], mean.g.v);
    _mm_store_ps(means[2], mean.b.v);
    _mm_store_ps(means[3], mean.a.v);
    _mm_store_ps(axes[0], axis.r.v);
    _mm_store_ps(axes[1], axis.g.v);
    _mm_store_ps(axes[2], axis.b.v);
    _mm_store_ps(axes[3], axis.a.v);
    for (int lane = 0; lane < 4; ++lane)
        out[lane] = encode_with_endpoints<Srgb>(scalar[lane],
                                                {values[0][lane], values[1][lane], values[2][lane], values[3][lane]},
                                                {values[4][lane], values[5][lane], values[6][lane], values[7][lane]},
                                                {means[0][lane], means[1][lane], means[2][lane], means[3][lane]},
                                                {axes[0][lane], axes[1][lane], axes[2][lane], axes[3][lane]});
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
    return {decode_channel<Srgb>(value[0]), decode_channel<Srgb>(value[1]), decode_channel<Srgb>(value[2]),
            value[3] * (1.0f / 255.0f)};
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
        const Float4 value = {decode_channel<Srgb>(rgba[0]), decode_channel<Srgb>(rgba[1]),
                              decode_channel<Srgb>(rgba[2]), rgba[3] * (1.0f / 255.0f)};
        result += value * (float(counts[entry]) * 0.25f);
    }
    return result;
}

template <bool Srgb> __device__ __forceinline__ uint8_t device_quantize(float target, uint32_t pbit, bool alpha)
{
    uint32_t best = pbit;
    float best_error = 1e30f;
    for (uint32_t q = 0; q < 128; ++q)
    {
        const uint32_t code = (q << 1) | pbit;
        const float value = Srgb && !alpha ? g_srgb_to_linear[code] : code * (1.0f / 255.0f);
        const float error = (target - value) * (target - value);
        if (error < best_error)
        {
            best_error = error;
            best = code;
        }
    }
    return uint8_t(best);
}

template <bool Srgb> __device__ __forceinline__ void device_quantize_endpoint(Float4 value, uint8_t output[4])
{
    float best = 1e30f;
    for (uint32_t pbit = 0; pbit < 2; ++pbit)
    {
        uint8_t candidate[4] = {
            device_quantize<Srgb>(value.r, pbit, false), device_quantize<Srgb>(value.g, pbit, false),
            device_quantize<Srgb>(value.b, pbit, false), device_quantize<Srgb>(value.a, pbit, true)};
        Float4 decoded = {decode_channel<Srgb>(candidate[0]), decode_channel<Srgb>(candidate[1]),
                          decode_channel<Srgb>(candidate[2]), candidate[3] * (1.0f / 255.0f)};
        const float error = length_sq(decoded - value);
        if (error < best)
        {
            best = error;
            for (int c = 0; c < 4; ++c)
                output[c] = candidate[c];
        }
    }
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
    uint32_t endpoints[2][4] = {};
    if (lane == 0)
    {
        for (Float4 &v : endpoints_float)
            v = {fminf(1, fmaxf(0, v.r)), fminf(1, fmaxf(0, v.g)), fminf(1, fmaxf(0, v.b)), fminf(1, fmaxf(0, v.a))};
        uint8_t quantized[2][4];
        device_quantize_endpoint<Srgb>(endpoints_float[0], quantized[0]);
        device_quantize_endpoint<Srgb>(endpoints_float[1], quantized[1]);
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
    if (lane == 0)
    {
        uint8_t selectors[16];
        selectors[0] = uint8_t(selector);
        for (int i = 1; i < 16; ++i)
            selectors[texel_index(i)] = uint8_t(__shfl_sync(mask, selector, i, 16));
        *output = pack_block(ep, selectors);
    }
}
#endif
} // namespace bc7
