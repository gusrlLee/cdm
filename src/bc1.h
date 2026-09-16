#pragma once

#include "base.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace bc1
{
    // ---------------------------------------------------------------------
    // BC1 types / constants
    // ---------------------------------------------------------------------

    // Linear mean pyramid. Separate channels keep downsampling contiguous and vectorizable.
    struct MeanImage
    {
        float *r = nullptr;
        float *g = nullptr;
        float *b = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;

        CDM_INLINE Float3 get(uint32_t x, uint32_t y) const
        {
            x = x < width ? x : width - 1;
            y = y < height ? y : height - 1;
            size_t index = (size_t)y * width + x;
            return {r[index], g[index], b[index]};
        }

        CDM_INLINE void set(uint32_t x, uint32_t y, const Float3 &value) const
        {
            size_t index = (size_t)y * width + x;
            r[index] = value.r;
            g[index] = value.g;
            b[index] = value.b;
        }
    };

    // Below this total sample variance, a block uses its mean as both endpoints.
    constexpr float kFlatVarianceEpsilon = 1e-10f;

    // Maps a linear child-sample index (parent*4 + quadrant) to its texel position
    // inside the 4x4 block, so results match a block's native index layout.
    constexpr uint8_t texel_map[16] = {
        0, 1, 4, 5,    // p00
        2, 3, 6, 7,    // p10
        8, 9, 12, 13,  // p01
        10, 11, 14, 15 // p11
    };

    CDM_INLINE constexpr uint32_t texel_index(uint32_t sample)
    {
        uint32_t parent = sample >> 2;
        uint32_t quadrant = sample & 3u;
        uint32_t x = ((parent & 1u) << 1) | (quadrant & 1u);
        uint32_t y = (parent & 2u) | (quadrant >> 1);
        return y * 4 + x;
    }

    CDM_INLINE uint32_t repeat_small_coordinate(uint32_t value, uint32_t valid)
    {
        return valid < 4 ? value % valid : value;
    }

    CDM_INLINE uint32_t repeat_small_sample(uint32_t sample, uint32_t width, uint32_t height)
    {
        uint32_t texel = texel_index(sample);
        uint32_t x = repeat_small_coordinate(texel & 3u, width);
        uint32_t y = repeat_small_coordinate(texel >> 2, height);
        return ((y >> 1) * 2 + (x >> 1)) * 4 + (y & 1u) * 2 + (x & 1u);
    }

    template <typename T>
    CDM_INLINE void repeat_small_samples(Vec3T<T> samples[16], uint32_t width, uint32_t height)
    {
        if (width >= 4 && height >= 4) return;
        Vec3T<T> original[16];
        for (uint32_t i = 0; i < 16; ++i) original[i] = samples[i];
        for (uint32_t i = 0; i < 16; ++i)
            samples[i] = original[repeat_small_sample(i, width, height)];
    }

    struct Rgb8
    {
        uint32_t r, g, b;
    };

    // ---------------------------------------------------------------------
    // Endpoint decode / palette reconstruction
    // ---------------------------------------------------------------------

    // Decode the two RGB565 endpoints and reconstruct the palette color for one selector.
    template <bool Opaque = false>
    CDM_INLINE Rgb8 palette_color_rgb8(uint16_t c0, uint16_t c1, uint32_t selector)
    {
        uint32_t r0 = (c0 >> 11) & 31u; r0 = (r0 << 3) | (r0 >> 2);
        uint32_t g0 = (c0 >> 5) & 63u;  g0 = (g0 << 2) | (g0 >> 4);
        uint32_t b0 = c0 & 31u;         b0 = (b0 << 3) | (b0 >> 2);
        uint32_t r1 = (c1 >> 11) & 31u; r1 = (r1 << 3) | (r1 >> 2);
        uint32_t g1 = (c1 >> 5) & 63u;  g1 = (g1 << 2) | (g1 >> 4);
        uint32_t b1 = c1 & 31u;         b1 = (b1 << 3) | (b1 >> 2);

        if (selector == 0) return {r0, g0, b0};
        if (selector == 1) return {r1, g1, b1};
        if (selector == 2)
        {
            if constexpr (Opaque)
                return {(2 * r0 + r1) / 3, (2 * g0 + g1) / 3, (2 * b0 + b1) / 3};
            if (c0 > c1)
                return {(2 * r0 + r1) / 3, (2 * g0 + g1) / 3, (2 * b0 + b1) / 3};
            return {(r0 + r1) >> 1, (g0 + g1) >> 1, (b0 + b1) >> 1};
        }
        if constexpr (!Opaque)
            if (c0 <= c1)
                return {};
        return {(r0 + 2 * r1) / 3, (g0 + 2 * g1) / 3, (b0 + 2 * b1) / 3};
    }

    // ---------------------------------------------------------------------
    // Selector decode
    // ---------------------------------------------------------------------

    CDM_INLINE uint32_t count_2x2_regions(uint32_t flags)
    {
        uint32_t horizontal = (flags & 0x11111111u) + ((flags >> 2) & 0x11111111u);
        return (horizontal & 0x00ff00ffu) + ((horizontal >> 8) & 0x00ff00ffu);
    }

    // For a given selector value, count how many texels in each of the block's four
    // 2x2 quadrants use that selector (one nibble per quadrant).
    CDM_INLINE uint32_t selector_region_counts(uint32_t indices, uint32_t selector)
    {
        uint32_t different = indices ^ (selector * 0x55555555u);
        different |= different >> 1;
        return count_2x2_regions((~different) & 0x55555555u);
    }

    template <uint32_t MaxValue>
    CDM_INLINE uint32_t quantize_with_thresholds(float value, const float *thresholds)
    {
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        uint32_t first = 0, count = MaxValue;
        while (count)
        {
            uint32_t step = count >> 1;
            uint32_t middle = first + step;
            if (value < thresholds[middle]) count = step;
            else { first = middle + 1; count -= step + 1; }
        }
        return first;
    }

    // Mean and covariance of the 16 child samples around their global mean.
    template <typename T>
    CDM_INLINE void compute_child_moments(const Vec3T<T> samples[16], Vec3T<T> &mean, SymMat3T<T> &cov)
    {
        mean = Vec3T<T>::zero();
        for (int i = 0; i < 16; ++i)
            mean += samples[i];
        mean = mean * T(1.0f / 16.0f);

        cov = SymMat3T<T>::zero();
        for (int i = 0; i < 16; ++i)
            cov.accumulate_outer(samples[i] - mean, T(1.0f / 16.0f));
    }

#if !defined(__CUDACC__)

    // Explicit 256-entry sRGB to Linear table used by the CPU paths.
static const float c_srgb_to_linear[256] = {
    0.0f, 0.000303527f, 0.000607054f, 0.000910581f, 0.001214108f, 0.001517635f, 0.001821162f, 0.002124689f,
    0.002428216f, 0.002731743f, 0.003035270f, 0.003346536f, 0.003676507f, 0.004024717f, 0.004391442f, 0.004776953f,
    0.005181517f, 0.005605392f, 0.006048833f, 0.006512091f, 0.006995410f, 0.007499032f, 0.008023193f, 0.008568126f,
    0.009134059f, 0.009721217f, 0.010329820f, 0.010960090f, 0.011612250f, 0.012286490f, 0.012983030f, 0.013702080f,
    0.014443840f, 0.015208510f, 0.015996290f, 0.016807380f, 0.017641950f, 0.018500220f, 0.019382360f, 0.020288560f,
    0.021219010f, 0.022173880f, 0.023153370f, 0.024157630f, 0.025186860f, 0.026241220f, 0.027320890f, 0.028426040f,
    0.029556830f, 0.030713440f, 0.031896030f, 0.033104770f, 0.034339810f, 0.035601310f, 0.036889450f, 0.038204370f,
    0.039546240f, 0.040915200f, 0.042311410f, 0.043735030f, 0.045186200f, 0.046665090f, 0.048171820f, 0.049706570f,
    0.051269460f, 0.052860650f, 0.054480280f, 0.056128490f, 0.057805430f, 0.059511240f, 0.061246050f, 0.063010020f,
    0.064803270f, 0.066625940f, 0.068478170f, 0.070360100f, 0.072271850f, 0.074213570f, 0.076185380f, 0.078187420f,
    0.080219820f, 0.082282710f, 0.084376210f, 0.086500460f, 0.088655590f, 0.090841710f, 0.093058960f, 0.095307470f,
    0.097587350f, 0.099898730f, 0.102241700f, 0.104616500f, 0.107023100f, 0.109461700f, 0.111932400f, 0.114435400f,
    0.116970700f, 0.119538400f, 0.122138800f, 0.124771800f, 0.127437700f, 0.130136500f, 0.132868300f, 0.135633300f,
    0.138431600f, 0.141263300f, 0.144128500f, 0.147027300f, 0.149959800f, 0.152926200f, 0.155926500f, 0.158960800f,
    0.162029400f, 0.165132200f, 0.168269400f, 0.171441100f, 0.174647400f, 0.177888400f, 0.181164200f, 0.184475000f,
    0.187820800f, 0.191201700f, 0.194617800f, 0.198069300f, 0.201556300f, 0.205078700f, 0.208636900f, 0.212230800f,
    0.215860500f, 0.219526200f, 0.223228000f, 0.226965900f, 0.230740000f, 0.234550600f, 0.238397600f, 0.242281100f,
    0.246201300f, 0.250158300f, 0.254152100f, 0.258182900f, 0.262250700f, 0.266355600f, 0.270497800f, 0.274677300f,
    0.278894300f, 0.283148700f, 0.287440800f, 0.291770600f, 0.296138300f, 0.300543800f, 0.304987300f, 0.309468900f,
    0.313988700f, 0.318546800f, 0.323143200f, 0.327778100f, 0.332451500f, 0.337163600f, 0.341914400f, 0.346704100f,
    0.351532600f, 0.356400100f, 0.361306800f, 0.366252600f, 0.371237700f, 0.376262100f, 0.381326000f, 0.386429400f,
    0.391572500f, 0.396755200f, 0.401977800f, 0.407240200f, 0.412542600f, 0.417885100f, 0.423267700f, 0.428690500f,
    0.434153600f, 0.439657200f, 0.445201200f, 0.450785800f, 0.456411000f, 0.462077000f, 0.467783800f, 0.473531500f,
    0.479320200f, 0.485149900f, 0.491020800f, 0.496933000f, 0.502886500f, 0.508881300f, 0.514917700f, 0.520995600f,
    0.527115100f, 0.533276400f, 0.539479500f, 0.545724500f, 0.552011400f, 0.558340400f, 0.564711500f, 0.571124800f,
    0.577580400f, 0.584078400f, 0.590618800f, 0.597201800f, 0.603827300f, 0.610495600f, 0.617206600f, 0.623960400f,
    0.630757100f, 0.637596900f, 0.644479700f, 0.651405600f, 0.658374800f, 0.665387300f, 0.672443200f, 0.679542500f,
    0.686685300f, 0.693871800f, 0.701101900f, 0.708375800f, 0.715693500f, 0.723055100f, 0.730460700f, 0.737910400f,
    0.745404200f, 0.752942200f, 0.760524500f, 0.768151100f, 0.775822200f, 0.783537800f, 0.791297900f, 0.799102700f,
    0.806952300f, 0.814846600f, 0.822785800f, 0.830769900f, 0.838799000f, 0.846873200f, 0.854992600f, 0.863157200f,
    0.871367100f, 0.879622400f, 0.887923100f, 0.896269400f, 0.904661200f, 0.913098700f, 0.921581900f, 0.930110900f,
    0.938685700f, 0.947306500f, 0.955973400f, 0.964686200f, 0.973445300f, 0.982250600f, 0.991102100f, 1.0f};

    // Sample a single texel at (tx, ty) clamped to valid block dimensions
    CDM_INLINE Float3 sample_texel(
        const Block64 &block, const Float3 palette[4],
        uint32_t tx, uint32_t ty,
        uint32_t valid_w, uint32_t valid_h)
    {
        // Clamp coordinates to valid texels to prevent sampling black padding
        tx = (tx < valid_w) ? tx : (valid_w - 1);
        ty = (ty < valid_h) ? ty : (valid_h - 1);
        uint32_t shift = (ty * 4 + tx) * 2;
        uint32_t idx = (block.indices >> shift) & 0x3;
        return palette[idx];
    }

    template <bool IsSrgb, bool Opaque = false>
    CDM_INLINE void get_palette_impl(uint16_t c0, uint16_t c1, Float3 palette[4])
    {
        for (int k = 0; k < 4; ++k)
        {
            Rgb8 color = palette_color_rgb8<Opaque>(c0, c1, k);
            if constexpr (IsSrgb)
                palette[k] = {c_srgb_to_linear[color.r], c_srgb_to_linear[color.g], c_srgb_to_linear[color.b]};
            else
                palette[k] = {color.r * (1.0f / 255.0f), color.g * (1.0f / 255.0f), color.b * (1.0f / 255.0f)};
        }
    }

    // Decode a BC1 block's 4-color palette into linear (or sRGB-decoded) float RGB.
    CDM_INLINE void get_palette_scalar(uint16_t c0, uint16_t c1, bool is_srgb, Float3 palette[4])
    {
        if (is_srgb) get_palette_impl<true>(c0, c1, palette);
        else get_palette_impl<false>(c0, c1, palette);
    }

    // ---------------------------------------------------------------------
    // Child target construction
    // ---------------------------------------------------------------------

    // Quadrant means with boundary clamping to prevent padding pixel contamination
    // Scalar decode path
    CDM_INLINE void get_quadrant_means_scalar(const Block64 &block, bool is_srgb, Float3 out_quadrants[4], uint32_t valid_w = 4, uint32_t valid_h = 4)
    {
        Float3 pal[4];
        get_palette_scalar(block.c0, block.c1, is_srgb, pal);

        if (valid_w == 4 && valid_h == 4)
        {
            out_quadrants[0] = out_quadrants[1] = out_quadrants[2] = out_quadrants[3] = Float3::zero();
            constexpr uint32_t shifts[4] = {0, 4, 16, 20};
            for (uint32_t selector = 0; selector < 4; ++selector)
            {
                uint32_t counts = selector_region_counts(block.indices, selector);
                for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
                    out_quadrants[quadrant] += pal[selector] * (float((counts >> shifts[quadrant]) & 0xfu) * 0.25f);
            }
            return;
        }

        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
        {
            uint32_t x = (quadrant & 1u) * 2u;
            uint32_t y = (quadrant >> 1u) * 2u;
            out_quadrants[quadrant] = (sample_texel(block, pal, x, y, valid_w, valid_h) +
                                       sample_texel(block, pal, x + 1, y, valid_w, valid_h) +
                                       sample_texel(block, pal, x, y + 1, valid_w, valid_h) +
                                       sample_texel(block, pal, x + 1, y + 1, valid_w, valid_h)) * 0.25f;
        }
    }

    // ---------------------------------------------------------------------
    // RGB565 quantization
    // ---------------------------------------------------------------------

    // Shared BC1 encoding helpers
    template <uint32_t MaxValue>
    CDM_INLINE std::array<float, MaxValue> make_srgb_quantize_thresholds()
    {
        std::array<float, MaxValue> result{};
        for (uint32_t i = 0; i < MaxValue; ++i)
        {
            constexpr uint32_t bits = MaxValue == 31 ? 5 : 6;
            uint32_t a = (i << (8 - bits)) | (i >> (2 * bits - 8));
            uint32_t b = ((i + 1) << (8 - bits)) | ((i + 1) >> (2 * bits - 8));
            float s = float(a + b) * (1.0f / 510.0f);
            result[i] = (s <= 0.04045f) ? (s / 12.92f) : powf((s + 0.055f) / 1.055f, 2.4f);
        }
        return result;
    }

    template <uint32_t MaxValue>
    CDM_INLINE uint32_t quantize_linear_channel(float value)
    {
        static const std::array<float, MaxValue> thresholds = make_srgb_quantize_thresholds<MaxValue>();
        return quantize_with_thresholds<MaxValue>(value, thresholds.data());
    }

    template <bool IsSrgb>
    CDM_INLINE uint16_t encode_rgb565(Float3 col)
    {
        col = clamp01(col);
        uint32_t r, g, b;
        if constexpr (IsSrgb)
        {
            r = quantize_linear_channel<31>(col.r);
            g = quantize_linear_channel<63>(col.g);
            b = quantize_linear_channel<31>(col.b);
        }
        else
        {
            r = (uint32_t)(col.r * 31.0f + 0.5f);
            g = (uint32_t)(col.g * 63.0f + 0.5f);
            b = (uint32_t)(col.b * 31.0f + 0.5f);
        }
        return (uint16_t)((r << 11) | (g << 5) | b);
    }

    // ---------------------------------------------------------------------
    // Fit -> Quantize -> Select -> Encode
    // ---------------------------------------------------------------------

    // Fit in linear RGB and choose selectors along the fitted principal axis.
    template <bool IsSrgb>
    CDM_INLINE Block64 encode_samples_scalar(const Float3 samples[16])
    {
        Float3 mean;
        SymMat3 cov;
        compute_child_moments(samples, mean, cov);
        bool flat = cov.rr + cov.gg + cov.bb < kFlatVarianceEpsilon;

        Float3 axis = compute_principal_axis(cov);
        float projections[16];
        float minimum = 1e30f, maximum = -1e30f;
        for (int i = 0; i < 16; ++i)
        {
            float projection = dot(samples[i] - mean, axis);
            projections[i] = projection;
            minimum = std::min(minimum, projection);
            maximum = std::max(maximum, projection);
        }

        Float3 p0 = clamp01(mean + axis * minimum);
        Float3 p1 = clamp01(mean + axis * maximum);
        uint16_t c0 = encode_rgb565<IsSrgb>(p0);
        uint16_t c1 = encode_rgb565<IsSrgb>(p1);
        if (c0 < c1) std::swap(c0, c1);
        else if (c0 == c1) { if (c1) --c1; else ++c0; }

        Float3 palette[4];
        get_palette_impl<IsSrgb, true>(c0, c1, palette);
        if (flat)
        {
            float best_distance = 1e30f;
            uint32_t best_selector = 0;
            for (uint32_t selector = 0; selector < 4; ++selector)
            {
                float distance = length_sq(mean - palette[selector]);
                if (distance < best_distance) { best_distance = distance; best_selector = selector; }
            }
            uint32_t indices = 0;
            for (int i = 0; i < 16; ++i) indices |= best_selector << (2 * texel_map[i]);
            return {c0, c1, indices};
        }
        struct ProjectionEntry { float value; uint32_t selector; } sorted[4];
        for (uint32_t selector = 0; selector < 4; ++selector)
        {
            ProjectionEntry entry{dot(palette[selector] - mean, axis), selector};
            int position = (int)selector;
            while (position && (entry.value < sorted[position - 1].value
                || (entry.value == sorted[position - 1].value && entry.selector < sorted[position - 1].selector)))
            {
                sorted[position] = sorted[position - 1];
                --position;
            }
            sorted[position] = entry;
        }

        ProjectionEntry unique[4];
        int unique_count = 0;
        for (const ProjectionEntry &entry : sorted)
        {
            if (unique_count && entry.value == unique[unique_count - 1].value)
                unique[unique_count - 1].selector = std::min(unique[unique_count - 1].selector, entry.selector);
            else
                unique[unique_count++] = entry;
        }

        float boundaries[3];
        uint32_t boundary_selectors[3];
        for (int entry = 1; entry < unique_count; ++entry)
        {
            boundaries[entry - 1] = (unique[entry - 1].value + unique[entry].value) * 0.5f;
            boundary_selectors[entry - 1] = std::min(unique[entry - 1].selector, unique[entry].selector);
        }

        uint32_t indices = 0;
        for (int i = 0; i < 16; ++i)
        {
            uint32_t best_selector = unique[0].selector;
            for (int entry = 1; entry < unique_count; ++entry)
            {
                if (projections[i] < boundaries[entry - 1])
                    break;
                if (projections[i] == boundaries[entry - 1])
                {
                    best_selector = boundary_selectors[entry - 1];
                    break;
                }
                best_selector = unique[entry].selector;
            }
            indices |= best_selector << (2 * texel_map[i]);
        }
        return {c0, c1, indices};
    }

    CDM_INLINE Block64 encode_samples_scalar(const Float3 samples[16], bool is_srgb)
    {
        return is_srgb ? encode_samples_scalar<true>(samples) : encode_samples_scalar<false>(samples);
    }

    // ---------------------------------------------------------------------
    // GenerateChild
    // ---------------------------------------------------------------------

    // Combine four BC1 parent blocks (2x2) into one child block, one mip level down.
    CDM_INLINE Block64 generate_child_block_scalar(const Block64 &p00, const Block64 &p10,
                                             const Block64 &p01, const Block64 &p11, bool is_srgb,
                                             Float3 parent_means[4],
                                             uint32_t valid_width, uint32_t valid_height)
    {
        Float3 samples[16];
        get_quadrant_means_scalar(p00, is_srgb, samples + 0);
        get_quadrant_means_scalar(p10, is_srgb, samples + 4);
        get_quadrant_means_scalar(p01, is_srgb, samples + 8);
        get_quadrant_means_scalar(p11, is_srgb, samples + 12);
        for (uint32_t parent = 0; parent < 4; ++parent)
            parent_means[parent] = (samples[parent * 4] + samples[parent * 4 + 1] +
                samples[parent * 4 + 2] + samples[parent * 4 + 3]) * 0.25f;
        repeat_small_samples(samples, valid_width, valid_height);
        return encode_samples_scalar(samples, is_srgb);
    }

    // Same as generate_child_block_scalar, but reads its 16 samples directly from the
    // mean pyramid instead of re-decoding BC1 blocks (used from mip level 2 onward).
    CDM_INLINE Block64 generate_child_block_from_means_scalar(const MeanImage &source, uint32_t block_x,
                                                        uint32_t block_y, bool is_srgb,
                                                        uint32_t valid_width, uint32_t valid_height)
    {
        Float3 samples[16];
        for (uint32_t i = 0; i < 16; ++i)
        {
            uint32_t local = texel_map[i];
            samples[i] = source.get(
                repeat_small_coordinate(block_x * 4 + (local & 3u), valid_width),
                repeat_small_coordinate(block_y * 4 + (local >> 2), valid_height));
        }
        return encode_samples_scalar(samples, is_srgb);
    }

#endif
}

// ===========================================================================
// CPU SIMD implementation (4 blocks processed in parallel per lane)
// ===========================================================================

#if !defined(__CUDACC__)

#include <immintrin.h>

// Lightweight 4-lane float vector wrapper for SSE
struct v4f
{
    __m128 v;

    CDM_INLINE v4f() = default;
    CDM_INLINE v4f(__m128 x) : v(x) {}
    CDM_INLINE v4f(float s) : v(_mm_set1_ps(s)) {}

    // Basic arithmetic operators (Vertical SIMD)
    CDM_INLINE v4f operator+(const v4f &o) const { return _mm_add_ps(v, o.v); }
    CDM_INLINE v4f operator-(const v4f &o) const { return _mm_sub_ps(v, o.v); }
    CDM_INLINE v4f operator*(const v4f &o) const { return _mm_mul_ps(v, o.v); }

    CDM_INLINE v4f &operator+=(const v4f &o)
    {
        v = _mm_add_ps(v, o.v);
        return *this;
    }
};

CDM_INLINE v4f clamp01(const v4f &a)
{
    return _mm_min_ps(_mm_max_ps(a.v, _mm_setzero_ps()), _mm_set1_ps(1.0f));
}

// 4-block parallel types using our existing templates
using Float3x4 = Vec3T<v4f>;
using SymMat3x4 = SymMat3T<v4f>;

// Principal axis for four blocks in parallel
CDM_INLINE Float3x4 compute_principal_axis(const SymMat3x4 &cov)
{
    const v4f inv_sqrt3 = v4f(0.57735027f);
    Float3x4 axis0 = {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    Float3x4 next0 = cov.multiply(axis0);
    v4f lsq0 = length_sq(next0);

    // The normal path needs one matrix-vector product. Alternate seeds are only
    // evaluated for the rare covariance orthogonal to (1,1,1).
    __m128 degenerate0 = _mm_cmplt_ps(lsq0.v, _mm_set1_ps(1e-9f));
    if (_mm_movemask_ps(degenerate0) == 0)
    {
        v4f inv_len = _mm_rsqrt_ps(_mm_max_ps(lsq0.v, _mm_set1_ps(1e-20f)));
        return next0 * inv_len;
    }

    const v4f inv_sqrt2 = v4f(0.70710678f);
    Float3x4 axis1 = {inv_sqrt2, v4f(-0.70710678f), v4f(0.0f)};
    Float3x4 next1 = cov.multiply(axis1);
    v4f lsq1 = length_sq(next1);

    // Branchless blend: if lsq0 < 1e-9f, use Candidate 1
    __m128 mask0 = degenerate0;
    Float3x4 next;
    next.r = _mm_blendv_ps(next0.r.v, next1.r.v, mask0);
    next.g = _mm_blendv_ps(next0.g.v, next1.g.v, mask0);
    next.b = _mm_blendv_ps(next0.b.v, next1.b.v, mask0);
    v4f lsq = _mm_blendv_ps(lsq0.v, lsq1.v, mask0);

    // Candidate 2: (0, 1, -1) / sqrt(2)
    __m128 mask1 = _mm_cmplt_ps(lsq.v, _mm_set1_ps(1e-9f));
    Float3x4 axis2 = {v4f(0.0f), inv_sqrt2, v4f(-0.70710678f)};
    Float3x4 next2 = cov.multiply(axis2);
    v4f lsq2 = length_sq(next2);

    next.r = _mm_blendv_ps(next.r.v, next2.r.v, mask1);
    next.g = _mm_blendv_ps(next.g.v, next2.g.v, mask1);
    next.b = _mm_blendv_ps(next.b.v, next2.b.v, mask1);
    lsq = _mm_blendv_ps(lsq.v, lsq2.v, mask1);

    // Normalize
    v4f inv_len = _mm_rsqrt_ps(_mm_max_ps(lsq.v, _mm_set1_ps(1e-20f)));
    Float3x4 result = next * inv_len;

    // Default flat blocks to (1, 0, 0)
    __m128 flat_mask = _mm_cmplt_ps(lsq.v, _mm_set1_ps(1e-20f));
    result.r = _mm_blendv_ps(result.r.v, _mm_set1_ps(1.0f), flat_mask);
    result.g = _mm_blendv_ps(result.g.v, _mm_setzero_ps(), flat_mask);
    result.b = _mm_blendv_ps(result.b.v, _mm_setzero_ps(), flat_mask);
    return result;
}

namespace bc1
{
    // Four-block SIMD decode path
    struct PaletteBatch
    {
        Float3x4 colors[4];
    };

    CDM_INLINE __m128i load_block_field_x4(const Block64 &b0, const Block64 &b1, const Block64 &b2, const Block64 &b3, bool color1)
    {
        return _mm_set_epi32(color1 ? b3.c1 : b3.c0, color1 ? b2.c1 : b2.c0,
                             color1 ? b1.c1 : b1.c0, color1 ? b0.c1 : b0.c0);
    }

    template <bool IsSrgb>
    CDM_INLINE v4f rgb8_to_float(__m128i value)
    {
        if constexpr (!IsSrgb)
            return _mm_mul_ps(_mm_cvtepi32_ps(value), _mm_set1_ps(1.0f / 255.0f));

        return _mm_i32gather_ps(c_srgb_to_linear, value, sizeof(float));
    }

    // Expand a 5-bit channel (extracted at `shift`) to 8-bit by bit replication.
    CDM_INLINE __m128i expand5_x4(__m128i c, int shift)
    {
        __m128i v = _mm_and_si128(_mm_srl_epi32(c, _mm_cvtsi32_si128(shift)), _mm_set1_epi32(31));
        return _mm_or_si128(_mm_slli_epi32(v, 3), _mm_srli_epi32(v, 2));
    }

    // Expand the 6-bit green channel to 8-bit by bit replication.
    CDM_INLINE __m128i expand6_x4(__m128i c)
    {
        __m128i v = _mm_and_si128(_mm_srli_epi32(c, 5), _mm_set1_epi32(63));
        return _mm_or_si128(_mm_slli_epi32(v, 2), _mm_srli_epi32(v, 4));
    }

    CDM_INLINE __m128i div3_x4(__m128i v)
    {
        return _mm_srli_epi32(_mm_mullo_epi32(v, _mm_set1_epi32(683)), 11);
    }

    // Decode the 4-color palette for four blocks at once, one SSE lane per block.
    template <bool IsSrgb, bool Opaque = false>
    CDM_INLINE PaletteBatch get_palette_x4(__m128i c0, __m128i c1)
    {
        __m128i channel0[3] = {expand5_x4(c0, 11), expand6_x4(c0), expand5_x4(c0, 0)};
        __m128i channel1[3] = {expand5_x4(c1, 11), expand6_x4(c1), expand5_x4(c1, 0)};
        __m128i four_color;
        if constexpr (Opaque) four_color = _mm_set1_epi32(-1);
        else four_color = _mm_cmpgt_epi32(c0, c1);
        PaletteBatch result;
        for (int channel = 0; channel < 3; ++channel)
        {
            __m128i c2_4 = div3_x4(_mm_add_epi32(_mm_slli_epi32(channel0[channel], 1), channel1[channel]));
            __m128i c3_4 = div3_x4(_mm_add_epi32(channel0[channel], _mm_slli_epi32(channel1[channel], 1)));
            __m128i c2, c3;
            if constexpr (Opaque)
            {
                c2 = c2_4;
                c3 = c3_4;
            }
            else
            {
                __m128i c2_3 = _mm_srli_epi32(_mm_add_epi32(channel0[channel], channel1[channel]), 1);
                c2 = _mm_blendv_epi8(c2_3, c2_4, four_color);
                c3 = _mm_and_si128(c3_4, four_color);
            }
            v4f values[4] = {rgb8_to_float<IsSrgb>(channel0[channel]), rgb8_to_float<IsSrgb>(channel1[channel]),
                             rgb8_to_float<IsSrgb>(c2), rgb8_to_float<IsSrgb>(c3)};
            for (int selector = 0; selector < 4; ++selector)
            {
                if (channel == 0) result.colors[selector].r = values[selector];
                else if (channel == 1) result.colors[selector].g = values[selector];
                else result.colors[selector].b = values[selector];
            }
        }
        return result;
    }

    CDM_INLINE __m128i count_2x2_regions_x4(__m128i flags)
    {
        const __m128i horizontal_mask = _mm_set1_epi32(0x11111111u);
        __m128i horizontal = _mm_add_epi32(_mm_and_si128(flags, horizontal_mask),
                                           _mm_and_si128(_mm_srli_epi32(flags, 2), horizontal_mask));
        const __m128i vertical_mask = _mm_set1_epi32(0x00ff00ffu);
        return _mm_add_epi32(_mm_and_si128(horizontal, vertical_mask),
                             _mm_and_si128(_mm_srli_epi32(horizontal, 8), vertical_mask));
    }

    template <int Shift>
    CDM_INLINE Float3x4 quadrant_mean_x4(const PaletteBatch &palette, const __m128i hist[4])
    {
        Float3x4 result = Float3x4::zero();
        const __m128 scale = _mm_set1_ps(0.25f);
        const __m128i mask = _mm_set1_epi32(15);
        for (int selector = 0; selector < 4; ++selector)
        {
            __m128i count = _mm_and_si128(_mm_srli_epi32(hist[selector], Shift), mask);
            v4f weight = _mm_mul_ps(_mm_cvtepi32_ps(count), scale);
            result += palette.colors[selector] * weight;
        }
        return result;
    }

    // Child target construction (SIMD): quadrant means for four blocks at once.
    template <bool IsSrgb>
    CDM_INLINE void get_quadrant_means_x4(const Block64 &b0, const Block64 &b1, const Block64 &b2, const Block64 &b3,
                                          Float3x4 out[4])
    {
        __m128i c0 = load_block_field_x4(b0, b1, b2, b3, false);
        __m128i c1 = load_block_field_x4(b0, b1, b2, b3, true);
        PaletteBatch palette = get_palette_x4<IsSrgb>(c0, c1);
        __m128i indices = _mm_set_epi32(b3.indices, b2.indices, b1.indices, b0.indices);

        const __m128i low_mask = _mm_set1_epi32(0x55555555u);
        __m128i low = _mm_and_si128(indices, low_mask);
        __m128i high = _mm_and_si128(_mm_srli_epi32(indices, 1), low_mask);
        __m128i hist[4] = {
            count_2x2_regions_x4(_mm_andnot_si128(_mm_or_si128(low, high), low_mask)),
            count_2x2_regions_x4(_mm_andnot_si128(high, low)),
            count_2x2_regions_x4(_mm_andnot_si128(low, high)),
            count_2x2_regions_x4(_mm_and_si128(low, high))};
        out[0] = quadrant_mean_x4<0>(palette, hist);
        out[1] = quadrant_mean_x4<4>(palette, hist);
        out[2] = quadrant_mean_x4<16>(palette, hist);
        out[3] = quadrant_mean_x4<20>(palette, hist);
    }

    // Fit -> Quantize -> Select -> Encode (SIMD): four child blocks packed from the
    // fitted endpoints, one SSE lane per block.
    template <bool IsSrgb>
    CDM_INLINE void pack_blocks_x4(const Float3x4 samples[16], const v4f projections[16],
                                               const Float3x4 &mean, const Float3x4 &axis, Float3x4 p0, Float3x4 p1,
                                               __m128 flat_mask, Block64 out[4])
    {
        p0 = {clamp01(p0.r), clamp01(p0.g), clamp01(p0.b)};
        p1 = {clamp01(p1.r), clamp01(p1.g), clamp01(p1.b)};
        alignas(16) float p0r[4], p0g[4], p0b[4], p1r[4], p1g[4], p1b[4];
        _mm_store_ps(p0r, p0.r.v); _mm_store_ps(p0g, p0.g.v); _mm_store_ps(p0b, p0.b.v);
        _mm_store_ps(p1r, p1.r.v); _mm_store_ps(p1g, p1.g.v); _mm_store_ps(p1b, p1.b.v);

        alignas(16) uint32_t c0[4], c1[4];
        for (int lane = 0; lane < 4; ++lane)
        {
            c0[lane] = encode_rgb565<IsSrgb>({p0r[lane], p0g[lane], p0b[lane]});
            c1[lane] = encode_rgb565<IsSrgb>({p1r[lane], p1g[lane], p1b[lane]});
            if (c0[lane] < c1[lane]) std::swap(c0[lane], c1[lane]);
            else if (c0[lane] == c1[lane]) { if (c1[lane]) --c1[lane]; else ++c0[lane]; }
        }

        PaletteBatch palette = get_palette_x4<IsSrgb, true>(
            _mm_load_si128((const __m128i *)c0), _mm_load_si128((const __m128i *)c1));
        v4f palette_projection[4];
        for (int selector = 0; selector < 4; ++selector)
            palette_projection[selector] = dot(palette.colors[selector] - mean, axis);
        bool any_flat = _mm_movemask_ps(flat_mask) != 0;

        __m128i packed_indices = _mm_setzero_si128();
        for (int i = 0; i < 16; ++i)
        {
            v4f best_distance(1e30f);
            __m128i best_selector = _mm_setzero_si128();
            for (int selector = 0; selector < 4; ++selector)
            {
                v4f delta = projections[i] - palette_projection[selector];
                v4f distance = delta * delta;
                if (any_flat)
                {
                    v4f rgb_distance = length_sq(samples[i] - palette.colors[selector]);
                    distance = _mm_blendv_ps(distance.v, rgb_distance.v, flat_mask);
                }
                __m128 mask = _mm_cmplt_ps(distance.v, best_distance.v);
                best_distance = _mm_blendv_ps(best_distance.v, distance.v, mask);
                best_selector = _mm_blendv_epi8(best_selector, _mm_set1_epi32(selector), _mm_castps_si128(mask));
            }
            packed_indices = _mm_or_si128(packed_indices,
                _mm_sll_epi32(best_selector, _mm_cvtsi32_si128(2 * texel_map[i])));
        }

        alignas(16) uint32_t indices[4];
        _mm_store_si128((__m128i *)indices, packed_indices);
        for (int lane = 0; lane < 4; ++lane)
            out[lane] = {(uint16_t)c0[lane], (uint16_t)c1[lane], indices[lane]};
    }

    template <bool IsSrgb>
    CDM_INLINE void encode_samples_x4(const Float3x4 samples[16], Block64 *out_blocks)
    {
        Float3x4 mean;
        SymMat3x4 cov;
        compute_child_moments(samples, mean, cov);
        __m128 flat_mask = _mm_cmplt_ps((cov.rr + cov.gg + cov.bb).v, _mm_set1_ps(kFlatVarianceEpsilon));
        Float3x4 axis = compute_principal_axis(cov);
        v4f projections[16];
        v4f minimum(1e30f), maximum(-1e30f);
        for (int i = 0; i < 16; ++i)
        {
            projections[i] = dot(samples[i] - mean, axis);
            minimum = _mm_min_ps(minimum.v, projections[i].v);
            maximum = _mm_max_ps(maximum.v, projections[i].v);
        }
        pack_blocks_x4<IsSrgb>(samples, projections, mean, axis,
            mean + axis * minimum, mean + axis * maximum, flat_mask, out_blocks);
    }

    CDM_INLINE Float3x4 block_means_x4(const Float3x4 quadrants[4])
    {
        return (quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3]) * v4f(0.25f);
    }

    CDM_INLINE void store_block_mean_pairs_x4(const Float3x4 even_quadrants[4], const Float3x4 odd_quadrants[4], const MeanImage &destination, uint32_t x, uint32_t y)
    {
        Float3x4 even = block_means_x4(even_quadrants);
        Float3x4 odd = block_means_x4(odd_quadrants);
        size_t index = (size_t)y * destination.width + x;
        _mm_storeu_ps(destination.r + index, _mm_unpacklo_ps(even.r.v, odd.r.v));
        _mm_storeu_ps(destination.r + index + 4, _mm_unpackhi_ps(even.r.v, odd.r.v));
        _mm_storeu_ps(destination.g + index, _mm_unpacklo_ps(even.g.v, odd.g.v));
        _mm_storeu_ps(destination.g + index + 4, _mm_unpackhi_ps(even.g.v, odd.g.v));
        _mm_storeu_ps(destination.b + index, _mm_unpacklo_ps(even.b.v, odd.b.v));
        _mm_storeu_ps(destination.b + index + 4, _mm_unpackhi_ps(even.b.v, odd.b.v));
    }

    // GenerateChild (SIMD): combine two rows of 2x4 parent blocks into four child blocks.
    template <bool IsSrgb>
    CDM_INLINE void generate_child_blocks_x4_impl(const Block64 *src0, const Block64 *src1, Block64 *out_blocks,
                                                   const MeanImage &means, uint32_t source_x, uint32_t source_y0, uint32_t source_y1,
                                                   uint32_t valid_width, uint32_t valid_height)
    {
        Float3x4 samples[16];
        get_quadrant_means_x4<IsSrgb>(src0[0], src0[2], src0[4], src0[6], samples + 0);
        get_quadrant_means_x4<IsSrgb>(src0[1], src0[3], src0[5], src0[7], samples + 4);
        get_quadrant_means_x4<IsSrgb>(src1[0], src1[2], src1[4], src1[6], samples + 8);
        get_quadrant_means_x4<IsSrgb>(src1[1], src1[3], src1[5], src1[7], samples + 12);
        store_block_mean_pairs_x4(samples + 0, samples + 4, means, source_x, source_y0);
        store_block_mean_pairs_x4(samples + 8, samples + 12, means, source_x, source_y1);
        repeat_small_samples(samples, valid_width, valid_height);
        encode_samples_x4<IsSrgb>(samples, out_blocks);
    }

    CDM_INLINE void generate_child_blocks_x4(const Block64 *src0, const Block64 *src1, Block64 *out_blocks, bool is_srgb,
                                               const MeanImage &means, uint32_t source_x,
                                               uint32_t source_y0, uint32_t source_y1,
                                               uint32_t valid_width, uint32_t valid_height)
    {
        if (is_srgb) generate_child_blocks_x4_impl<true>(src0, src1, out_blocks, means, source_x, source_y0, source_y1, valid_width, valid_height);
        else generate_child_blocks_x4_impl<false>(src0, src1, out_blocks, means, source_x, source_y0, source_y1, valid_width, valid_height);
    }

    // Same as generate_child_blocks_x4, but reads its 16 samples directly from the mean
    // pyramid (used from mip level 2 onward), handling a partial (<4) lane count at the
    // right edge of a row.
    template <bool IsSrgb>
    CDM_INLINE void generate_child_blocks_from_means_x4(const MeanImage &source,
                                                         uint32_t block_x, uint32_t block_y, uint32_t valid_lanes,
                                                         Block64 *destination, uint32_t valid_width, uint32_t valid_height)
    {
        Float3x4 samples[16];
        if (valid_width >= 4 && valid_height >= 4
            && valid_lanes == 4 && (block_x + 4) * 4 <= source.width && (block_y + 1) * 4 <= source.height)
        {
            for (uint32_t i = 0; i < 16; ++i)
            {
                uint32_t local = texel_map[i];
                size_t index = (size_t)(block_y * 4 + (local >> 2)) * source.width + block_x * 4 + (local & 3u);
                samples[i] = {
                    _mm_set_ps(source.r[index + 12], source.r[index + 8], source.r[index + 4], source.r[index]),
                    _mm_set_ps(source.g[index + 12], source.g[index + 8], source.g[index + 4], source.g[index]),
                    _mm_set_ps(source.b[index + 12], source.b[index + 8], source.b[index + 4], source.b[index])};
            }
        }
        else
        {
            for (uint32_t i = 0; i < 16; ++i)
            {
                uint32_t local = texel_map[i];
                Float3 lane[4];
                for (uint32_t j = 0; j < 4; ++j)
                {
                    uint32_t lane_x = block_x + std::min(j, valid_lanes - 1);
                    lane[j] = source.get(
                        repeat_small_coordinate(lane_x * 4 + (local & 3u), valid_width),
                        repeat_small_coordinate(block_y * 4 + (local >> 2), valid_height));
                }
                samples[i] = {
                    _mm_set_ps(lane[3].r, lane[2].r, lane[1].r, lane[0].r),
                    _mm_set_ps(lane[3].g, lane[2].g, lane[1].g, lane[0].g),
                    _mm_set_ps(lane[3].b, lane[2].b, lane[1].b, lane[0].b)};
            }
        }
        Block64 encoded[4];
        encode_samples_x4<IsSrgb>(samples, encoded);
        for (uint32_t lane = 0; lane < valid_lanes; ++lane)
            destination[lane] = encoded[lane];
    }

    CDM_INLINE void generate_child_blocks_from_means_x4(const MeanImage &source,
                                                         uint32_t block_x, uint32_t block_y, uint32_t valid_lanes,
                                                         Block64 *destination, bool is_srgb,
                                                         uint32_t valid_width, uint32_t valid_height)
    {
        if (is_srgb) generate_child_blocks_from_means_x4<true>(source, block_x, block_y, valid_lanes, destination, valid_width, valid_height);
        else generate_child_blocks_from_means_x4<false>(source, block_x, block_y, valid_lanes, destination, valid_width, valid_height);
    }
}

#endif
