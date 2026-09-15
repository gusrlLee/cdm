#pragma once

#include "base.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace bc1
{
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

    struct Rgb8
    {
        uint32_t r, g, b;
    };

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

    CDM_INLINE uint32_t count_2x2_regions(uint32_t flags)
    {
        uint32_t horizontal = (flags & 0x11111111u) + ((flags >> 2) & 0x11111111u);
        return (horizontal & 0x00ff00ffu) + ((horizontal >> 8) & 0x00ff00ffu);
    }

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

    // Shared BC1 encoding helpers
    template <uint32_t MaxValue>
    CDM_INLINE uint32_t quantize_linear_channel(float value)
    {
        static const std::array<float, MaxValue> thresholds = [] {
            std::array<float, MaxValue> result{};
            for (uint32_t i = 0; i < MaxValue; ++i)
            {
                float s = (float(i) + 0.5f) / float(MaxValue);
                result[i] = (s <= 0.04045f) ? (s / 12.92f) : powf((s + 0.055f) / 1.055f, 2.4f);
            }
            return result;
        }();

        return quantize_with_thresholds<MaxValue>(value, thresholds.data());
    }

    template <bool IsSrgb>
    CDM_INLINE uint16_t encode_rgb565_fast(Float3 col)
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

    CDM_INLINE uint16_t encode_rgb565_fast(Float3 col, bool is_srgb)
    {
        return is_srgb ? encode_rgb565_fast<true>(col) : encode_rgb565_fast<false>(col);
    }

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

    CDM_INLINE void get_palette_scalar(uint16_t c0, uint16_t c1, bool is_srgb, Float3 palette[4])
    {
        if (is_srgb) get_palette_impl<true>(c0, c1, palette);
        else get_palette_impl<false>(c0, c1, palette);
    }

    CDM_INLINE void get_opaque_palette_scalar(uint16_t c0, uint16_t c1, bool is_srgb, Float3 palette[4])
    {
        if (is_srgb) get_palette_impl<true, true>(c0, c1, palette);
        else get_palette_impl<false, true>(c0, c1, palette);
    }

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

    template <typename T>
    CDM_INLINE void compute_child_moments(const Vec3T<T> samples[16], Vec3T<T> &mean, SymMat3T<T> &cov)
    {
        constexpr float quarter = 0.25f;
        Vec3T<T> parent_means[4];
        SymMat3T<T> within = SymMat3T<T>::zero();

        mean = Vec3T<T>::zero();
        for (int parent = 0; parent < 4; ++parent)
        {
            Vec3T<T> parent_mean = Vec3T<T>::zero();
            for (int i = 0; i < 4; ++i)
                parent_mean += samples[parent * 4 + i];
            parent_mean = parent_mean * T(quarter);
            parent_means[parent] = parent_mean;
            mean += parent_mean;

            for (int i = 0; i < 4; ++i)
                within.accumulate_outer(samples[parent * 4 + i] - parent_mean, T(quarter));
        }
        mean = mean * T(quarter);

        // Law of total covariance: E[Cov(X|parent)] + Cov(E[X|parent]).
        cov = within * T(quarter);
        for (const Vec3T<T> &parent_mean : parent_means)
            cov.accumulate_outer(parent_mean - mean, T(quarter));
    }

    // Scalar encoder
    CDM_INLINE Block64 encode_samples_scalar(const Float3 samples[16], bool is_srgb)
    {
        constexpr float inv3 = 1.0f / 3.0f;
        Float3 mean;
        SymMat3 cov;
        compute_child_moments(samples, mean, cov);

        // 3. Find principal axis (PCA) and inital endpoints
        Float3 axis = compute_principal_axis(cov);

        float min_proj = 1e30f;
        float max_proj = -1e30f;

        for (int i = 0; i < 16; i++)
        {
            float proj = dot(samples[i] - mean, axis);
            if (proj < min_proj)
                min_proj = proj;
            if (proj > max_proj)
                max_proj = proj;
        }

        Float3 p0 = mean + axis * min_proj;
        Float3 p1 = mean + axis * max_proj;

        // 4. Refine endpoints using least-squares fitting
        Float3 dir = p1 - p0;
        float len_sq = length_sq(dir) + 1e-12f;
        float inv_len_sq = 1.0f / len_sq;

        float sum_w = 0.0f;
        float sum_w2 = 0.0f;
        Float3 sum_y = Float3::zero();
        Float3 sum_wy = Float3::zero();
        for (int i = 0; i < 16; ++i)
        {
            float t = dot(samples[i] - p0, dir) * inv_len_sq;
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            float w = roundf(t * 3.0f) * inv3; // snap to 0, 1/3, 2/3, 1
            sum_w += w;
            sum_w2 += w * w;
            sum_y += samples[i];
            sum_wy += samples[i] * w;
        }

        solve_least_squares_endpoints(sum_w, sum_w2, sum_y, sum_wy, mean, p0, p1);

        // 5. Quantize to RGB565 and enforce opaque mode (c0 > c1)
        uint16_t c0 = encode_rgb565_fast(p0, is_srgb);
        uint16_t c1 = encode_rgb565_fast(p1, is_srgb);

        if (c0 < c1)
        {
            std::swap(c0, c1);
        }
        else if (c0 == c1)
        {
            if (c1 > 0)
                c1--;
            else if (c0 < 0xFFFF)
                c0++;
        }

        // 6. Build final palette and assign 2-bit selectors
        Float3 final_colors[4];
        get_opaque_palette_scalar(c0, c1, is_srgb, final_colors);

        uint32_t indices = 0;
        for (int i = 0; i < 16; ++i)
        {
            float min_dist = 1e30f;
            uint32_t best_idx = 0;
            for (uint32_t k = 0; k < 4; ++k)
            {
                float dist = length_sq(samples[i] - final_colors[k]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_idx = k;
                }
            }
            indices |= (best_idx << (2 * texel_map[i]));
        }
        return Block64{c0, c1, indices};
    }

    CDM_INLINE Block64 generate_child_block_scalar(const Block64 &p00, const Block64 &p10,
                                             const Block64 &p01, const Block64 &p11, bool is_srgb,
                                             Float3 parent_means[4] = nullptr)
    {
        Float3 samples[16];
        get_quadrant_means_scalar(p00, is_srgb, samples + 0);
        get_quadrant_means_scalar(p10, is_srgb, samples + 4);
        get_quadrant_means_scalar(p01, is_srgb, samples + 8);
        get_quadrant_means_scalar(p11, is_srgb, samples + 12);
        if (parent_means)
        {
            for (uint32_t parent = 0; parent < 4; ++parent)
                parent_means[parent] = (samples[parent * 4] + samples[parent * 4 + 1] +
                    samples[parent * 4 + 2] + samples[parent * 4 + 3]) * 0.25f;
        }
        return encode_samples_scalar(samples, is_srgb);
    }

    CDM_INLINE Block64 generate_child_block_from_means_scalar(const MeanImage &source, uint32_t block_x,
                                                        uint32_t block_y, bool is_srgb)
    {
        Float3 samples[16];
        for (uint32_t i = 0; i < 16; ++i)
        {
            uint32_t local = texel_map[i];
            samples[i] = source.get(block_x * 4 + (local & 3u), block_y * 4 + (local >> 2));
        }
        return encode_samples_scalar(samples, is_srgb);
    }


#endif
}
