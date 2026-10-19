#pragma once

#include "bc6h.h"
#include "bc1_simd.h"

namespace bc6h
{
    inline __m128 sanitize_hdr_x4(__m128 value)
    {
        const __m128i magnitude = _mm_and_si128(_mm_castps_si128(value), _mm_set1_epi32(0x7fffffff));
        const __m128 finite = _mm_castsi128_ps(_mm_cmplt_epi32(magnitude, _mm_set1_epi32(0x7f800000)));
        return _mm_min_ps(_mm_max_ps(_mm_and_ps(value, finite), _mm_setzero_ps()), _mm_set1_ps(65504.0f));
    }

    inline void encode_samples_x4(const Float3x4 samples[16], Block output[4])
    {
        Float3x4 clean[16];
        for (uint32_t i = 0; i < 16; ++i)
            clean[i] = {sanitize_hdr_x4(samples[i].r), sanitize_hdr_x4(samples[i].g), sanitize_hdr_x4(samples[i].b)};
        Float3x4 mean;
        SymMat3x4 covariance;
        bc1::compute_child_moments(clean, mean, covariance);
        // HDR quantization needs a normalized axis beyond the reciprocal-sqrt approximation.
        const Float3x4 estimated_axis = compute_principal_axis(covariance);
        const __m128 inverse_length = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(length_sq(estimated_axis)));
        const Float3x4 axis = estimated_axis * inverse_length;
        __m128 projections[16];
        __m128 minimum = _mm_set1_ps(1e30f), maximum = _mm_set1_ps(-1e30f);
        for (uint32_t i = 0; i < 16; ++i)
        {
            projections[i] = dot(clean[i] - mean, axis);
            minimum = _mm_min_ps(minimum, projections[i]);
            maximum = _mm_max_ps(maximum, projections[i]);
        }
        const __m128 scale = _mm_max_ps(_mm_max_ps(_mm_max_ps(mean.r, mean.g), mean.b), _mm_set1_ps(5.96046448e-8f));
        const __m128 threshold = _mm_mul_ps(_mm_mul_ps(scale, scale), _mm_set1_ps(1e-12f));
        const __m128 variance = _mm_add_ps(_mm_add_ps(covariance.rr, covariance.gg), covariance.bb);
        const __m128 flat = _mm_cmple_ps(variance, threshold);
        const Float3x4 low = select(mean + axis * minimum, mean, flat);
        const Float3x4 high = select(mean + axis * maximum, mean, flat);
        alignas(16) float values[2][3][4], means[3][4], axes[3][4];
        _mm_store_ps(values[0][0], low.r);
        _mm_store_ps(values[0][1], low.g);
        _mm_store_ps(values[0][2], low.b);
        _mm_store_ps(values[1][0], high.r);
        _mm_store_ps(values[1][1], high.g);
        _mm_store_ps(values[1][2], high.b);
        _mm_store_ps(means[0], mean.r);
        _mm_store_ps(means[1], mean.g);
        _mm_store_ps(means[2], mean.b);
        _mm_store_ps(axes[0], axis.r);
        _mm_store_ps(axes[1], axis.g);
        _mm_store_ps(axes[2], axis.b);
        uint32_t endpoints[4][2][3];
        alignas(16) float palette_projections[16][4];
        for (uint32_t lane = 0; lane < 4; ++lane)
        {
            for (uint32_t e = 0; e < 2; ++e)
                for (uint32_t c = 0; c < 3; ++c)
                    endpoints[lane][e][c] = quantize_endpoint(values[e][c][lane]);
            const Float3 lane_mean = {means[0][lane], means[1][lane], means[2][lane]};
            const Float3 lane_axis = {axes[0][lane], axes[1][lane], axes[2][lane]};
            for (uint32_t s = 0; s < 16; ++s)
                palette_projections[s][lane] = dot(palette_color(endpoints[lane], s) - lane_mean, lane_axis);
        }
        uint32_t selectors[4][16]{};
        for (uint32_t i = 0; i < 16; ++i)
        {
            __m128 best = _mm_set1_ps(1e30f);
            __m128i selected = _mm_setzero_si128();
            for (uint32_t s = 0; s < 16; ++s)
            {
                const __m128 delta = _mm_sub_ps(projections[i], _mm_load_ps(palette_projections[s]));
                const __m128 error = _mm_mul_ps(delta, delta);
                const __m128 closer = _mm_cmplt_ps(error, best);
                best = _mm_blendv_ps(best, error, closer);
                selected = _mm_blendv_epi8(selected, _mm_set1_epi32(s), _mm_castps_si128(closer));
            }
            uint32_t indices[4];
            _mm_storeu_si128(reinterpret_cast<__m128i *>(indices), selected);
            for (uint32_t lane = 0; lane < 4; ++lane)
                selectors[lane][texel_index(i)] = indices[lane];
        }
        for (uint32_t lane = 0; lane < 4; ++lane)
            output[lane] = pack_mode11(endpoints[lane], selectors[lane]);
    }
}
