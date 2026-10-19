#pragma once

#include "bc7.h"
#include <immintrin.h>

namespace bc7
{
    // Four RGBA channels accumulate together; opaque fitting still uses SymMat3.
    inline void compute_encoding_moments_simd(const Float4 *samples, uint32_t count, bool has_alpha, Float4 &mean,
                                              SymMat4 &covariance, Float4 &axis)
    {
        __m128 sum = _mm_setzero_ps();
        for (uint32_t i = 0; i < count; ++i)
        {
            sum = _mm_add_ps(sum, _mm_loadu_ps(&samples[i].r));
        }
        const __m128 weight = _mm_set1_ps(1.0f / static_cast<float>(count));
        __m128 center = _mm_mul_ps(sum, weight);
        if (!has_alpha)
        {
            center = _mm_blend_ps(center, _mm_set1_ps(1.0f), 8);
        }
        _mm_storeu_ps(&mean.r, center);
        __m128 rows[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
        for (uint32_t i = 0; i < count; ++i)
        {
            __m128 d = _mm_sub_ps(_mm_loadu_ps(&samples[i].r), center);
            rows[0] = _mm_add_ps(rows[0], _mm_mul_ps(_mm_mul_ps(d, _mm_shuffle_ps(d, d, 0)), weight));
            rows[1] = _mm_add_ps(rows[1], _mm_mul_ps(_mm_mul_ps(d, _mm_shuffle_ps(d, d, 85)), weight));
            rows[2] = _mm_add_ps(rows[2], _mm_mul_ps(_mm_mul_ps(d, _mm_shuffle_ps(d, d, 170)), weight));
            if (has_alpha)
            {
                rows[3] = _mm_add_ps(rows[3], _mm_mul_ps(_mm_mul_ps(d, _mm_shuffle_ps(d, d, 255)), weight));
            }
        }
        alignas(16) float matrix[4][4];
        for (uint32_t i = 0; i < 4; ++i)
        {
            _mm_store_ps(matrix[i], rows[i]);
        }
        covariance = {matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3], matrix[1][1],
                      matrix[1][2], matrix[1][3], matrix[2][2], matrix[2][3], matrix[3][3]};
        if (has_alpha)
        {
            axis = compute_principal_axis(covariance);
        }
        else
        {
            const SymMat3 rgb = {covariance.rr, covariance.rg, covariance.rb,
                                 covariance.gg, covariance.gb, covariance.bb};
            const Float3 direction = compute_principal_axis(rgb);
            axis = {direction.r, direction.g, direction.b, 0.0f};
        }
    }

    // Each SIMD lane projects one texel; tail subsets have fewer than four texels.
    inline void project_samples_simd(const Float4 *samples, uint32_t count, const Float4 &mean, const Float4 &axis,
                                     float *projections, float &minimum, float &maximum)
    {
        uint32_t i = 0;
        for (; i + 4 <= count; i += 4)
        {
            __m128 r = _mm_loadu_ps(&samples[i].r);
            __m128 g = _mm_loadu_ps(&samples[i + 1].r);
            __m128 b = _mm_loadu_ps(&samples[i + 2].r);
            __m128 a = _mm_loadu_ps(&samples[i + 3].r);
            _MM_TRANSPOSE4_PS(r, g, b, a);
            r = _mm_mul_ps(_mm_sub_ps(r, _mm_set1_ps(mean.r)), _mm_set1_ps(axis.r));
            g = _mm_mul_ps(_mm_sub_ps(g, _mm_set1_ps(mean.g)), _mm_set1_ps(axis.g));
            b = _mm_mul_ps(_mm_sub_ps(b, _mm_set1_ps(mean.b)), _mm_set1_ps(axis.b));
            a = _mm_mul_ps(_mm_sub_ps(a, _mm_set1_ps(mean.a)), _mm_set1_ps(axis.a));
            _mm_storeu_ps(projections + i, _mm_add_ps(_mm_add_ps(_mm_add_ps(r, g), b), a));
        }
        for (; i < count; ++i)
        {
            projections[i] = dot(samples[i] - mean, axis);
        }
        minimum = 1e30f;
        maximum = -1e30f;
        for (i = 0; i < count; ++i)
        {
            minimum = std::min(minimum, projections[i]);
            maximum = std::max(maximum, projections[i]);
        }
    }

    inline void compute_partition_subset_stats_simd(const Float4 samples[16], uint32_t partition,
                                                    PartitionSubsetStats stats[2], bool has_alpha,
                                                    float sample_projections[16])
    {
        for (uint32_t subset = 0; subset < 2; ++subset)
        {
            Float4 grouped[16];
            uint32_t texels[16];
            uint32_t count = 0;
            for (uint32_t t = 0; t < 16; ++t)
            {
                if (partition_subset(partition, t) == subset)
                {
                    texels[count] = t;
                    grouped[count++] = samples[t];
                }
            }
            Float4 mean, axis;
            SymMat4 covariance;
            compute_encoding_moments_simd(grouped, count, has_alpha, mean, covariance, axis);
            float projections[16], minimum, maximum;
            project_samples_simd(grouped, count, mean, axis, projections, minimum, maximum);
            for (uint32_t i = 0; i < count; ++i)
            {
                sample_projections[texels[i]] = projections[i];
            }
            stats[subset] = {count, mean, covariance, axis, minimum, maximum};
        }
    }

    template <uint32_t Mode>
    inline void assign_selectors_simd(uint32_t partition, const uint8_t quantized[4][4], const uint8_t pbits[4],
                                      uint8_t selectors[16], const Float4 means[2], const Float4 axes[2],
                                      const float sample_projections[16])
    {
        constexpr uint32_t mode = Mode;
        const uint32_t subsets = mode == 6 ? 1 : 2;
        const uint32_t size = mode == 6 ? 16 : mode == 1 ? 8 : 4;
        Float4 palette[2][16];
        make_palette(mode, quantized, pbits, palette);
        float projections[2][16];
        for (uint32_t s = 0; s < subsets; ++s)
        {
            float minimum, maximum;
            project_samples_simd(palette[s], size, means[s], axes[s], projections[s], minimum, maximum);
        }
        for (uint32_t t = 0; t < 16; t += 4)
        {
            int subset[4]{};
            for (uint32_t i = 0; i < 4; ++i)
            {
                subset[i] = mode == 6 ? 0 : partition_subset(partition, t + i);
            }
            const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(-subset[0], -subset[1], -subset[2], -subset[3]));
            const __m128 target = _mm_loadu_ps(sample_projections + t);
            __m128 best_error = _mm_set1_ps(1e30f);
            __m128i best = _mm_setzero_si128();
            for (uint32_t index = 0; index < size; ++index)
            {
                const __m128 p0 = _mm_set1_ps(projections[0][index]);
                const __m128 p1 = mode == 6 ? p0 : _mm_set1_ps(projections[1][index]);
                const __m128 delta = _mm_sub_ps(target, _mm_blendv_ps(p0, p1, mask));
                const __m128 error = _mm_mul_ps(delta, delta);
                const __m128 better = _mm_cmplt_ps(error, best_error);
                best_error = _mm_min_ps(best_error, error);
                best = _mm_blendv_epi8(best, _mm_set1_epi32(index), _mm_castps_si128(better));
            }
            alignas(16) uint32_t selected[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(selected), best);
            for (uint32_t i = 0; i < 4; ++i)
            {
                selectors[t + i] = static_cast<uint8_t>(selected[i]);
            }
        }
    }

    inline Block encode_samples_simd(const Float4 samples[16])
    {
        const bool has_alpha = has_alpha_samples(samples);
        Float4 mean, axis;
        SymMat4 covariance;
        compute_encoding_moments_simd(samples, 16, has_alpha, mean, covariance, axis);
        float projections[16], minimum, maximum;
        project_samples_simd(samples, 16, mean, axis, projections, minimum, maximum);
        const bool flat = covariance.rr + covariance.gg + covariance.bb + covariance.aa < 1e-10f;
        const Float4 low = flat ? mean : clamp01(mean + axis * minimum);
        const Float4 high = flat ? mean : clamp01(mean + axis * maximum);
        uint32_t mode = 6;
        uint32_t partition = 0;
        PartitionSubsetStats stats[2];
        float subset_projections[16];
        if (!flat)
        {
            partition = choose_partition(desired_partition_mask(projections));
            compute_partition_subset_stats_simd(samples, partition, stats, has_alpha, subset_projections);
            mode = choose_partitioned_mode(stats, has_alpha, covariance, axis, low, high);
        }
        uint8_t quantized[4][4]{};
        uint8_t pbits[4]{};
        uint8_t selectors[16];
        Float4 means[2] = {mean, {}};
        Float4 axes[2] = {axis, {}};
        if (mode == 6)
        {
            quantize_mode6_endpoint(low, quantized[0], pbits[0]);
            quantize_mode6_endpoint(high, quantized[1], pbits[1]);
        }
        else
        {
            Float4 endpoints[4];
            for (uint32_t s = 0; s < 2; ++s)
            {
                means[s] = stats[s].mean;
                axes[s] = stats[s].axis;
                endpoints[s * 2] = clamp01(means[s] + axes[s] * stats[s].lo);
                endpoints[s * 2 + 1] = clamp01(means[s] + axes[s] * stats[s].hi);
            }
            quantize_partition_endpoints(endpoints, mode, quantized, pbits);
        }
        // Specialize palette sizes and subset access instead of branching inside SIMD loops.
        if (mode == 6)
        {
            assign_selectors_simd<6>(partition, quantized, pbits, selectors, means, axes, projections);
        }
        else if (mode == 1)
        {
            assign_selectors_simd<1>(partition, quantized, pbits, selectors, means, axes, subset_projections);
        }
        else
        {
            assign_selectors_simd<7>(partition, quantized, pbits, selectors, means, axes, subset_projections);
        }
        fix_anchors(mode, partition, quantized, pbits, selectors);
        return pack_block(mode, partition, quantized, pbits, selectors);
    }

    inline Block generate_child_block_simd(const Block &p00, const Block &p10, const Block &p01, const Block &p11,
                                           Float4 parent_means[4], uint32_t valid_width, uint32_t valid_height,
                                           uint32_t source_width, uint32_t source_height)
    {
        Float4 samples[16];
        prepare_child_samples(p00, p10, p01, p11, parent_means, valid_width, valid_height, source_width, source_height,
                              samples);
        return encode_samples_simd(samples);
    }

    inline Block generate_child_block_from_means_simd(const MeanImage4 &source, uint32_t block_x, uint32_t block_y,
                                                      uint32_t valid_width, uint32_t valid_height)
    {
        Float4 samples[16];
        for (uint32_t y = 0; y < 4; ++y)
        {
            for (uint32_t x = 0; x < 4; ++x)
            {
                samples[y * 4 + x] = source.get(block_x * 4 + x % valid_width, block_y * 4 + y % valid_height);
            }
        }
        return encode_samples_simd(samples);
    }
}
