#pragma once

#include "bc4.h"

#include <immintrin.h>

namespace bc4
{
    // Build 4 BC4 UNORM palettes, one block per lane.
    inline void get_palette_unorm_x4(__m128i e0, __m128i e1, __m128 palette[8])
    {
        const __m128 endpoint_scale = _mm_set1_ps(scale);
        const __m128 scale7 = _mm_set1_ps(scale / 7.0f);
        const __m128 scale5 = _mm_set1_ps(scale / 5.0f);

        const __m128i eight_value = _mm_cmpgt_epi32(e0, e1);
        const __m128 mode_mask = _mm_castsi128_ps(eight_value);

        palette[0] = _mm_mul_ps(_mm_cvtepi32_ps(e0), endpoint_scale);
        palette[1] = _mm_mul_ps(_mm_cvtepi32_ps(e1), endpoint_scale);

        for (uint32_t selector = 2; selector < 8; ++selector)
        {
            const int weight = static_cast<int>(selector) - 1;

            const __m128i numerator7 = _mm_add_epi32(_mm_mullo_epi32(e0, _mm_set1_epi32(7 - weight)),
                                                     _mm_mullo_epi32(e1, _mm_set1_epi32(weight)));
            const __m128 value7 = _mm_mul_ps(_mm_cvtepi32_ps(numerator7), scale7);

            __m128 value5;

            if (selector < 6)
            {
                const __m128i numerator5 = _mm_add_epi32(_mm_mullo_epi32(e0, _mm_set1_epi32(5 - weight)),
                                                         _mm_mullo_epi32(e1, _mm_set1_epi32(weight)));
                value5 = _mm_mul_ps(_mm_cvtepi32_ps(numerator5), scale5);
            }
            else
            {
                value5 = _mm_set1_ps(selector == 6 ? 0.0f : 1.0f);
            }

            palette[selector] = _mm_blendv_ps(value5, value7, mode_mask);
        }
    }

    // Compute quadrant means for four full BC4 blocks.
    inline void get_quadrant_means_unorm_x4(const Block &b0, const Block &b1, const Block &b2, const Block &b3,
                                            __m128 means[4])
    {
        const __m128i e0 = _mm_setr_epi32(b0.e0, b1.e0, b2.e0, b3.e0);
        const __m128i e1 = _mm_setr_epi32(b0.e1, b1.e1, b2.e1, b3.e1);

        __m128 palette[8];
        get_palette_unorm_x4(e0, e1, palette);

        const __m128i one = _mm_set1_epi32(1);
        const __m128 weight = _mm_set1_ps(0.25f);

        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
        {
            const uint32_t start_x = (quadrant % 2) * 2;
            const uint32_t start_y = (quadrant / 2) * 2;

            __m128i counts[8];
            for (uint32_t selector = 0; selector < 8; ++selector)
            {
                counts[selector] = _mm_setzero_si128();
            }

            for (uint32_t dy = 0; dy < 2; ++dy)
            {
                for (uint32_t dx = 0; dx < 2; ++dx)
                {
                    const uint32_t x = start_x + dx;
                    const uint32_t y = start_y + dy;

                    const __m128i indices = _mm_setr_epi32(get_selector(b0, x, y), get_selector(b1, x, y),
                                                           get_selector(b2, x, y), get_selector(b3, x, y));

                    for (uint32_t selector = 0; selector < 8; ++selector)
                    {
                        const __m128i match = _mm_cmpeq_epi32(indices, _mm_set1_epi32(static_cast<int>(selector)));
                        counts[selector] = _mm_add_epi32(counts[selector], _mm_and_si128(match, one));
                    }
                }
            }

            __m128 mean = _mm_setzero_ps();

            for (uint32_t selector = 0; selector < 8; ++selector)
            {
                const __m128 fraction = _mm_mul_ps(_mm_cvtepi32_ps(counts[selector]), weight);
                mean = _mm_add_ps(mean, _mm_mul_ps(palette[selector], fraction));
            }

            means[quadrant] = mean;
        }
    }

    // Fit endpoints independently for four child blocks.
    inline void fit_endpoints_x4(const __m128 samples[16], __m128 &low, __m128 &high)
    {
        const __m128 weight = _mm_set1_ps(1.0f / 16.0f);
        __m128 mean = _mm_setzero_ps();

        for (uint32_t i = 0; i < 16; ++i)
        {
            mean = _mm_add_ps(mean, samples[i]);
        }

        mean = _mm_mul_ps(mean, weight);

        low = samples[0];
        high = samples[0];
        __m128 variance = _mm_setzero_ps();

        for (uint32_t i = 0; i < 16; ++i)
        {
            low = _mm_min_ps(low, samples[i]);
            high = _mm_max_ps(high, samples[i]);

            const __m128 delta = _mm_sub_ps(samples[i], mean);
            const __m128 squared = _mm_mul_ps(delta, delta);

            variance = _mm_add_ps(variance, _mm_mul_ps(squared, weight));
        }

        const __m128 flat = _mm_cmplt_ps(variance, _mm_set1_ps(1e-10f));

        low = _mm_blendv_ps(low, mean, flat);
        high = _mm_blendv_ps(high, mean, flat);
    }

    // Quantize four finite endpoint values to integers in [0, 255].
    inline __m128i quantize_unorm_x4(__m128 value)
    {
        const __m128 clamped = _mm_min_ps(_mm_max_ps(value, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        const __m128 scaled = _mm_mul_ps(clamped, _mm_set1_ps(255.0f));
        const __m128 rounded = _mm_add_ps(scaled, _mm_set1_ps(0.5f));

        return _mm_cvttps_epi32(rounded);
    }

    // Choose one selector per block for four samples.
    inline __m128i find_best_selector_x4(__m128 sample, const __m128 palette[8])
    {
        __m128i best_selector = _mm_setzero_si128();

        const __m128 first_delta = _mm_sub_ps(sample, palette[0]);
        __m128 best_error = _mm_mul_ps(first_delta, first_delta);

        for (uint32_t selector = 1; selector < 8; ++selector)
        {
            const __m128 delta = _mm_sub_ps(sample, palette[selector]);
            const __m128 error = _mm_mul_ps(delta, delta);
            const __m128 closer = _mm_cmplt_ps(error, best_error);

            best_error = _mm_blendv_ps(best_error, error, closer);
            best_selector =
                _mm_blendv_epi8(best_selector, _mm_set1_epi32(static_cast<int>(selector)), _mm_castps_si128(closer));
        }

        return best_selector;
    }

    // Encode four blocks from quadrant-grouped samples.
    inline void encode_samples_x4(const __m128 samples[16], Block output[4])
    {
        __m128 low;
        __m128 high;
        fit_endpoints_x4(samples, low, high);

        const __m128i e0 = quantize_unorm_x4(high);
        const __m128i e1 = quantize_unorm_x4(low);

        __m128 palette[8];
        get_palette_unorm_x4(e0, e1, palette);

        uint32_t endpoints0[4];
        uint32_t endpoints1[4];

        _mm_storeu_si128(reinterpret_cast<__m128i *>(endpoints0), e0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(endpoints1), e1);

        for (uint32_t lane = 0; lane < 4; ++lane)
        {
            output[lane] = {};
            output[lane].e0 = static_cast<uint8_t>(endpoints0[lane]);
            output[lane].e1 = static_cast<uint8_t>(endpoints1[lane]);
        }

        for (uint32_t i = 0; i < 16; ++i)
        {
            const __m128i selectors = find_best_selector_x4(samples[i], palette);

            uint32_t indices[4];
            _mm_storeu_si128(reinterpret_cast<__m128i *>(indices), selectors);

            const uint32_t texel = texel_map[i];
            const uint32_t x = texel % 4;
            const uint32_t y = texel / 4;

            for (uint32_t lane = 0; lane < 4; ++lane)
            {
                set_selector(output[lane], x, y, indices[lane]);
            }
        }
    }

    // Store eight parent means; destination must contain an 8x1 area.
    inline void store_block_mean_pairs_x4(const __m128 left_quadrants[4], const __m128 right_quadrants[4],
                                          const MeanImage &destination, uint32_t x, uint32_t y)
    {
        const __m128 weight = _mm_set1_ps(0.25f);

        const __m128 left =
            _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(left_quadrants[0], left_quadrants[1]), left_quadrants[2]),
                                  left_quadrants[3]),
                       weight);
        const __m128 right =
            _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(right_quadrants[0], right_quadrants[1]), right_quadrants[2]),
                                  right_quadrants[3]),
                       weight);

        const size_t index = static_cast<size_t>(y) * destination.width + x;

        _mm_storeu_ps(destination.r + index, _mm_unpacklo_ps(left, right));
        _mm_storeu_ps(destination.r + index + 4, _mm_unpackhi_ps(left, right));

        const __m128 zero = _mm_setzero_ps();
        _mm_storeu_ps(destination.g + index, zero);
        _mm_storeu_ps(destination.g + index + 4, zero);
        _mm_storeu_ps(destination.b + index, zero);
        _mm_storeu_ps(destination.b + index + 4, zero);
    }

    // Generate four full child blocks from two rows of eight parents.
    // The mean destination must contain an 8x2 area.
    inline void generate_child_blocks_x4(const Block *row0, const Block *row1, const MeanImage &means,
                                         uint32_t source_x, uint32_t source_y, Block output[4])
    {
        __m128 samples[16];

        get_quadrant_means_unorm_x4(row0[0], row0[2], row0[4], row0[6], samples);
        get_quadrant_means_unorm_x4(row0[1], row0[3], row0[5], row0[7], samples + 4);
        get_quadrant_means_unorm_x4(row1[0], row1[2], row1[4], row1[6], samples + 8);
        get_quadrant_means_unorm_x4(row1[1], row1[3], row1[5], row1[7], samples + 12);

        store_block_mean_pairs_x4(samples, samples + 4, means, source_x, source_y);
        store_block_mean_pairs_x4(samples + 8, samples + 12, means, source_x, source_y + 1);

        encode_samples_x4(samples, output);
    }

    // Source must contain four full blocks in a 16x4 area.
    inline void generate_child_blocks_from_means_x4(const MeanImage &source, uint32_t block_x, uint32_t block_y,
                                                    Block output[4])
    {
        __m128 samples[16];

        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = texel_map[i];
            const uint32_t x = block_x * 4 + (texel & 3u);
            const uint32_t y = block_y * 4 + (texel >> 2);
            const size_t index = static_cast<size_t>(y) * source.width + x;
            samples[i] = _mm_setr_ps(source.r[index], source.r[index + 4], source.r[index + 8], source.r[index + 12]);
        }

        encode_samples_x4(samples, output);
    }

} // namespace bc4