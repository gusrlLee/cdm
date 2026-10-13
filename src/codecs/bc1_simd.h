#pragma once

#include "codecs/bc1.h"
#include "core/math_simd.h"

namespace bc1
{
    // Compute the mean and covariance for four blocks.
    inline void compute_child_moments(const Float3x4 samples[16], Float3x4 &mean, SymMat3x4 &covariance)
    {
        const __m128 zero = _mm_setzero_ps();
        const __m128 weight = _mm_set1_ps(1.0f / 16.0f);

        mean = {zero, zero, zero};

        for (uint32_t i = 0; i < 16; ++i)
        {
            mean += samples[i];
        }

        mean = mean * weight;
        covariance = {zero, zero, zero, zero, zero, zero};

        for (uint32_t i = 0; i < 16; ++i)
        {
            const Float3x4 deviation = samples[i] - mean;
            accumulate_outer(covariance, deviation, weight);
        }
    }

    // Reuse scalar quantization for each block's endpoint.
    inline __m128i encode_rgb565(const Float3x4 &color, bool is_srgb)
    {
        float r[4];
        float g[4];
        float b[4];

        _mm_storeu_ps(r, color.r);
        _mm_storeu_ps(g, color.g);
        _mm_storeu_ps(b, color.b);

        uint32_t endpoints[4];

        for (uint32_t lane = 0; lane < 4; ++lane)
        {
            const Float3 endpoint = {r[lane], g[lane], b[lane]};
            endpoints[lane] = encode_rgb565(endpoint, is_srgb);
        }

        return _mm_setr_epi32(
            static_cast<int>(endpoints[0]),
            static_cast<int>(endpoints[1]),
            static_cast<int>(endpoints[2]),
            static_cast<int>(endpoints[3]));
    }

    // Convert four 8-bit channel values to linear floats.
    // Each integer lane must be in the range [0, 255].
    inline __m128 decode_channel(__m128i value, bool is_srgb)
    {
        if (is_srgb)
        {
            // Read one LUT entry per block.
            return _mm_i32gather_ps(c_srgb_to_linear, value, sizeof(float));
        }

        const __m128 channel = _mm_cvtepi32_ps(value);
        return _mm_mul_ps(channel, _mm_set1_ps(inv_255));
    }

    // Extract red or blue and expand 5 bits to 8 bits.
    inline __m128i expand5_x4(__m128i color, uint32_t shift)
    {
        const __m128i shifted = _mm_srl_epi32(color, _mm_cvtsi32_si128(static_cast<int>(shift)));
        const __m128i value = _mm_and_si128(shifted, _mm_set1_epi32(31));
        return _mm_or_si128(_mm_slli_epi32(value, 3), _mm_srli_epi32(value, 2));
    }

    // Extract green and expand 6 bits to 8 bits.
    inline __m128i expand6_x4(__m128i color)
    {
        const __m128i shifted = _mm_srli_epi32(color, 5);
        const __m128i value = _mm_and_si128(shifted, _mm_set1_epi32(63));
        return _mm_or_si128(_mm_slli_epi32(value, 2), _mm_srli_epi32(value, 4));
    }

    // Compute integer division by 3 for values in [0, 765].
    // BC1 interpolation sums cannot exceed 2 * 255 + 255.
    // 683 / 2048 ≈ 1/3 -> (value * 683) >> 11
    inline __m128i div3_x4(__m128i value)
    {
        const __m128i product = _mm_mullo_epi32(value, _mm_set1_epi32(683)); // (value * 683)
        return _mm_srli_epi32(product, 11);                                  // (value * 683) >> 11
    }

    // Build four palettes, with one block per lane.
    inline void get_palette_x4(__m128i c0, __m128i c1, bool is_srgb, Float3x4 palette[4], bool opaque = false)
    {
        const __m128i channel0[3] = {expand5_x4(c0, 11), expand6_x4(c0), expand5_x4(c0, 0)};
        const __m128i channel1[3] = {expand5_x4(c1, 11), expand6_x4(c1), expand5_x4(c1, 0)};

        // Encoding forces four colors; decoding follows endpoint order.
        const __m128i four_color = opaque ? _mm_set1_epi32(-1) : _mm_cmpgt_epi32(c0, c1);

        for (uint32_t channel = 0; channel < 3; ++channel)
        {
            const __m128i a = channel0[channel];
            const __m128i b = channel1[channel];

            const __m128i sum2 = _mm_add_epi32(_mm_slli_epi32(a, 1), b);
            const __m128i sum3 = _mm_add_epi32(a, _mm_slli_epi32(b, 1));

            const __m128i color2_four = div3_x4(sum2);
            const __m128i color3_four = div3_x4(sum3);
            const __m128i midpoint = _mm_srli_epi32(_mm_add_epi32(a, b), 1);

            // Three-color mode uses a midpoint and black for selector 3.
            const __m128i color2 = _mm_blendv_epi8(midpoint, color2_four, four_color);
            const __m128i color3 = _mm_and_si128(color3_four, four_color);

            const __m128 values[4] = {
                decode_channel(a, is_srgb),
                decode_channel(b, is_srgb),
                decode_channel(color2, is_srgb),
                decode_channel(color3, is_srgb)};

            for (uint32_t selector = 0; selector < 4; ++selector)
            {
                if (channel == 0)
                    palette[selector].r = values[selector];
                else if (channel == 1)
                    palette[selector].g = values[selector];
                else
                    palette[selector].b = values[selector];
            }
        }
    }

    // Sum one-bit flags within each 2x2 quadrant for four blocks.
    inline __m128i count_2x2_regions_x4(__m128i flags)
    {
        const __m128i horizontal_mask = _mm_set1_epi32(0x11111111);
        const __m128i vertical_mask = _mm_set1_epi32(0x00ff00ff);

        // Add flags from neighboring columns.
        const __m128i left = _mm_and_si128(flags, horizontal_mask);
        const __m128i right = _mm_and_si128(_mm_srli_epi32(flags, 2), horizontal_mask);
        const __m128i horizontal = _mm_add_epi32(left, right);

        // Add the two rows in each quadrant.
        const __m128i top = _mm_and_si128(horizontal, vertical_mask);
        const __m128i bottom = _mm_and_si128(_mm_srli_epi32(horizontal, 8), vertical_mask);

        return _mm_add_epi32(top, bottom);
    }

    // Count each selector in every quadrant for four blocks.
    inline void selector_region_counts_x4(__m128i indices, __m128i histogram[4])
    {
        const __m128i mask = _mm_set1_epi32(0x55555555);

        // Align both bits of each selector to the same bit positions.
        const __m128i low = _mm_and_si128(indices, mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi32(indices, 1), mask);

        // A selector matches when its two bits match the pattern.
        const __m128i flags0 = _mm_andnot_si128(_mm_or_si128(low, high), mask); // 00
        const __m128i flags1 = _mm_andnot_si128(high, low);  // 01
        const __m128i flags2 = _mm_andnot_si128(low, high);  // 10
        const __m128i flags3 = _mm_and_si128(low, high);     // 11

        histogram[0] = count_2x2_regions_x4(flags0);
        histogram[1] = count_2x2_regions_x4(flags1);
        histogram[2] = count_2x2_regions_x4(flags2);
        histogram[3] = count_2x2_regions_x4(flags3);
    }

    // Compute one quadrant's mean color for four blocks.
    template <uint32_t Shift>
    inline Float3x4 quadrant_mean_x4(const Float3x4 palette[4], const __m128i histogram[4])
    {
        const __m128 zero = _mm_setzero_ps();
        const __m128 scale = _mm_set1_ps(0.25f);
        const __m128i mask = _mm_set1_epi32(15);

        Float3x4 mean = {zero, zero, zero};

        for (uint32_t selector = 0; selector < 4; ++selector)
        {
            // Extract this quadrant's count from each block.
            const __m128i shifted = _mm_srli_epi32(histogram[selector], Shift);
            const __m128i count = _mm_and_si128(shifted, mask);
            const __m128 weight = _mm_mul_ps(_mm_cvtepi32_ps(count), scale);

            mean += palette[selector] * weight;
        }

        return mean;
    }

    // Compute quadrant means for four full 4x4 blocks.
    inline void get_quadrant_means_x4(const Block& b0, const Block& b1, const Block& b2, const Block& b3, bool is_srgb, Float3x4 means[4])
    {
        const __m128i c0 = _mm_setr_epi32(b0.c0, b1.c0, b2.c0, b3.c0);
        const __m128i c1 = _mm_setr_epi32(b0.c1, b1.c1, b2.c1, b3.c1);

        Float3x4 palette[4];
        get_palette_x4(c0, c1, is_srgb, palette);

        const __m128i indices = _mm_setr_epi32(
            static_cast<int>(b0.indices),
            static_cast<int>(b1.indices),
            static_cast<int>(b2.indices),
            static_cast<int>(b3.indices));

        __m128i histogram[4];
        selector_region_counts_x4(indices, histogram);

        means[0] = quadrant_mean_x4<0>(palette, histogram);
        means[1] = quadrant_mean_x4<4>(palette, histogram);
        means[2] = quadrant_mean_x4<16>(palette, histogram);
        means[3] = quadrant_mean_x4<20>(palette, histogram);
    }

    // Choose the nearest palette projection for each sample.
    inline __m128i encode_selectors_x4(const __m128 projections[16], const __m128 palette_projections[4])
    {
        __m128i indices = _mm_setzero_si128();

        for (uint32_t i = 0; i < 16; ++i)
        {
            __m128 best_distance = _mm_set1_ps(1e30f);
            __m128i best_selector = _mm_setzero_si128();

            for (uint32_t selector = 0; selector < 4; ++selector)
            {
                const __m128 delta = _mm_sub_ps(projections[i], palette_projections[selector]);
                const __m128 distance = _mm_mul_ps(delta, delta);
                const __m128 closer = _mm_cmplt_ps(distance, best_distance);

                // Strict comparison keeps the lower selector on ties.
                best_distance = _mm_blendv_ps(best_distance, distance, closer);
                best_selector = _mm_blendv_epi8(best_selector, _mm_set1_epi32(static_cast<int>(selector)), _mm_castps_si128(closer));
            }

            // Place the selector in its row-major texel position.
            const __m128i shift = _mm_cvtsi32_si128(static_cast<int>(2 * texel_map[i]));
            const __m128i packed = _mm_sll_epi32(best_selector, shift);

            indices = _mm_or_si128(indices, packed);
        }

        return indices;
    }

    // Quantize endpoints and pack four BC1 blocks.
    inline void pack_blocks_x4(
        const __m128 projections[16],
        const Float3x4& mean, const Float3x4& axis,
        const Float3x4& p0, const Float3x4& p1,
        bool is_srgb, Block output[4])
    {
        const __m128i endpoint0 = encode_rgb565(p0, is_srgb);
        const __m128i endpoint1 = encode_rgb565(p1, is_srgb);

        // Put the larger endpoint first in each block.
        const __m128i c0 = _mm_max_epi32(endpoint0, endpoint1);
        const __m128i c1 = _mm_min_epi32(endpoint0, endpoint1);

        Float3x4 palette[4];
        get_palette_x4(c0, c1, is_srgb, palette, true);

        __m128 palette_projections[4];
        for (uint32_t selector = 0; selector < 4; ++selector)
        {
            palette_projections[selector] = dot(palette[selector] - mean, axis);
        }

        const __m128i packed_indices = encode_selectors_x4(projections, palette_projections);

        uint32_t colors0[4];
        uint32_t colors1[4];
        uint32_t indices[4];

        _mm_storeu_si128(reinterpret_cast<__m128i*>(colors0), c0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(colors1), c1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(indices), packed_indices);

        for (uint32_t lane = 0; lane < 4; ++lane)
        {
            output[lane] = {static_cast<uint16_t>(colors0[lane]), static_cast<uint16_t>(colors1[lane]), indices[lane]};
        }
    }


    // Encode four blocks from their 16 samples.
    inline void encode_samples_x4(const Float3x4 samples[16], bool is_srgb, Block output[4])
    {
        Float3x4 mean;
        SymMat3x4 covariance;
        compute_child_moments(samples, mean, covariance);

        const __m128 variance = _mm_add_ps(_mm_add_ps(covariance.rr, covariance.gg), covariance.bb);
        const __m128 flat = _mm_cmplt_ps(variance, _mm_set1_ps(1e-10f));

        const Float3x4 axis = compute_principal_axis(covariance);

        __m128 projections[16];
        __m128 minimum = _mm_set1_ps(1e30f);
        __m128 maximum = _mm_set1_ps(-1e30f);

        // Find the projection range independently for each block.
        for (uint32_t i = 0; i < 16; ++i)
        {
            const __m128 projection = dot(samples[i] - mean, axis);
            projections[i] = projection;

            minimum = _mm_min_ps(minimum, projection);
            maximum = _mm_max_ps(maximum, projection);
        }

        // Use the mean for blocks with almost no color variation.
        const Float3x4 p0 = select(mean + axis * minimum, mean, flat);
        const Float3x4 p1 = select(mean + axis * maximum, mean, flat);

        pack_blocks_x4(projections, mean, axis, p0, p1, is_srgb, output);
    }

    // Store eight parent means in their original horizontal order.
    // The destination must contain eight valid texels starting at (x, y).
    inline void store_block_mean_pairs_x4(
        const Float3x4 left_quadrants[4],
        const Float3x4 right_quadrants[4],
        const MeanImage& destination, uint32_t x, uint32_t y)
    {
        const __m128 weight = _mm_set1_ps(0.25f);

        const Float3x4 left = (left_quadrants[0] + left_quadrants[1] + left_quadrants[2] + left_quadrants[3]) * weight;
        const Float3x4 right = (right_quadrants[0] + right_quadrants[1] + right_quadrants[2] + right_quadrants[3]) * weight;
        const size_t index = static_cast<size_t>(y) * destination.width + x;

        // Interleave the left and right parents of each child block.
        _mm_storeu_ps(destination.r + index, _mm_unpacklo_ps(left.r, right.r));
        _mm_storeu_ps(destination.r + index + 4, _mm_unpackhi_ps(left.r, right.r));

        _mm_storeu_ps(destination.g + index, _mm_unpacklo_ps(left.g, right.g));
        _mm_storeu_ps(destination.g + index + 4, _mm_unpackhi_ps(left.g, right.g));

        _mm_storeu_ps(destination.b + index, _mm_unpacklo_ps(left.b, right.b));
        _mm_storeu_ps(destination.b + index + 4, _mm_unpackhi_ps(left.b, right.b));
    }

    // Generate four full child blocks from two rows of eight parent blocks.
    // source_x and source_y are positions in the parent mean image.
    inline void generate_child_blocks_x4(
        const Block* row0, const Block* row1,
        bool is_srgb, const MeanImage& means,
        uint32_t source_x, uint32_t source_y,
        Block output[4])
    {
        Float3x4 samples[16];

        // Each lane gathers the same parent position for one child block.
        get_quadrant_means_x4(row0[0], row0[2], row0[4], row0[6], is_srgb, samples);
        get_quadrant_means_x4(row0[1], row0[3], row0[5], row0[7], is_srgb, samples + 4);
        get_quadrant_means_x4(row1[0], row1[2], row1[4], row1[6], is_srgb, samples + 8);
        get_quadrant_means_x4(row1[1], row1[3], row1[5], row1[7], is_srgb, samples + 12);

        // Preserve parent means for mip level 2.
        store_block_mean_pairs_x4(samples, samples + 4, means, source_x, source_y);
        store_block_mean_pairs_x4(samples + 8, samples + 12, means, source_x, source_y + 1);

        encode_samples_x4(samples, is_srgb, output);
    }

    // Encode four full blocks from the mean image.
    // The source must contain the entire 16x4 texel area.
    inline void generate_child_blocks_from_means_x4(
        const MeanImage& source,
        uint32_t block_x, uint32_t block_y,
        bool is_srgb, Block output[4])
    {
        Float3x4 samples[16];

        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint32_t texel = texel_map[i];
            const uint32_t x = block_x * 4 + (texel & 3u);
            const uint32_t y = block_y * 4 + (texel >> 2);
            const size_t index = static_cast<size_t>(y) * source.width + x;

            // Gather the same local texel from four neighboring blocks.
            samples[i] = {
                _mm_setr_ps(source.r[index], source.r[index + 4], source.r[index + 8], source.r[index + 12]),
                _mm_setr_ps(source.g[index], source.g[index + 4], source.g[index + 8], source.g[index + 12]),
                _mm_setr_ps(source.b[index], source.b[index + 4],source.b[index + 8], source.b[index + 12])};
        }

        encode_samples_x4(samples, is_srgb, output);
    }

} // namespace bc1