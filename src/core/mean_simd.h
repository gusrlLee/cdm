#pragma once

#include "mean.h"

#include <algorithm>
#include <immintrin.h>

// Average each 2x2 region into one output value.
inline void downsample_channel_simd(const float *src, float *dst, uint32_t src_width, uint32_t src_height,
                                    uint32_t dst_width, uint32_t begin_row, uint32_t end_row)
{
    const __m128 weight = _mm_set1_ps(0.25f); // for box filtering

    for (uint32_t y = begin_row; y < end_row; ++y)
    {
        const uint32_t y0 = y * 2;
        const uint32_t y1 = std::min(y0 + 1, src_height - 1);

        const float *row0 = src + static_cast<size_t>(y0) * src_width;
        const float *row1 = src + static_cast<size_t>(y1) * src_width;
        float *output = dst + static_cast<size_t>(y) * dst_width;

        uint32_t x = 0;

        // Read eight source columns to produce four output values.
        for (; x + 4 <= dst_width; x += 4)
        {
            const uint32_t sx = x * 2;

            const __m128 top0 = _mm_loadu_ps(row0 + sx);
            const __m128 top1 = _mm_loadu_ps(row0 + sx + 4);
            const __m128 bottom0 = _mm_loadu_ps(row1 + sx);
            const __m128 bottom1 = _mm_loadu_ps(row1 + sx + 4);

            const __m128 ll = _mm_shuffle_ps(top0, top1, _MM_SHUFFLE(2, 0, 2, 0));
            const __m128 lr = _mm_shuffle_ps(top0, top1, _MM_SHUFFLE(3, 1, 3, 1));
            const __m128 rl = _mm_shuffle_ps(bottom0, bottom1, _MM_SHUFFLE(2, 0, 2, 0));
            const __m128 rr = _mm_shuffle_ps(bottom0, bottom1, _MM_SHUFFLE(3, 1, 3, 1));

            const __m128 sum = _mm_add_ps(_mm_add_ps(_mm_add_ps(ll, lr), rl), rr);
            _mm_storeu_ps(output + x, _mm_mul_ps(sum, weight));
        }

        // Handle remaining columns and images only one texel wide.
        for (; x < dst_width; ++x)
        {
            const uint32_t x0 = x * 2;
            const uint32_t x1 = std::min(x0 + 1, src_width - 1);

            const float ll = row0[x0];
            const float lr = row0[x1];
            const float rl = row1[x0];
            const float rr = row1[x1];

            output[x] = (ll + lr + rl + rr) * 0.25f;
        }
    }
}

// Downsample all three channels within the assigned rows.
inline void downsample_means_simd(const MeanImage &src, const MeanImage &dst, uint32_t begin_row, uint32_t end_row)
{
    downsample_channel_simd(src.r, dst.r, src.width, src.height, dst.width, begin_row, end_row);
    downsample_channel_simd(src.g, dst.g, src.width, src.height, dst.width, begin_row, end_row);
    downsample_channel_simd(src.b, dst.b, src.width, src.height, dst.width, begin_row, end_row);
}