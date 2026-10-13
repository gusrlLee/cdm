#pragma once

#include "math.h"

#include <cstddef>
#include <cstdint>

struct MeanImage
{
    float* r = nullptr;
    float* g = nullptr;
    float* b = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

    // Clamp reads to the valid image area.
    CDM_INLINE Float3 get(uint32_t x, uint32_t y) const
    {
        x = x < width ? x : width - 1;
        y = y < height ? y : height - 1;

        const size_t index = static_cast<size_t>(y) * width + x;
        return {r[index], g[index], b[index]};
    }

    // Ignore writes outside the valid image area.
    CDM_INLINE void set(uint32_t x, uint32_t y, const Float3& color) const
    {
        if (x >= width || y >= height)
        {
            return;
        }

        const size_t index = static_cast<size_t>(y) * width + x;
        r[index] = color.r;
        g[index] = color.g;
        b[index] = color.b;
    }
};

inline void downsample_means(const MeanImage& src, const MeanImage &dst, uint32_t begin_row, uint32_t end_row)
{
    for (uint32_t y = begin_row; y < end_row; ++y)
    {
        for (uint32_t x = 0; x < dst.width; ++x)
        {
            const uint32_t sx = x * 2;
            const uint32_t sy = y * 2;

            const Float3 ll = src.get(sx, sy);
            const Float3 lr = src.get(sx + 1, sy);
            const Float3 rl = src.get(sx, sy + 1);
            const Float3 rr = src.get(sx + 1, sy + 1);

            dst.set(x, y, (ll + lr + rl + rr) * 0.25f);
        }
    }
}