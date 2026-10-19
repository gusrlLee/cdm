#include "cpu.h"
#include "core/task_dispatcher.h"
#include "core/mean_simd.h"

#include "codecs/bc1.h"
#include "codecs/bc1_simd.h"

#include "codecs/bc4.h"
#include "codecs/bc4_simd.h"
#include "codecs/bc7_simd.h"
#include "codecs/bc5_simd.h"
#include "codecs/bc2_simd.h"
#include "codecs/bc3_simd.h"
#include "codecs/bc6h_simd.h"

#include <iostream>
#include <algorithm>
#include <memory>
#include <new>

static void generate_mip_level_bc1(Image &img, uint32_t level, MeanImage mean_img, uint32_t begin_row, uint32_t end_row, bool use_simd)
{
    const MipLevel &previous = img.mips[level - 1];
    const MipLevel &current = img.mips[level];

    const bc1::Block *src = reinterpret_cast<const bc1::Block *>(img.data + previous.byte_offset);
    bc1::Block *dst = reinterpret_cast<bc1::Block *>(img.data + current.byte_offset);

    for (uint32_t by = begin_row; by < end_row; ++by)
    {
        const uint32_t valid_height = bc1::block_valid_extent(current.height, by);
        uint32_t bx = 0;

        // Process four full blocks at a time.
        if (use_simd && valid_height == 4)
        {
            while (bx + 4 <= current.block_count_x)
            {
                // Keep partial blocks on the scalar path.
                if (bx * 4 + 16 > current.width)
                    break;

                const size_t output = static_cast<size_t>(by) * current.block_count_x + bx;

                if (level >= 2)
                {
                    if (bx * 4 + 16 > mean_img.width || by * 4 + 4 > mean_img.height)
                        break;

                    bc1::generate_child_blocks_from_means_x4(mean_img, bx, by, img.is_srgb, dst + output);
                }
                else
                {
                    const uint32_t x0 = bx * 2;
                    const uint32_t y0 = by * 2;

                    // Require full parent blocks and valid mean storage.
                    if (x0 * 4 + 32 > previous.width || y0 * 4 + 8 > previous.height || x0 + 8 > mean_img.width || y0 + 2 > mean_img.height)
                        break;

                    const bc1::Block *row0 = src + static_cast<size_t>(y0) * previous.block_count_x + x0;
                    const bc1::Block *row1 = row0 + previous.block_count_x;

                    bc1::generate_child_blocks_x4(row0, row1, img.is_srgb, mean_img, x0, y0, dst + output);
                }

                bx += 4;
            }
        }

        // Process remaining blocks, or all blocks in scalar mode.
        for (; bx < current.block_count_x; ++bx)
        {
            const uint32_t valid_width = bc1::block_valid_extent(current.width, bx);
            const size_t output = static_cast<size_t>(by) * current.block_count_x + bx;

            if (level >= 2)
            {
                dst[output] = bc1::generate_child_block_from_means_scalar(mean_img, bx, by, img.is_srgb, valid_width, valid_height);
                continue;
            }

            const uint32_t x0 = bx * 2;
            const uint32_t y0 = by * 2;
            const uint32_t x1 = std::min(x0 + 1, previous.block_count_x - 1);
            const uint32_t y1 = std::min(y0 + 1, previous.block_count_y - 1);

            const bc1::Block *row0 = src + static_cast<size_t>(y0) * previous.block_count_x;
            const bc1::Block *row1 = src + static_cast<size_t>(y1) * previous.block_count_x;

            Float3 parent_means[4];

            dst[output] = bc1::generate_child_block_scalar(
                row0[x0], row0[x1], row1[x0], row1[x1],
                img.is_srgb, parent_means, valid_width, valid_height,
                previous.width - x0 * 4, previous.height - y0 * 4);

            mean_img.set(x0, y0, parent_means[0]);
            mean_img.set(x1, y0, parent_means[1]);
            mean_img.set(x0, y1, parent_means[2]);
            mean_img.set(x1, y1, parent_means[3]);
        }
    }
}

// Generate BC4 blocks within the assigned rows
static void generate_mip_level_bc4(Image &img, uint32_t level, MeanImage mean_img, uint32_t begin_row, uint32_t end_row, bool use_simd)
{
    const MipLevel &previous = img.mips[level - 1];
    const MipLevel &current = img.mips[level];

    const bc4::Block *src = reinterpret_cast<const bc4::Block *>(img.data + previous.byte_offset);
    bc4::Block *dst = reinterpret_cast<bc4::Block *>(img.data + current.byte_offset);

    for (uint32_t by = begin_row; by < end_row; ++by)
    {
        const uint32_t valid_height = std::min(4u, current.height - by * 4);

        uint32_t bx = 0;

        if (use_simd && valid_height == 4)
        {
            while (bx + 4 <= current.block_count_x)
            {
                if (bx * 4 + 16 > current.width)
                    break;

                const size_t output = static_cast<size_t>(by) * current.block_count_x + bx;

                if (level >= 2)
                {
                    if (bx * 4 + 16 > mean_img.width || by * 4 + 4 > mean_img.height)
                        break;

                    bc4::generate_child_blocks_from_means_x4(mean_img, bx, by, dst + output);
                }
                else
                {
                    const uint32_t x0 = bx * 2;
                    const uint32_t y0 = by * 2;

                    if (x0 * 4 + 32 > previous.width || y0 * 4 + 8 > previous.height || x0 + 8 > mean_img.width || y0 + 2 > mean_img.height)
                        break;

                    const bc4::Block *row0 = src + static_cast<size_t>(y0) * previous.block_count_x + x0;
                    const bc4::Block *row1 = row0 + previous.block_count_x;
                    bc4::generate_child_blocks_x4(row0, row1, mean_img, x0, y0, dst + output);
                }

                bx += 4;
            }
        }

        for (; bx < current.block_count_x; ++bx)
        {
            const uint32_t valid_width =
                std::min(4u, current.width - bx * 4);
            const size_t output =
                static_cast<size_t>(by) * current.block_count_x + bx;

            if (level >= 2)
            {
                dst[output] = bc4::generate_child_block_from_means_scalar(mean_img, bx, by, valid_width, valid_height);
                continue;
            }

            const uint32_t x0 = bx * 2;
            const uint32_t y0 = by * 2;
            const uint32_t x1 = std::min(x0 + 1, previous.block_count_x - 1);
            const uint32_t y1 = std::min(y0 + 1, previous.block_count_y - 1);

            const bc4::Block *row0 = src + static_cast<size_t>(y0) * previous.block_count_x;
            const bc4::Block *row1 = src + static_cast<size_t>(y1) * previous.block_count_x;

            float parent_means[4];

            dst[output] = bc4::generate_child_block_scalar(
                row0[x0], row0[x1], row1[x0], row1[x1],
                parent_means, valid_width, valid_height,
                previous.width - x0 * 4, previous.height - y0 * 4);

            mean_img.set(x0, y0, {parent_means[0], 0.0f, 0.0f});
            mean_img.set(x1, y0, {parent_means[1], 0.0f, 0.0f});
            mean_img.set(x0, y1, {parent_means[2], 0.0f, 0.0f});
            mean_img.set(x1, y1, {parent_means[3], 0.0f, 0.0f});
        }
    }
}

static void generate_mip_level_bc5_bc6h(Image &img, uint32_t level, const MeanImage &means,
                                   uint32_t begin_row, uint32_t end_row, bool use_simd)
{
    const MipLevel &previous = img.mips[level - 1];
    const MipLevel &current = img.mips[level];
    const auto *src = img.data + previous.byte_offset;
    auto *dst = img.data + current.byte_offset;
    for (uint32_t by = begin_row; by < end_row; ++by)
    {
        for (uint32_t bx = 0; bx < current.block_count_x;)
        {
            const uint32_t height = std::min(4u, current.height - by * 4);
            const bool vectorize = use_simd && height == 4 && bx * 4 + 16 <= current.width;
            const uint32_t lanes = vectorize ? 4 : 1;
            Float3 samples[4][16];
            for (uint32_t lane = 0; lane < lanes; ++lane)
            {
                const uint32_t x = bx + lane;
                const uint32_t width = std::min(4u, current.width - x * 4);
                if (level >= 2)
                {
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        const uint32_t texel = bc1::texel_map[i];
                        samples[lane][i] = means.get(x * 4 + (texel % 4) % width, by * 4 + (texel / 4) % height);
                    }
                    continue;
                }
                const uint32_t x0 = x * 2, y0 = by * 2;
                for (uint32_t parent = 0; parent < 4; ++parent)
                {
                    const uint32_t sx = std::min(x0 + (parent & 1u), previous.block_count_x - 1);
                    const uint32_t sy = std::min(y0 + (parent >> 1u), previous.block_count_y - 1);
                    const uint8_t *block = src + (static_cast<size_t>(sy) * previous.block_count_x + sx) * 16;
                    const uint32_t pw = std::min(4u, previous.width - sx * 4);
                    const uint32_t ph = std::min(4u, previous.height - sy * 4);
                    if (img.format == Format::BC5_UNORM || img.format == Format::BC5_SNORM)
                        bc5::get_quadrant_means(*reinterpret_cast<const bc5::Block *>(block), samples[lane] + parent * 4, pw, ph, img.format == Format::BC5_SNORM);
                    else
                        bc6h::get_quadrant_means(*reinterpret_cast<const bc6h::Block *>(block), samples[lane] + parent * 4, pw, ph);
                }
                bc1::repeat_small_samples(samples[lane], width, height);
                for (uint32_t parent = 0; parent < 4; ++parent)
                {
                    const Float3 *q = samples[lane] + parent * 4;
                    means.set(x0 + (parent & 1u), y0 + (parent >> 1u), (q[0] + q[1] + q[2] + q[3]) * 0.25f);
                }
            }
            const size_t output = static_cast<size_t>(by) * current.block_count_x + bx;
            if (vectorize)
            {
                Float3x4 packed[16];
                for (uint32_t i = 0; i < 16; ++i)
                    packed[i] = {_mm_setr_ps(samples[0][i].r, samples[1][i].r, samples[2][i].r, samples[3][i].r),
                                 _mm_setr_ps(samples[0][i].g, samples[1][i].g, samples[2][i].g, samples[3][i].g),
                                 _mm_setr_ps(samples[0][i].b, samples[1][i].b, samples[2][i].b, samples[3][i].b)};
                if (img.format == Format::BC5_UNORM || img.format == Format::BC5_SNORM)
                    bc5::encode_samples_x4(packed, reinterpret_cast<bc5::Block *>(dst) + output, img.format == Format::BC5_SNORM);
                else
                    bc6h::encode_samples_x4(packed, reinterpret_cast<bc6h::Block *>(dst) + output);
            }
            else if (img.format == Format::BC5_UNORM || img.format == Format::BC5_SNORM)
                reinterpret_cast<bc5::Block *>(dst)[output] = bc5::encode_samples_scalar(samples[0], img.format == Format::BC5_SNORM);
            else
                reinterpret_cast<bc6h::Block *>(dst)[output] = bc6h::encode_samples_scalar(samples[0]);
            bx += lanes;
        }
    }
}

static void generate_mip_level_bc2_bc3(Image &img, uint32_t level, const MeanImage4 &means,
                                   uint32_t begin_row, uint32_t end_row, bool use_simd)
{
    const MipLevel &previous = img.mips[level - 1];
    const MipLevel &current = img.mips[level];
    const auto *src = img.data + previous.byte_offset;
    auto *dst = img.data + current.byte_offset;
    for (uint32_t by = begin_row; by < end_row; ++by)
    {
        for (uint32_t bx = 0; bx < current.block_count_x;)
        {
            const uint32_t height = std::min(4u, current.height - by * 4);
            const bool vectorize = use_simd && height == 4 && bx * 4 + 16 <= current.width;
            const uint32_t lanes = vectorize ? 4 : 1;
            Float4 samples[4][16];
            for (uint32_t lane = 0; lane < lanes; ++lane)
            {
                const uint32_t x = bx + lane;
                const uint32_t width = std::min(4u, current.width - x * 4);
                if (level >= 2)
                {
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        const uint32_t texel = bc1::texel_map[i];
                        samples[lane][i] = means.get(x * 4 + (texel % 4) % width, by * 4 + (texel / 4) % height);
                    }
                    continue;
                }
                const uint32_t x0 = x * 2, y0 = by * 2;
                for (uint32_t parent = 0; parent < 4; ++parent)
                {
                    const uint32_t sx = std::min(x0 + (parent & 1u), previous.block_count_x - 1);
                    const uint32_t sy = std::min(y0 + (parent >> 1u), previous.block_count_y - 1);
                    const uint8_t *block = src + (static_cast<size_t>(sy) * previous.block_count_x + sx) * 16;
                    const uint32_t pw = std::min(4u, previous.width - sx * 4);
                    const uint32_t ph = std::min(4u, previous.height - sy * 4);
                    if (img.format == Format::BC2)
                        bc2::get_quadrant_means(*reinterpret_cast<const bc2::Block *>(block), img.is_srgb, samples[lane] + parent * 4, pw, ph);
                    else
                        bc3::get_quadrant_means(*reinterpret_cast<const bc3::Block *>(block), img.is_srgb, samples[lane] + parent * 4, pw, ph);
                }
                Float4 original[16];
                std::copy_n(samples[lane], 16, original);
                for (uint32_t i = 0; i < 16; ++i)
                {
                    const uint32_t texel = bc1::texel_map[i];
                    const uint32_t local = ((texel / 4) % height) * 4 + (texel % 4) % width;
                    samples[lane][i] = original[bc1::texel_map[local]];
                }
                for (uint32_t parent = 0; parent < 4; ++parent)
                {
                    const Float4 *q = samples[lane] + parent * 4;
                    means.set(x0 + (parent & 1u), y0 + (parent >> 1u), (q[0] + q[1] + q[2] + q[3]) * 0.25f);
                }
            }
            const size_t output = static_cast<size_t>(by) * current.block_count_x + bx;
            if (img.format == Format::BC2)
            {
                auto *blocks = reinterpret_cast<bc2::Block *>(dst) + output;
                if (vectorize) bc2::encode_samples_x4(samples, img.is_srgb, blocks);
                else *blocks = bc2::encode_samples_scalar(samples[0], img.is_srgb);
            }
            else
            {
                auto *blocks = reinterpret_cast<bc3::Block *>(dst) + output;
                if (vectorize) bc3::encode_samples_x4(samples, img.is_srgb, blocks);
                else *blocks = bc3::encode_samples_scalar(samples[0], img.is_srgb);
            }
            bx += lanes;
        }
    }
}

static void generate_mip_level_bc7(Image &img, uint32_t level, const MeanImage4 &means,
                                    uint32_t begin_row, uint32_t end_row, bool use_simd)
{
    const MipLevel &previous = img.mips[level - 1];
    const MipLevel &current = img.mips[level];
    const bc7::Block *src = reinterpret_cast<const bc7::Block *>(img.data + previous.byte_offset);
    bc7::Block *dst = reinterpret_cast<bc7::Block *>(img.data + current.byte_offset);

    for (uint32_t by = begin_row; by < end_row; ++by)
    {
        const uint32_t valid_height = std::min(4u, current.height - by * 4);
        for (uint32_t bx = 0; bx < current.block_count_x; ++bx)
        {
            const uint32_t valid_width = std::min(4u, current.width - bx * 4);
            const size_t output = static_cast<size_t>(by) * current.block_count_x + bx;
            if (level >= 2)
            {
                if (use_simd)
                {
                    dst[output] = bc7::generate_child_block_from_means_simd(means, bx, by, valid_width, valid_height);
                }
                else
                {
                    dst[output] = bc7::generate_child_block_from_means_scalar(means, bx, by, valid_width, valid_height);
                }
                continue;
            }

            const uint32_t x0 = bx * 2;
            const uint32_t y0 = by * 2;
            const uint32_t x1 = std::min(x0 + 1, previous.block_count_x - 1);
            const uint32_t y1 = std::min(y0 + 1, previous.block_count_y - 1);
            const bc7::Block *row0 = src + static_cast<size_t>(y0) * previous.block_count_x;
            const bc7::Block *row1 = src + static_cast<size_t>(y1) * previous.block_count_x;
            Float4 parent_means[4];
            if (use_simd)
            {
                dst[output] = bc7::generate_child_block_simd(
                    row0[x0], row0[x1], row1[x0], row1[x1], parent_means,
                    valid_width, valid_height, previous.width - x0 * 4, previous.height - y0 * 4);
            }
            else
            {
                dst[output] = bc7::generate_child_block_scalar(
                    row0[x0], row0[x1], row1[x0], row1[x1], parent_means,
                    valid_width, valid_height, previous.width - x0 * 4, previous.height - y0 * 4);
            }
            means.set(x0, y0, parent_means[0]);
            means.set(x1, y0, parent_means[1]);
            means.set(x0, y1, parent_means[2]);
            means.set(x1, y1, parent_means[3]);
        }
    }
}

static bool generate_mipmaps_rgba_cpu(Image &img, bool use_simd)
{
    if (img.mip_count == 1)
    {
        return true;
    }
    const uint32_t bw = img.mips[0].block_count_x;
    const uint32_t bh = img.mips[0].block_count_y;
    const uint32_t sw = (bw + 1) / 2;
    const uint32_t sh = (bh + 1) / 2;
    const size_t base_count = static_cast<size_t>(bw) * bh;
    const size_t scratch_count = static_cast<size_t>(sw) * sh;
    std::unique_ptr<float[]> base(new (std::nothrow) float[base_count * 4]);
    std::unique_ptr<float[]> scratch_data(new (std::nothrow) float[scratch_count * 4]);
    if (!base || !scratch_data)
    {
        return false;
    }

    const uint32_t width = img.mip_count > 2 ? img.mips[2].width : 1;
    const uint32_t height = img.mip_count > 2 ? img.mips[2].height : 1;
    MeanImage4 means{base.get(), base.get() + base_count, base.get() + base_count * 2,
                     base.get() + base_count * 3, width, height};
    MeanImage4 scratch{scratch_data.get(), scratch_data.get() + scratch_count,
                       scratch_data.get() + scratch_count * 2,
                       scratch_data.get() + scratch_count * 3, sw, sh};

    static TaskDispatcher dispatcher;
    for (uint32_t level = 1; level < img.mip_count; ++level)
    {
        const MipLevel &current = img.mips[level];
        if (level >= 3)
        {
            scratch.width = current.width;
            scratch.height = current.height;
            const size_t pixels = static_cast<size_t>(current.width) * current.height;
            dispatcher.parallel_rows(current.height, pixels,
                                     [&](uint32_t begin, uint32_t end)
                                     {
                                         downsample_means(means, scratch, begin, end);
                                     });
            std::swap(means, scratch);
        }
        const size_t blocks = static_cast<size_t>(current.block_count_x) * current.block_count_y;
        dispatcher.parallel_rows(current.block_count_y, blocks,
                                 [&](uint32_t begin, uint32_t end)
                                 {
                                     if (img.format == Format::BC2 || img.format == Format::BC3)
                                         generate_mip_level_bc2_bc3(img, level, means, begin, end, use_simd);
                                     else
                                         generate_mip_level_bc7(img, level, means, begin, end, use_simd);
                                 });
    }
    return true;
}

bool generate_mipmaps_cpu(Image &img, bool use_simd)
{
    if (!img.data || img.mip_count == 0)
        return false;

    // Validate the format before allocating mean buffers.
    switch (img.format)
    {
    case Format::BC1:
        break;

    case Format::BC4_UNORM:
    case Format::BC5_UNORM:
    case Format::BC5_SNORM:
    case Format::BC6H_UF16:
        break;

    case Format::BC2:
    case Format::BC3:
        return generate_mipmaps_rgba_cpu(img, use_simd);

    case Format::BC7:
        return img.is_srgb && generate_mipmaps_rgba_cpu(img, use_simd);

    default:
        return false;
    }

    // No additional mip levels need to be generated.
    if (img.mip_count == 1)
        return true;

    const uint32_t bw = img.mips[0].block_count_x;
    const uint32_t bh = img.mips[0].block_count_y;
    const uint32_t sw = (bw + 1) / 2;
    const uint32_t sh = (bh + 1) / 2;

    const size_t base_count = static_cast<size_t>(bw) * bh;
    const size_t scratch_count = static_cast<size_t>(sw) * sh;

    std::unique_ptr<float[]> base(new (std::nothrow) float[base_count * 3]);
    std::unique_ptr<float[]> scratch_data(new (std::nothrow) float[scratch_count * 3]);

    if (!base || !scratch_data)
        return false;

    // Store RGB channels seperately in each buffer.
    const uint32_t width = img.mip_count > 2 ? img.mips[2].width : 1;
    const uint32_t height = img.mip_count > 2 ? img.mips[2].height : 1;

    MeanImage means{base.get(), base.get() + base_count, base.get() + base_count * 2, width, height};
    MeanImage scratch{scratch_data.get(), scratch_data.get() + scratch_count, scratch_data.get() + scratch_count * 2, sw, sh};

    static TaskDispatcher dispatcher;
    for (uint32_t level = 1; level < img.mip_count; ++level)
    {
        const MipLevel &current = img.mips[level];

        if (level >= 3)
        {
            scratch.width = current.width;
            scratch.height = current.height;

            const size_t pixels = static_cast<size_t>(scratch.width) * scratch.height;
            dispatcher.parallel_rows(scratch.height, pixels,
                                     [&](uint32_t begin, uint32_t end)
                                     {
                                         if (use_simd)
                                             downsample_means_simd(means, scratch, begin, end);
                                         else
                                             downsample_means(means, scratch, begin, end);
                                     });

            std::swap(means, scratch);
        }

        const size_t blocks = static_cast<size_t>(current.block_count_x) * current.block_count_y;

        dispatcher.parallel_rows(
            current.block_count_y, blocks,
            [&](uint32_t begin, uint32_t end)
            {
                // Dispatch once per row task, outside the block loop.
                switch (img.format)
                {
                case Format::BC1:
                    generate_mip_level_bc1(img, level, means, begin, end, use_simd);
                    break;

                case Format::BC4_UNORM:
                    generate_mip_level_bc4(img, level, means, begin, end, use_simd);
                    break;

                case Format::BC5_UNORM:
                case Format::BC5_SNORM:
                case Format::BC6H_UF16:
                    generate_mip_level_bc5_bc6h(img, level, means, begin, end, use_simd);
                    break;

                default:
                    break; // Unsupported formats were rejected above.
                }
            });
    }

    return true;
}