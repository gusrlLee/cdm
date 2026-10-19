#include "bc4_cuda.cuh"

// Keep BC4's first kernel load independent of the larger HDR/RGBA module.
static __global__ void generate_base_mip_bc4(const bc4::Block *source, MipLevel source_level, bc4::Block *destination, MipLevel destination_level, MeanImage means)
{
    const uint32_t output = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    const uint32_t output_count = destination_level.block_count_x * destination_level.block_count_y;

    if (output >= output_count)
        return;

    const uint32_t lane = threadIdx.x & 15u;
    const unsigned mask = half_warp_mask();

    const uint32_t bx = output % destination_level.block_count_x;
    const uint32_t by = output / destination_level.block_count_x;
    const uint32_t parent = lane >> 2u;
    const uint32_t quadrant = lane & 3u;

    const uint32_t last_x = source_level.block_count_x - 1;
    const uint32_t last_y = source_level.block_count_y - 1;
    const uint32_t x0 = bx * 2 < last_x ? bx * 2 : last_x;
    const uint32_t y0 = by * 2 < last_y ? by * 2 : last_y;
    const uint32_t x1 = x0 < last_x ? x0 + 1 : x0;
    const uint32_t y1 = y0 < last_y ? y0 + 1 : y0;

    const uint32_t sx = (parent & 1u) ? x1 : x0;
    const uint32_t sy = (parent & 2u) ? y1 : y0;

    const uint32_t source_remaining_x = source_level.width - sx * 4;
    const uint32_t source_remaining_y = source_level.height - sy * 4;
    const uint32_t source_width = source_remaining_x < 4 ? source_remaining_x : 4;
    const uint32_t source_height = source_remaining_y < 4 ? source_remaining_y : 4;

    const size_t source_index = static_cast<size_t>(sy) * source_level.block_count_x + sx;

    float sample = bc4::device_quadrant_mean(source[source_index], quadrant, source_width, source_height);

    const uint32_t remaining_x = destination_level.width - bx * 4;
    const uint32_t remaining_y = destination_level.height - by * 4;
    const uint32_t valid_width = remaining_x < 4 ? remaining_x : 4;
    const uint32_t valid_height = remaining_y < 4 ? remaining_y : 4;

    if (valid_width < 4 || valid_height < 4)
        sample = bc4::repeat_small_sample(sample, valid_width, valid_height);

    const float parent_mean = sum4(sample, mask) * 0.25f;

    const bool unique_parent = parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);

    if (quadrant == 0 && unique_parent)
        means.set(sx, sy, {parent_mean, 0.0f, 0.0f});

    bc4::encode_half_warp(sample, destination + output);
}

static __global__ void generate_mean_mip_bc4(MeanImage source, bc4::Block *destination, MipLevel destination_level, MeanImage next_means)
{
    const uint32_t output = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    const uint32_t output_count = destination_level.block_count_x * destination_level.block_count_y;

    if (output >= output_count)
        return;

    const uint32_t lane = threadIdx.x & 15u;
    const unsigned mask = half_warp_mask();

    const uint32_t bx = output % destination_level.block_count_x;
    const uint32_t by = output / destination_level.block_count_x;
    const uint32_t x = (lane & 1u) | ((lane >> 1u) & 2u);
    const uint32_t y = ((lane >> 1u) & 1u) | ((lane >> 2u) & 2u);

    float sample = source.get(bx * 4 + x, by * 4 + y).r;

    const float mean = sum4(sample, mask) * 0.25f;

    if (next_means.r != nullptr && (lane & 3u) == 0)
    {
        const uint32_t quadrant = lane >> 2u;
        const uint32_t next_x = bx * 2 + (quadrant & 1u);
        const uint32_t next_y = by * 2 + (quadrant >> 1u);

        next_means.set(next_x, next_y, {mean, 0.0f, 0.0f});
    }

    const uint32_t remaining_x = destination_level.width - bx * 4;
    const uint32_t remaining_y = destination_level.height - by * 4;
    const uint32_t valid_width = remaining_x < 4 ? remaining_x : 4;
    const uint32_t valid_height = remaining_y < 4 ? remaining_y : 4;

    if (valid_width < 4 || valid_height < 4)
        sample = bc4::repeat_small_sample(sample, valid_width, valid_height);

    bc4::encode_half_warp(sample, destination + output);
}

namespace bc4
{
    void generate_base_mip_cuda(const Block *source, MipLevel source_level, Block *destination,
                               MipLevel destination_level, MeanImage means, uint32_t groups, uint32_t threads)
    {
        generate_base_mip_bc4<<<groups, threads>>>(source, source_level, destination, destination_level, means);
    }

    void generate_mean_mip_cuda(MeanImage source, Block *destination, MipLevel destination_level,
                               MeanImage next_means, uint32_t groups, uint32_t threads)
    {
        generate_mean_mip_bc4<<<groups, threads>>>(source, destination, destination_level, next_means);
    }
}
