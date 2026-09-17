#include "bc1.h"
#include "bc6h.h"
#include "bc7.h"
#include "mip.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

using Color = Float3;
using MeanImage = MeanImage3;
using Color4 = Float4;
static_assert(sizeof(Block64) == 8);

static_assert(texel_index(0) == 0 && texel_index(3) == 5 && texel_index(10) == 12 && texel_index(15) == 15);

// ---------------------------------------------------------------------
// Shared CUDA utilities are defined in base.h.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Kernels only map blocks to half-warps and invoke codec operations.
// ---------------------------------------------------------------------

// Level 1: derive child blocks directly from the base mip's BC1 blocks, and
// record the mean pyramid's base level along the way.
template <bool Srgb>
static __global__ void generate_base_mip(const Block64 *source, uint32_t source_width, uint32_t source_height,
                                         Block64 *destination, uint32_t destination_width, uint32_t destination_height,
                                         MeanImage means, uint32_t valid_width, uint32_t valid_height)
{
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    uint32_t output_count = destination_width * destination_height;
    if (output_index >= output_count)
        return;

    uint32_t sample_index = threadIdx.x & 15u;

    unsigned mask = half_warp_mask();
    uint32_t output_x = sample_index == 0 ? output_index % destination_width : 0;
    uint32_t output_y = sample_index == 0 ? output_index / destination_width : 0;

    output_x = __shfl_sync(mask, output_x, 0, 16);
    output_y = __shfl_sync(mask, output_y, 0, 16);

    uint32_t parent = sample_index >> 2;
    uint32_t quadrant = sample_index & 3u;
    uint32_t x0 = clamp_index(output_x * 2, source_width);
    uint32_t x1 = clamp_index(x0 + 1, source_width);
    uint32_t y0 = clamp_index(output_y * 2, source_height);
    uint32_t y1 = clamp_index(y0 + 1, source_height);
    uint32_t source_x = (parent & 1u) ? x1 : x0;
    uint32_t source_y = (parent & 2u) ? y1 : y0;

    Color sample = bc1::device_quadrant_mean<Srgb>(source[source_y * source_width + source_x], quadrant);
    Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};

    bool unique_parent =
        parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
    if (quadrant == 0 && unique_parent)
    {
        size_t index = (size_t)source_y * means.width + source_x;
        means.r[index] = parent_mean.r;
        means.g[index] = parent_mean.g;
        means.b[index] = parent_mean.b;
    }
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t lane = repeat_small_sample(sample_index, valid_width, valid_height);
        sample = {__shfl_sync(mask, sample.r, lane, 16), __shfl_sync(mask, sample.g, lane, 16),
                  __shfl_sync(mask, sample.b, lane, 16)};
    }
    bc1::encode_half_warp<Srgb>(sample, sample_index, destination + output_index);
}

// Later levels read the mean pyramid and emit the next mean level in the same
// pass.
template <bool Srgb>
static __global__ void generate_mean_mip(MeanImage source, Block64 *destination, uint32_t destination_width,
                                         uint32_t destination_height, MeanImage next_means, uint32_t valid_width,
                                         uint32_t valid_height)
{
    uint32_t sample_index = threadIdx.x & 15u;
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    uint32_t output_count = destination_width * destination_height;

    if (output_index >= output_count)
        return;

    unsigned mask = half_warp_mask();
    uint32_t local = texel_index(sample_index);
    uint32_t output_x = sample_index == 0 ? output_index % destination_width : 0;
    uint32_t output_y = sample_index == 0 ? output_index / destination_width : 0;

    output_x = __shfl_sync(mask, output_x, 0, 16);
    output_y = __shfl_sync(mask, output_y, 0, 16);

    uint32_t x = clamp_index(output_x * 4 + (local & 3u), source.width);
    uint32_t y = clamp_index(output_y * 4 + (local >> 2), source.height);
    size_t index = (size_t)y * source.width + x;
    Color sample = {source.r[index], source.g[index], source.b[index]};
    Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};

    if (next_means.r != nullptr && (sample_index & 3u) == 0)
    {
        uint32_t quadrant = sample_index >> 2;
        uint32_t next_x = output_x * 2 + (quadrant & 1u);
        uint32_t next_y = output_y * 2 + (quadrant >> 1);
        if (next_x < next_means.width && next_y < next_means.height)
        {
            size_t next_index = (size_t)next_y * next_means.width + next_x;
            next_means.r[next_index] = parent_mean.r;
            next_means.g[next_index] = parent_mean.g;
            next_means.b[next_index] = parent_mean.b;
        }
    }

    // Repeat only encoder inputs; the next linear means above retain their
    // original box calculation.
    if (valid_width < 4 || valid_height < 4)
    {
        x = clamp_index(repeat_small_coordinate(output_x * 4 + (local & 3u), valid_width), source.width);
        y = clamp_index(repeat_small_coordinate(output_y * 4 + (local >> 2), valid_height), source.height);
        index = (size_t)y * source.width + x;
        sample = {source.r[index], source.g[index], source.b[index]};
    }
    bc1::encode_half_warp<Srgb>(sample, sample_index, destination + output_index);
}

static __global__ void generate_bc6h_base_mip(const bc6h::Block *source, uint32_t source_width, uint32_t source_height,
                                              bc6h::Block *destination, uint32_t destination_width,
                                              uint32_t destination_height, MeanImage means, uint32_t valid_width,
                                              uint32_t valid_height)
{
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    if (output_index >= destination_width * destination_height)
        return;
    uint32_t lane = threadIdx.x & 15u;
    unsigned mask = half_warp_mask();
    uint32_t ox = lane == 0 ? output_index % destination_width : 0;
    uint32_t oy = lane == 0 ? output_index / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16);
    oy = __shfl_sync(mask, oy, 0, 16);
    uint32_t parent = lane >> 2, quadrant = lane & 3u;
    uint32_t x0 = clamp_index(ox * 2, source_width), x1 = clamp_index(x0 + 1, source_width);
    uint32_t y0 = clamp_index(oy * 2, source_height), y1 = clamp_index(y0 + 1, source_height);
    uint32_t sx = parent & 1u ? x1 : x0, sy = parent & 2u ? y1 : y0;
    Color sample = bc6h::device_quadrant_mean(source[sy * source_width + sx], quadrant);
    Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};
    bool unique =
        parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
    if (quadrant == 0 && unique)
    {
        size_t i = size_t(sy) * means.width + sx;
        means.r[i] = parent_mean.r;
        means.g[i] = parent_mean.g;
        means.b[i] = parent_mean.b;
    }
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t source_lane = repeat_small_sample(lane, valid_width, valid_height);
        sample = {__shfl_sync(mask, sample.r, source_lane, 16), __shfl_sync(mask, sample.g, source_lane, 16),
                  __shfl_sync(mask, sample.b, source_lane, 16)};
    }
    bc6h::encode_half_warp(sample, lane, destination + output_index);
}

static __global__ void generate_bc6h_mean_mip(MeanImage source, bc6h::Block *destination, uint32_t destination_width,
                                              uint32_t destination_height, MeanImage next_means, uint32_t valid_width,
                                              uint32_t valid_height)
{
    uint32_t lane = threadIdx.x & 15u;
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    if (output_index >= destination_width * destination_height)
        return;
    unsigned mask = half_warp_mask();
    uint32_t local = texel_index(lane);
    uint32_t ox = lane == 0 ? output_index % destination_width : 0;
    uint32_t oy = lane == 0 ? output_index / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16);
    oy = __shfl_sync(mask, oy, 0, 16);
    uint32_t x = clamp_index(ox * 4 + (local & 3u), source.width);
    uint32_t y = clamp_index(oy * 4 + (local >> 2), source.height);
    size_t index = size_t(y) * source.width + x;
    Color sample = {source.r[index], source.g[index], source.b[index]};
    Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};
    if (next_means.r && (lane & 3u) == 0)
    {
        uint32_t quadrant = lane >> 2, nx = ox * 2 + (quadrant & 1u), ny = oy * 2 + (quadrant >> 1);
        if (nx < next_means.width && ny < next_means.height)
        {
            size_t ni = size_t(ny) * next_means.width + nx;
            next_means.r[ni] = parent_mean.r;
            next_means.g[ni] = parent_mean.g;
            next_means.b[ni] = parent_mean.b;
        }
    }
    if (valid_width < 4 || valid_height < 4)
    {
        x = clamp_index(repeat_small_coordinate(ox * 4 + (local & 3u), valid_width), source.width);
        y = clamp_index(repeat_small_coordinate(oy * 4 + (local >> 2), valid_height), source.height);
        index = size_t(y) * source.width + x;
        sample = {source.r[index], source.g[index], source.b[index]};
    }
    bc6h::encode_half_warp(sample, lane, destination + output_index);
}

template <bool Srgb>
static __global__ void generate_bc7_base(const bc7::Block *source, uint32_t source_width, uint32_t source_height,
                                         bc7::Block *destination, uint32_t destination_width,
                                         uint32_t destination_height, bc7::MeanImage means, uint32_t valid_width,
                                         uint32_t valid_height)
{
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4), lane = threadIdx.x & 15u;
    if (output_index >= destination_width * destination_height)
        return;
    unsigned mask = half_warp_mask();
    uint32_t ox = lane == 0 ? output_index % destination_width : 0,
             oy = lane == 0 ? output_index / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16);
    oy = __shfl_sync(mask, oy, 0, 16);
    uint32_t parent = lane >> 2, quadrant = lane & 3;
    uint32_t x0 = clamp_index(ox * 2, source_width), x1 = clamp_index(x0 + 1, source_width),
             y0 = clamp_index(oy * 2, source_height), y1 = clamp_index(y0 + 1, source_height);
    uint32_t sx = (parent & 1) ? x1 : x0, sy = (parent & 2) ? y1 : y0;
    Color4 sample = bc7::device_quadrant_mean<Srgb>(source[sy * source_width + sx], quadrant);
    Color4 pm = {sum4(sample.r, mask) * .25f, sum4(sample.g, mask) * .25f, sum4(sample.b, mask) * .25f,
                 sum4(sample.a, mask) * .25f};
    bool unique =
        parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
    if (quadrant == 0 && unique)
    {
        size_t i = size_t(sy) * means.width + sx;
        means.r[i] = pm.r;
        means.g[i] = pm.g;
        means.b[i] = pm.b;
        means.a[i] = pm.a;
    }
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t l = repeat_small_sample(lane, valid_width, valid_height);
        sample = {__shfl_sync(mask, sample.r, l, 16), __shfl_sync(mask, sample.g, l, 16),
                  __shfl_sync(mask, sample.b, l, 16), __shfl_sync(mask, sample.a, l, 16)};
    }
    bc7::encode_half_warp<Srgb>(sample, lane, destination + output_index);
}

template <bool Srgb>
static __global__ void generate_bc7_mean(bc7::MeanImage source, bc7::Block *destination, uint32_t destination_width,
                                         uint32_t destination_height, bc7::MeanImage next, uint32_t valid_width,
                                         uint32_t valid_height)
{
    uint32_t oi = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4), lane = threadIdx.x & 15u;
    if (oi >= destination_width * destination_height)
        return;
    unsigned mask = half_warp_mask();
    uint32_t ox = lane == 0 ? oi % destination_width : 0, oy = lane == 0 ? oi / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16);
    oy = __shfl_sync(mask, oy, 0, 16);
    uint32_t t = texel_index(lane);
    uint32_t x = clamp_index(ox * 4 + (t & 3), source.width), y = clamp_index(oy * 4 + (t >> 2), source.height);
    size_t i = size_t(y) * source.width + x;
    Color4 sample = {source.r[i], source.g[i], source.b[i], source.a[i]};
    Color4 pm = {sum4(sample.r, mask) * .25f, sum4(sample.g, mask) * .25f, sum4(sample.b, mask) * .25f,
                 sum4(sample.a, mask) * .25f};
    if (next.r && (lane & 3) == 0)
    {
        uint32_t q = lane >> 2, nx = ox * 2 + (q & 1), ny = oy * 2 + (q >> 1);
        if (nx < next.width && ny < next.height)
        {
            size_t n = size_t(ny) * next.width + nx;
            next.r[n] = pm.r;
            next.g[n] = pm.g;
            next.b[n] = pm.b;
            next.a[n] = pm.a;
        }
    }
    if (valid_width < 4 || valid_height < 4)
    {
        x = clamp_index(repeat_small_coordinate(ox * 4 + (t & 3), valid_width), source.width);
        y = clamp_index(repeat_small_coordinate(oy * 4 + (t >> 2), valid_height), source.height);
        i = size_t(y) * source.width + x;
        sample = {source.r[i], source.g[i], source.b[i], source.a[i]};
    }
    bc7::encode_half_warp<Srgb>(sample, lane, destination + oi);
}

// ---------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------

template <typename T> class DeviceBuffer
{
  public:
    ~DeviceBuffer()
    {
        if (data_)
            cudaFree(data_);
    }
    bool allocate(size_t count)
    {
        if (count <= capacity_)
            return true;
        if (data_)
            cudaFree(data_);
        data_ = nullptr;
        capacity_ = 0;
        if (cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)) != cudaSuccess)
            return false;
        capacity_ = count;
        return true;
    }
    T *get() const
    {
        return data_;
    }

  private:
    T *data_ = nullptr;
    size_t capacity_ = 0;
};

static std::array<float, 256> make_srgb_decode_table()
{
    std::array<float, 256> values{};
    for (size_t i = 0; i < values.size(); ++i)
    {
        float s = float(i) * (1.0f / 255.0f);
        values[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    }
    return values;
}

template <size_t MaxValue> static std::array<float, MaxValue> make_srgb_quantize_thresholds()
{
    std::array<float, MaxValue> values{};
    for (size_t i = 0; i < values.size(); ++i)
    {
        constexpr uint32_t bits = MaxValue == 31 ? 5 : 6;
        uint32_t a = (uint32_t(i) << (8 - bits)) | (uint32_t(i) >> (2 * bits - 8));
        uint32_t b = ((uint32_t(i) + 1) << (8 - bits)) | ((uint32_t(i) + 1) >> (2 * bits - 8));
        float s = float(a + b) * (1.0f / 510.0f);
        values[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    }
    return values;
}

static bool initialize_tables()
{
    static bool initialized = false;
    if (initialized)
        return true;
    static const std::array<float, 256> srgb = make_srgb_decode_table();
    static const std::array<float, 31> threshold31 = make_srgb_quantize_thresholds<31>();
    static const std::array<float, 63> threshold63 = make_srgb_quantize_thresholds<63>();
    initialized = cudaMemcpyToSymbol(g_srgb_to_linear, srgb.data(), sizeof(srgb)) == cudaSuccess &&
                  cudaMemcpyToSymbol(g_threshold31, threshold31.data(), sizeof(threshold31)) == cudaSuccess &&
                  cudaMemcpyToSymbol(g_threshold63, threshold63.data(), sizeof(threshold63)) == cudaSuccess;
    return initialized;
}

struct CudaWorkspace
{
    DeviceBuffer<uint8_t> data;
    DeviceBuffer<float> base;
    DeviceBuffer<float> scratch;

    bool prepare(const Image *image)
    {
        const size_t base_count = (size_t)image->mips[0].block_count_x * image->mips[0].block_count_y;
        const size_t scratch_count =
            (size_t)((image->mips[0].block_count_x + 1) / 2) * ((image->mips[0].block_count_y + 1) / 2);
        const size_t channels = image->format == Format::BC7 ? 4 : 3;
        return data.allocate(image->data_size) && base.allocate(base_count * channels) &&
               scratch.allocate(scratch_count * channels);
    }
};

static CudaWorkspace g_workspace;

// ---------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------

template <bool Srgb>
static bool generate_cuda_impl(Image *image, uint8_t *device_data, MeanImage means, MeanImage scratch)
{
    constexpr uint32_t threads = 256;
    for (uint32_t level = 1; level < image->mip_count; ++level)
    {
        const MipLevel &previous = image->mips[level - 1];
        const MipLevel &current = image->mips[level];

        auto *destination = reinterpret_cast<Block64 *>(device_data + current.byte_offset);
        uint32_t output_count = current.block_count_x * current.block_count_y;
        constexpr uint32_t outputs_per_block = threads / 16;
        uint32_t blocks = (output_count + outputs_per_block - 1) / outputs_per_block;
        if (level == 1)
        {
            const auto *source = reinterpret_cast<const Block64 *>(device_data + previous.byte_offset);
            generate_base_mip<Srgb><<<blocks, threads>>>(source, previous.block_count_x, previous.block_count_y,
                                                         destination, current.block_count_x, current.block_count_y,
                                                         means, current.width, current.height);
        }
        else
        {
            MeanImage next_means = {};
            if (level + 1 < image->mip_count)
            {
                scratch.width = (means.width + 1) / 2;
                scratch.height = (means.height + 1) / 2;
                next_means = scratch;
            }

            generate_mean_mip<Srgb><<<blocks, threads>>>(means, destination, current.block_count_x,
                                                         current.block_count_y, next_means, current.width,
                                                         current.height);

            if (level + 1 < image->mip_count)
            {
                std::swap(means, scratch);
            }
        }
        if (cudaGetLastError() != cudaSuccess)
            return false;
    }
    return true;
}

static bool generate_bc6h_cuda_impl(Image *image, uint8_t *device_data, MeanImage means, MeanImage scratch)
{
    constexpr uint32_t threads = 256;
    constexpr uint32_t outputs_per_block = threads / 16;
    for (uint32_t level = 1; level < image->mip_count; ++level)
    {
        const MipLevel &previous = image->mips[level - 1];
        const MipLevel &current = image->mips[level];
        auto *destination = reinterpret_cast<bc6h::Block *>(device_data + current.byte_offset);
        uint32_t output_count = current.block_count_x * current.block_count_y;
        uint32_t blocks = (output_count + outputs_per_block - 1) / outputs_per_block;
        if (level == 1)
        {
            const auto *source = reinterpret_cast<const bc6h::Block *>(device_data + previous.byte_offset);
            generate_bc6h_base_mip<<<blocks, threads>>>(source, previous.block_count_x, previous.block_count_y,
                                                        destination, current.block_count_x, current.block_count_y,
                                                        means, current.width, current.height);
        }
        else
        {
            MeanImage next_means = {};
            if (level + 1 < image->mip_count)
            {
                scratch.width = (means.width + 1) / 2;
                scratch.height = (means.height + 1) / 2;
                next_means = scratch;
            }
            generate_bc6h_mean_mip<<<blocks, threads>>>(means, destination, current.block_count_x,
                                                        current.block_count_y, next_means, current.width,
                                                        current.height);
            if (level + 1 < image->mip_count)
                std::swap(means, scratch);
        }
        if (cudaGetLastError() != cudaSuccess)
            return false;
    }
    return true;
}

template <bool Srgb>
static bool generate_bc7_cuda_impl(Image *image, uint8_t *device_data, bc7::MeanImage means, bc7::MeanImage scratch)
{
    constexpr uint32_t threads = 256, outputs_per_block = threads / 16;
    for (uint32_t level = 1; level < image->mip_count; ++level)
    {
        const MipLevel &previous = image->mips[level - 1], &current = image->mips[level];
        auto *destination = reinterpret_cast<bc7::Block *>(device_data + current.byte_offset);
        uint32_t count = current.block_count_x * current.block_count_y,
                 blocks = (count + outputs_per_block - 1) / outputs_per_block;
        if (level == 1)
        {
            const auto *source = reinterpret_cast<const bc7::Block *>(device_data + previous.byte_offset);
            generate_bc7_base<Srgb><<<blocks, threads>>>(source, previous.block_count_x, previous.block_count_y,
                                                         destination, current.block_count_x, current.block_count_y,
                                                         means, current.width, current.height);
        }
        else
        {
            bc7::MeanImage next = {};
            if (level + 1 < image->mip_count)
            {
                scratch.width = (means.width + 1) / 2;
                scratch.height = (means.height + 1) / 2;
                next = scratch;
            }
            generate_bc7_mean<Srgb><<<blocks, threads>>>(means, destination, current.block_count_x,
                                                         current.block_count_y, next, current.width, current.height);
            if (level + 1 < image->mip_count)
                std::swap(means, scratch);
        }
        if (cudaGetLastError() != cudaSuccess)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// Public GPU entry point
// ---------------------------------------------------------------------

bool prepare_mipmaps_cuda(const Image *image)
{
    return image && cudaFree(nullptr) == cudaSuccess && initialize_tables() && g_workspace.prepare(image);
}

bool generate_mipmaps_cuda(Image *image)
{
    if (image->mip_count <= 1)
        return true;
    if (!initialize_tables())
        return false;

    const uint32_t base_width = image->mips[0].block_count_x;
    const uint32_t base_height = image->mips[0].block_count_y;
    const size_t base_count = (size_t)base_width * base_height;
    const uint32_t scratch_width = (base_width + 1) / 2;
    const uint32_t scratch_height = (base_height + 1) / 2;
    const size_t scratch_count = (size_t)scratch_width * scratch_height;

    if (!g_workspace.prepare(image))
        return false;

    if (cudaMemcpy(g_workspace.data.get(), image->data, image->mips[0].byte_size, cudaMemcpyHostToDevice) !=
        cudaSuccess)
        return false;

    MeanImage means = {g_workspace.base.get(), g_workspace.base.get() + base_count,
                       g_workspace.base.get() + base_count * 2, base_width, base_height};
    MeanImage scratch = {g_workspace.scratch.get(), g_workspace.scratch.get() + scratch_count,
                         g_workspace.scratch.get() + scratch_count * 2, scratch_width, scratch_height};

    bc7::MeanImage means4 = {g_workspace.base.get(),
                             g_workspace.base.get() + base_count,
                             g_workspace.base.get() + base_count * 2,
                             g_workspace.base.get() + base_count * 3,
                             base_width,
                             base_height};
    bc7::MeanImage scratch4 = {g_workspace.scratch.get(),
                               g_workspace.scratch.get() + scratch_count,
                               g_workspace.scratch.get() + scratch_count * 2,
                               g_workspace.scratch.get() + scratch_count * 3,
                               scratch_width,
                               scratch_height};
    bool generated =
        image->format == Format::BC6H_UF16
            ? generate_bc6h_cuda_impl(image, g_workspace.data.get(), means, scratch)
            : (image->format == Format::BC7
                   ? (image->is_srgb ? generate_bc7_cuda_impl<true>(image, g_workspace.data.get(), means4, scratch4)
                                     : generate_bc7_cuda_impl<false>(image, g_workspace.data.get(), means4, scratch4))
                   : (image->is_srgb ? generate_cuda_impl<true>(image, g_workspace.data.get(), means, scratch)
                                     : generate_cuda_impl<false>(image, g_workspace.data.get(), means, scratch)));
    const size_t generated_offset = image->mips[1].byte_offset;
    return generated && cudaMemcpy(image->data + generated_offset, g_workspace.data.get() + generated_offset,
                                   image->data_size - generated_offset, cudaMemcpyDeviceToHost) == cudaSuccess;
}
