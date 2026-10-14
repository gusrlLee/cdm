#include "bc1.h"
#include "bc2.h"
#include "bc3.h"
#include "bc4.h"
#include "bc5.h"
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
                                         MeanImage means, uint32_t valid_width, uint32_t valid_height,
                                         uint32_t source_pixel_width, uint32_t source_pixel_height)
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

    valid_width = block_valid_extent(valid_width, output_x);
    valid_height = block_valid_extent(valid_height, output_y);
    uint32_t parent = sample_index >> 2;
    uint32_t quadrant = sample_index & 3u;
    uint32_t x0 = clamp_index(output_x * 2, source_width);
    uint32_t x1 = clamp_index(x0 + 1, source_width);
    uint32_t y0 = clamp_index(output_y * 2, source_height);
    uint32_t y1 = clamp_index(y0 + 1, source_height);
    uint32_t source_x = (parent & 1u) ? x1 : x0;
    uint32_t source_y = (parent & 2u) ? y1 : y0;

    Color sample = bc1::device_quadrant_mean<Srgb>(source[source_y * source_width + source_x], quadrant,
                                                   block_valid_extent(source_pixel_width, source_x),
                                                   block_valid_extent(source_pixel_height, source_y));
    // Form stored means from valid child samples, including singleton dimensions.
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t lane = repeat_small_sample(sample_index, valid_width, valid_height);
        sample = {__shfl_sync(mask, sample.r, lane, 16), __shfl_sync(mask, sample.g, lane, 16),
                  __shfl_sync(mask, sample.b, lane, 16)};
    }
    Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};

    bool unique_parent =
        parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
    if (quadrant == 0 && unique_parent && source_x < means.width && source_y < means.height)
    {
        size_t index = (size_t)source_y * means.width + source_x;
        means.r[index] = parent_mean.r;
        means.g[index] = parent_mean.g;
        means.b[index] = parent_mean.b;
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

    valid_width = block_valid_extent(valid_width, output_x);
    valid_height = block_valid_extent(valid_height, output_y);
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
        x = clamp_index(output_x * 4 + repeat_small_coordinate(local & 3u, valid_width), source.width);
        y = clamp_index(output_y * 4 + repeat_small_coordinate(local >> 2, valid_height), source.height);
        index = (size_t)y * source.width + x;
        sample = {source.r[index], source.g[index], source.b[index]};
    }
    bc1::encode_half_warp<Srgb>(sample, sample_index, destination + output_index);
}

static __global__ void generate_bc6h_base_mip(const bc6h::Block *source, uint32_t source_width, uint32_t source_height,
                                              bc6h::Block *destination, uint32_t destination_width,
                                              uint32_t destination_height, MeanImage means, uint32_t valid_width,
                                              uint32_t valid_height, uint32_t source_pixel_width, uint32_t source_pixel_height)
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
    valid_width = block_valid_extent(valid_width, ox);
    valid_height = block_valid_extent(valid_height, oy);
    uint32_t parent = lane >> 2, quadrant = lane & 3u;
    uint32_t x0 = clamp_index(ox * 2, source_width), x1 = clamp_index(x0 + 1, source_width);
    uint32_t y0 = clamp_index(oy * 2, source_height), y1 = clamp_index(y0 + 1, source_height);
    uint32_t sx = parent & 1u ? x1 : x0, sy = parent & 2u ? y1 : y0;
    Color sample = bc6h::device_quadrant_mean(source[sy * source_width + sx], quadrant,
                                              block_valid_extent(source_pixel_width, sx),
                                              block_valid_extent(source_pixel_height, sy));
    // Form stored means from valid child samples, including singleton dimensions.
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t source_lane = repeat_small_sample(lane, valid_width, valid_height);
        sample = {__shfl_sync(mask, sample.r, source_lane, 16), __shfl_sync(mask, sample.g, source_lane, 16),
                  __shfl_sync(mask, sample.b, source_lane, 16)};
    }
    Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};
    bool unique =
        parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
    if (quadrant == 0 && unique && sx < means.width && sy < means.height)
    {
        size_t i = size_t(sy) * means.width + sx;
        means.r[i] = parent_mean.r;
        means.g[i] = parent_mean.g;
        means.b[i] = parent_mean.b;
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
    valid_width = block_valid_extent(valid_width, ox);
    valid_height = block_valid_extent(valid_height, oy);
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
        x = clamp_index(ox * 4 + repeat_small_coordinate(local & 3u, valid_width), source.width);
        y = clamp_index(oy * 4 + repeat_small_coordinate(local >> 2, valid_height), source.height);
        index = size_t(y) * source.width + x;
        sample = {source.r[index], source.g[index], source.b[index]};
    }
    bc6h::encode_half_warp(sample, lane, destination + output_index);
}

struct Bc7CudaCodec
{
    using Block = bc7::Block;
    using Color = Float4;
    using MeanImage = MeanImage4;
    template <bool Srgb> static __device__ Color device_quadrant_mean(const Block &b, uint32_t q,
                                                                      uint32_t valid_width, uint32_t valid_height)
    {
        return bc7::device_quadrant_mean<Srgb>(b, q, valid_width, valid_height);
    }
    template <bool Srgb> static __device__ void encode_half_warp(Color s, uint32_t lane, Block *out)
    {
        bc7::encode_half_warp<Srgb>(s, lane, out);
    }
};

__device__ __forceinline__ Float3 quadrant_sum(Float3 s, unsigned mask)
{
    return {sum4(s.r, mask) * .25f, sum4(s.g, mask) * .25f, sum4(s.b, mask) * .25f};
}
__device__ __forceinline__ Float4 quadrant_sum(Float4 s, unsigned mask)
{
    return {sum4(s.r, mask) * .25f, sum4(s.g, mask) * .25f, sum4(s.b, mask) * .25f, sum4(s.a, mask) * .25f};
}
__device__ __forceinline__ Float3 shuffle_sample(Float3 s, uint32_t lane, unsigned mask)
{
    return {__shfl_sync(mask, s.r, lane, 16), __shfl_sync(mask, s.g, lane, 16), __shfl_sync(mask, s.b, lane, 16)};
}
__device__ __forceinline__ Float4 shuffle_sample(Float4 s, uint32_t lane, unsigned mask)
{
    return {__shfl_sync(mask, s.r, lane, 16), __shfl_sync(mask, s.g, lane, 16),
            __shfl_sync(mask, s.b, lane, 16), __shfl_sync(mask, s.a, lane, 16)};
}

template <bool Srgb, typename Codec>
static __global__ void generate_channel_base(const typename Codec::Block *source, uint32_t source_width, uint32_t source_height,
                                         typename Codec::Block *destination, uint32_t destination_width,
                                         uint32_t destination_height, typename Codec::MeanImage means, uint32_t valid_width,
                                         uint32_t valid_height, uint32_t source_pixel_width, uint32_t source_pixel_height)
{
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4), lane = threadIdx.x & 15u;
    if (output_index >= destination_width * destination_height)
        return;
    unsigned mask = half_warp_mask();
    uint32_t ox = lane == 0 ? output_index % destination_width : 0,
             oy = lane == 0 ? output_index / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16);
    oy = __shfl_sync(mask, oy, 0, 16);
    valid_width = block_valid_extent(valid_width, ox);
    valid_height = block_valid_extent(valid_height, oy);
    uint32_t parent = lane >> 2, quadrant = lane & 3;
    uint32_t x0 = clamp_index(ox * 2, source_width), x1 = clamp_index(x0 + 1, source_width),
             y0 = clamp_index(oy * 2, source_height), y1 = clamp_index(y0 + 1, source_height);
    uint32_t sx = (parent & 1) ? x1 : x0, sy = (parent & 2) ? y1 : y0;
    typename Codec::Color sample = Codec::template device_quadrant_mean<Srgb>(source[sy * source_width + sx], quadrant,
                                                                              block_valid_extent(source_pixel_width, sx),
                                                                              block_valid_extent(source_pixel_height, sy));
    // Form stored means from valid child samples, including singleton dimensions.
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t l = repeat_small_sample(lane, valid_width, valid_height);
        sample = shuffle_sample(sample, l, mask);
    }
    const auto pm = quadrant_sum(sample, mask);
    bool unique =
        parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
    if (quadrant == 0 && unique && sx < means.width && sy < means.height)
    {
        means.set(sx, sy, pm);
    }
    Codec::template encode_half_warp<Srgb>(sample, lane, destination + output_index);
}

template <bool Srgb, typename Codec>
static __global__ void generate_channel_mean(typename Codec::MeanImage source, typename Codec::Block *destination, uint32_t destination_width,
                                         uint32_t destination_height, typename Codec::MeanImage next, uint32_t valid_width,
                                         uint32_t valid_height)
{
    uint32_t oi = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4), lane = threadIdx.x & 15u;
    if (oi >= destination_width * destination_height)
        return;
    unsigned mask = half_warp_mask();
    uint32_t ox = lane == 0 ? oi % destination_width : 0, oy = lane == 0 ? oi / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16);
    oy = __shfl_sync(mask, oy, 0, 16);
    valid_width = block_valid_extent(valid_width, ox);
    valid_height = block_valid_extent(valid_height, oy);
    uint32_t t = texel_index(lane);
    uint32_t x = clamp_index(ox * 4 + (t & 3), source.width), y = clamp_index(oy * 4 + (t >> 2), source.height);
    typename Codec::Color sample = source.get(x, y);
    const auto pm = quadrant_sum(sample, mask);
    if (next.channel(0) && (lane & 3) == 0)
    {
        uint32_t q = lane >> 2, nx = ox * 2 + (q & 1), ny = oy * 2 + (q >> 1);
        if (nx < next.width && ny < next.height)
        {
            next.set(nx, ny, pm);
        }
    }
    if (valid_width < 4 || valid_height < 4)
    {
        x = clamp_index(ox * 4 + repeat_small_coordinate(t & 3, valid_width), source.width);
        y = clamp_index(oy * 4 + repeat_small_coordinate(t >> 2, valid_height), source.height);
        sample = source.get(x, y);
    }
    Codec::template encode_half_warp<Srgb>(sample, lane, destination + oi);
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
        const size_t channels = image->format == Format::BC4_UNORM || image->format == Format::BC4_SNORM ? 1 :
                                image->format == Format::BC5_UNORM || image->format == Format::BC5_SNORM ? 2 :
                                image->format == Format::BC2 || image->format == Format::BC3 || image->format == Format::BC7 ? 4 : 3;
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
    // The buffer has block-count capacity, but only mip2's logical cells are active.
    means.width = image->mip_count > 2 ? image->mips[2].width : 1;
    means.height = image->mip_count > 2 ? image->mips[2].height : 1;
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
                                                         means, current.width, current.height,
                                                         previous.width, previous.height);
        }
        else
        {
            MeanImage next_means = {};
            if (level + 1 < image->mip_count)
            {
                scratch.width = image->mips[level + 1].width;
                scratch.height = image->mips[level + 1].height;
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
    // The buffer has block-count capacity, but only mip2's logical cells are active.
    means.width = image->mip_count > 2 ? image->mips[2].width : 1;
    means.height = image->mip_count > 2 ? image->mips[2].height : 1;
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
                                                        means, current.width, current.height,
                                                        previous.width, previous.height);
        }
        else
        {
            MeanImage next_means = {};
            if (level + 1 < image->mip_count)
            {
                scratch.width = image->mips[level + 1].width;
                scratch.height = image->mips[level + 1].height;
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

template <bool Srgb, typename Codec>
static bool generate_channel_cuda_impl(Image *image, uint8_t *device_data, typename Codec::MeanImage means, typename Codec::MeanImage scratch)
{
    // The buffer has block-count capacity, but only mip2's logical cells are active.
    means.width = image->mip_count > 2 ? image->mips[2].width : 1;
    means.height = image->mip_count > 2 ? image->mips[2].height : 1;
    constexpr uint32_t threads = 256, outputs_per_block = threads / 16;
    for (uint32_t level = 1; level < image->mip_count; ++level)
    {
        const MipLevel &previous = image->mips[level - 1], &current = image->mips[level];
        auto *destination = reinterpret_cast<typename Codec::Block *>(device_data + current.byte_offset);
        uint32_t count = current.block_count_x * current.block_count_y,
                 blocks = (count + outputs_per_block - 1) / outputs_per_block;
        if (level == 1)
        {
            const auto *source = reinterpret_cast<const typename Codec::Block *>(device_data + previous.byte_offset);
            generate_channel_base<Srgb, Codec><<<blocks, threads>>>(source, previous.block_count_x, previous.block_count_y,
                                                         destination, current.block_count_x, current.block_count_y,
                                                         means, current.width, current.height,
                                                         previous.width, previous.height);
        }
        else
        {
            typename Codec::MeanImage next = {};
            if (level + 1 < image->mip_count)
            {
                scratch.width = image->mips[level + 1].width;
                scratch.height = image->mips[level + 1].height;
                next = scratch;
            }
            generate_channel_mean<Srgb, Codec><<<blocks, threads>>>(means, destination, current.block_count_x,
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

template <typename Mean> static Mean device_mean_view(float *storage, size_t capacity, uint32_t w, uint32_t h)
{
    if constexpr (Mean::channel_count == 3)
        return {storage, storage + capacity, storage + capacity * 2, w, h};
    else if constexpr (Mean::channel_count == 4)
        return {storage, storage + capacity, storage + capacity * 2, storage + capacity * 3, w, h};
    else
    {
        Mean result{};
        for (uint32_t c = 0; c < Mean::channel_count; ++c) result.planes[c] = storage + capacity * c;
        result.width = w;
        result.height = h;
        return result;
    }
}

template <typename Codec> static bool dispatch_channel_cuda(Image *image)
{
    const uint32_t w = image->mips[0].block_count_x, h = image->mips[0].block_count_y;
    const uint32_t sw = (w + 1) / 2, sh = (h + 1) / 2;
    using Mean = typename Codec::MeanImage;
    const Mean means = device_mean_view<Mean>(g_workspace.base.get(), size_t(w) * h, w, h);
    const Mean scratch = device_mean_view<Mean>(g_workspace.scratch.get(), size_t(sw) * sh, sw, sh);
    return image->is_srgb ? generate_channel_cuda_impl<true, Codec>(image, g_workspace.data.get(), means, scratch)
                          : generate_channel_cuda_impl<false, Codec>(image, g_workspace.data.get(), means, scratch);
}

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

    bool generated = false;
    switch (image->format)
    {
    case Format::BC1:
    case Format::BC6H_UF16:
    {
        MeanImage means = device_mean_view<MeanImage>(g_workspace.base.get(), base_count, base_width, base_height);
        MeanImage scratch = device_mean_view<MeanImage>(g_workspace.scratch.get(), scratch_count, scratch_width, scratch_height);
        generated = image->format == Format::BC6H_UF16
                        ? generate_bc6h_cuda_impl(image, g_workspace.data.get(), means, scratch)
                        : (image->is_srgb ? generate_cuda_impl<true>(image, g_workspace.data.get(), means, scratch)
                                          : generate_cuda_impl<false>(image, g_workspace.data.get(), means, scratch));
        break;
    }
    case Format::BC2: generated = dispatch_channel_cuda<bc2::Codec>(image); break;
    case Format::BC3: generated = dispatch_channel_cuda<bc3::Codec>(image); break;
    case Format::BC4_UNORM: generated = dispatch_channel_cuda<bc4::Codec<false>>(image); break;
    case Format::BC4_SNORM: generated = dispatch_channel_cuda<bc4::Codec<true>>(image); break;
    case Format::BC5_UNORM: generated = dispatch_channel_cuda<bc5::Codec<false>>(image); break;
    case Format::BC5_SNORM: generated = dispatch_channel_cuda<bc5::Codec<true>>(image); break;
    case Format::BC7: generated = dispatch_channel_cuda<Bc7CudaCodec>(image); break;
    default: return false;
    }
    const size_t generated_offset = image->mips[1].byte_offset;
    return generated && cudaMemcpy(image->data + generated_offset, g_workspace.data.get() + generated_offset,
                                   image->data_size - generated_offset, cudaMemcpyDeviceToHost) == cudaSuccess;
}
