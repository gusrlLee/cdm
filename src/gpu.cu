#include "gpu.h"

#include "core/color.h"
#include "core/color_cuda.cuh"
#include "core/buffer.h"
#include "core/mean.h"
#include "core/warp.cuh"

#include "codecs/bc1_cuda.cuh"
#include "codecs/bc4_cuda.cuh"
#include "codecs/bc7_cuda.cuh"
#include "codecs/bc5_cuda.cuh"
#include "codecs/bc2_cuda.cuh"
#include "codecs/bc3_cuda.cuh"
#include "codecs/bc6h_cuda.cuh"

#include <cuda_runtime.h>
#include <array>
#include <cstdio>

// Print the CUDA error and let the caller handle failure.
static bool cuda_chk(cudaError_t result, const char *operation)
{
    if (result == cudaSuccess)
        return true;

    std::fprintf(stderr, "CUDA error (%s): %s\n", operation, cudaGetErrorString(result));
    return false;
}

static bool initialize_color_tables()
{
    static bool init = false;
    if (init)
        return true;

    const auto &thresholds31 = bc1::get_srgb_quantize_thresholds<31>();
    const auto &thresholds63 = bc1::get_srgb_quantize_thresholds<63>();

    if (!cuda_chk(cudaMemcpyToSymbol(g_srgb_to_linear, c_srgb_to_linear, sizeof(c_srgb_to_linear)),
                  "Upload sRGB table"))
        return false;

    if (!cuda_chk(cudaMemcpyToSymbol(g_threshold31, thresholds31.data(), thresholds31.size() * sizeof(float)),
                  "Upload RGB565 5-bit thresholds"))
        return false;

    if (!cuda_chk(cudaMemcpyToSymbol(g_threshold63, thresholds63.data(), thresholds63.size() * sizeof(float)),
                  "Upload RGB565 6-bit thresholds"))
        return false;

    init = true;
    return true;
}

static bool initialize_bc7_tables()
{
    static bool initialized = false;
    if (!initialized)
    {
        initialized = cuda_chk(bc7::upload_tables_cuda(), "Upload BC7 tables");
    }
    return initialized;
}

bool initialize_tables_cuda()
{
    return initialize_color_tables() && initialize_bc7_tables();
}

struct CudaWorkspace
{
    DeviceBuffer<uint8_t> data;
    DeviceBuffer<float> base;
    DeviceBuffer<float> scratch;

    bool prepare(const Image &image)
    {
        const uint32_t bw = image.mips[0].block_count_x;
        const uint32_t bh = image.mips[0].block_count_y;

        const uint32_t sw = (bw + 1) / 2;
        const uint32_t sh = (bh + 1) / 2;

        const size_t base_count = static_cast<size_t>(bw) * bh;
        const size_t scratch_count = static_cast<size_t>(sw) * sh;

        const size_t channels =
            (image.format == Format::BC7 || image.format == Format::BC2 || image.format == Format::BC3) ? 4 : 3;
        return data.allocate(image.data_size) && base.allocate(base_count * channels) &&
               scratch.allocate(scratch_count * channels);
    }
};

// View one GPU allocation as three separate RGB channels.
static MeanImage device_mean_view(float *data, size_t channel_capacity, uint32_t width, uint32_t height)
{
    return {data, data + channel_capacity, data + channel_capacity * 2, width, height};
}

// For ablation study
static constexpr bool bc1_use_histogram = false;
static constexpr bool bc1_use_mean_pyramid = false;

// Generate mip 1 and store the means needed by later levels.
static __global__ void generate_base_mip(const bc1::Block *source, MipLevel source_level, bc1::Block *destination,
                                         MipLevel destination_level, MeanImage means, bool is_srgb)
{
    const uint32_t output = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    const uint32_t output_count = destination_level.block_count_x * destination_level.block_count_y;

    // The entire group exits together.
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

    Float3 sample{};
    if (bc1_use_histogram)
    {
        sample = bc1::device_quadrant_mean(source[source_index], quadrant, is_srgb, source_width, source_height);
    }
    else 
    {
        sample = bc1::device_quadrant_mean_texels(source[source_index], quadrant, is_srgb, source_width, source_height);
    }

    const uint32_t remaining_x = destination_level.width - bx * 4;
    const uint32_t remaining_y = destination_level.height - by * 4;
    const uint32_t valid_width = remaining_x < 4 ? remaining_x : 4;
    const uint32_t valid_height = remaining_y < 4 ? remaining_y : 4;

    if (valid_width < 4 || valid_height < 4)
        sample = bc1::repeat_small_sample(sample, valid_width, valid_height);
    if (means.r != nullptr)
    {
        const Float3 parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};
    
        // Avoid duplicate writes when parent coordinates are clamped.
        const bool unique_parent =
            parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
    
        if (quadrant == 0 && unique_parent)
            means.set(sx, sy, parent_mean);
    }

    bc1::encode_half_warp(sample, is_srgb, destination + output);
}

// Encode a mip level and build the next mean image.
static __global__ void generate_mean_mip(MeanImage source, bc1::Block *destination, MipLevel destination_level,
                                         MeanImage next_means, bool is_srgb)
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

    Float3 sample = source.get(bx * 4 + x, by * 4 + y);

    // Preserve the box-filtered means before repeating encoder samples.
    const Float3 mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};

    if (next_means.r != nullptr && (lane & 3u) == 0)
    {
        const uint32_t quadrant = lane >> 2u;
        const uint32_t next_x = bx * 2 + (quadrant & 1u);
        const uint32_t next_y = by * 2 + (quadrant >> 1u);

        next_means.set(next_x, next_y, mean);
    }

    const uint32_t remaining_x = destination_level.width - bx * 4;
    const uint32_t remaining_y = destination_level.height - by * 4;
    const uint32_t valid_width = remaining_x < 4 ? remaining_x : 4;
    const uint32_t valid_height = remaining_y < 4 ? remaining_y : 4;

    if (valid_width < 4 || valid_height < 4)
        sample = bc1::repeat_small_sample(sample, valid_width, valid_height);

    bc1::encode_half_warp(sample, is_srgb, destination + output);
}

__device__ __forceinline__ Float3 repeat_extended_sample(Float3 sample, uint32_t width, uint32_t height)
{
    return bc1::repeat_small_sample(sample, width, height);
}

__device__ __forceinline__ Float3 average_extended_sample(Float3 sample, unsigned mask)
{
    return {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};
}

// Generate mip 1 and store the means needed by later levels.
__device__ __forceinline__ Float4 repeat_extended_sample(Float4 sample, uint32_t width, uint32_t height)
{
    const Float3 rgb = bc1::repeat_small_sample({sample.r, sample.g, sample.b}, width, height);
    return {rgb.r, rgb.g, rgb.b, bc4::repeat_small_sample(sample.a, width, height)};
}

__device__ __forceinline__ Float4 average_extended_sample(Float4 sample, unsigned mask)
{
    return {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f,
            sum4(sample.a, mask) * 0.25f};
}

// The flag selects sRGB for BC2/3 and signed channels for BC5. BC6H ignores it.
template <typename Block, typename Means>
static __global__ void generate_extended_base_mip(const Block *source, MipLevel source_level, Block *destination,
                                                  MipLevel destination_level, Means means, bool color_mode)
{
    const uint32_t output = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    const uint32_t output_count = destination_level.block_count_x * destination_level.block_count_y;

    // The entire group exits together.
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

    auto sample = device_quadrant_mean(source[source_index], quadrant, color_mode, source_width, source_height);

    const uint32_t remaining_x = destination_level.width - bx * 4;
    const uint32_t remaining_y = destination_level.height - by * 4;
    const uint32_t valid_width = remaining_x < 4 ? remaining_x : 4;
    const uint32_t valid_height = remaining_y < 4 ? remaining_y : 4;

    if (valid_width < 4 || valid_height < 4)
        sample = repeat_extended_sample(sample, valid_width, valid_height);

    const auto parent_mean = average_extended_sample(sample, mask);

    // Avoid duplicate writes when parent coordinates are clamped.
    const bool unique_parent =
        parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);

    if (quadrant == 0 && unique_parent)
        means.set(sx, sy, parent_mean);

    encode_half_warp(sample, color_mode, destination + output);
}

// Encode a mip level and build the next mean image.
template <typename Block, typename Means>
static __global__ void generate_extended_mean_mip(Means source, Block *destination, MipLevel destination_level,
                                                  Means next_means, bool color_mode)
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

    auto sample = source.get(bx * 4 + x, by * 4 + y);

    // Preserve the box-filtered means before repeating encoder samples.
    const auto mean = average_extended_sample(sample, mask);

    if (next_means.r != nullptr && (lane & 3u) == 0)
    {
        const uint32_t quadrant = lane >> 2u;
        const uint32_t next_x = bx * 2 + (quadrant & 1u);
        const uint32_t next_y = by * 2 + (quadrant >> 1u);

        next_means.set(next_x, next_y, mean);
    }

    const uint32_t remaining_x = destination_level.width - bx * 4;
    const uint32_t remaining_y = destination_level.height - by * 4;
    const uint32_t valid_width = remaining_x < 4 ? remaining_x : 4;
    const uint32_t valid_height = remaining_y < 4 ? remaining_y : 4;

    if (valid_width < 4 || valid_height < 4)
        sample = repeat_extended_sample(sample, valid_width, valid_height);

    encode_half_warp(sample, color_mode, destination + output);
}

bool generate_mipmaps_cuda(Image &image, float *generation_time_ms)
{
    if (generation_time_ms)
        *generation_time_ms = 0.0f;

    if (!image.data || image.mip_count == 0)
        return false;
    switch (image.format)
    {
    case Format::BC1:
    case Format::BC2:
    case Format::BC3:
    case Format::BC4_UNORM:
    case Format::BC5_UNORM:
    case Format::BC5_SNORM:
    case Format::BC6H_UF16:
        break;
    case Format::BC7:
        if (!image.is_srgb)
            return false;
        break;
    default:
        return false;
    }
    if (image.mip_count == 1)
        return true;

    static CudaWorkspace workspace;

    if ((image.format == Format::BC1 || image.format == Format::BC2 || image.format == Format::BC3) && !initialize_color_tables())
        return false;

    if (image.format == Format::BC7 && !initialize_bc7_tables())
        return false;

    if (!workspace.prepare(image))
        return false;

    const MipLevel &base_level = image.mips[0];
    uint8_t *device_data = workspace.data.get();

    // Upload only mip 0.
    if (!cuda_chk(cudaMemcpy(device_data + base_level.byte_offset, image.data + base_level.byte_offset,
                             base_level.byte_size, cudaMemcpyHostToDevice),
                  "Upload mip 0"))
        return false;

    const uint32_t bw = base_level.block_count_x;
    const uint32_t bh = base_level.block_count_y;
    const size_t base_count = static_cast<size_t>(bw) * bh;
    const size_t scratch_count = static_cast<size_t>((bw + 1) / 2) * ((bh + 1) / 2);

    const uint32_t width = image.mip_count > 2 ? image.mips[2].width : 1;
    const uint32_t height = image.mip_count > 2 ? image.mips[2].height : 1;

    MeanImage means = device_mean_view(workspace.base.get(), base_count, width, height);
    MeanImage scratch = device_mean_view(workspace.scratch.get(), scratch_count, 1, 1);

    MeanImage4 rgba_means{}, rgba_scratch{};
    if (image.format == Format::BC7 || image.format == Format::BC2 || image.format == Format::BC3)
    {
        float *base = workspace.base.get();
        float *scratch_data = workspace.scratch.get();
        rgba_means = {base, base + base_count, base + base_count * 2, base + base_count * 3, width, height};
        rgba_scratch = {scratch_data,
                        scratch_data + scratch_count,
                        scratch_data + scratch_count * 2,
                        scratch_data + scratch_count * 3,
                        1,
                        1};
    }

    const uint32_t threads = 256;
    const uint32_t blocks_per_group = threads / 16;

    // Measure GPU mip generation after allocation and input upload.
    cudaEvent_t start;
    cudaEvent_t end;

    if (!cuda_chk(cudaEventCreate(&start), "Create start event"))
        return false;

    if (!cuda_chk(cudaEventCreate(&end), "Create end event"))
    {
        cudaEventDestroy(start);
        return false;
    }

    if (!cuda_chk(cudaEventRecord(start), "Record start event"))
    {
        cudaEventDestroy(end);
        cudaEventDestroy(start);
        return false;
    }

    for (uint32_t level = 1; level < image.mip_count; ++level)
    {
        const MipLevel &current = image.mips[level];
        const uint32_t block_count = current.block_count_x * current.block_count_y;
        const uint32_t groups = (block_count + blocks_per_group - 1) / blocks_per_group;

        if (image.format == Format::BC1 && !bc1_use_mean_pyramid)
        {
            const MipLevel &previous = image.mips[level - 1];

            const auto *source = reinterpret_cast<const bc1::Block *>(device_data + previous.byte_offset);
            auto *destination = reinterpret_cast<bc1::Block *>(device_data + current.byte_offset);

            generate_base_mip<<<groups, threads>>>(source, previous, destination, current, MeanImage{}, image.is_srgb);

            if (!cuda_chk(cudaGetLastError(), "Launch BC1 ablation kernel"))
            {
                cudaEventDestroy(end);
                cudaEventDestroy(start);
                return false;
            }

            continue;
        }

        if (image.format == Format::BC2 || image.format == Format::BC3)
        {
            auto *destination = reinterpret_cast<bc2::Block *>(device_data + current.byte_offset);
            if (level == 1)
            {
                const auto *source = reinterpret_cast<const bc2::Block *>(device_data + base_level.byte_offset);
                if (image.format == Format::BC2)
                    generate_extended_base_mip<<<groups, threads>>>(source, base_level, destination, current,
                                                                    rgba_means, image.is_srgb);
                else
                    generate_extended_base_mip<<<groups, threads>>>(
                        reinterpret_cast<const bc3::Block *>(source), base_level,
                        reinterpret_cast<bc3::Block *>(destination), current, rgba_means, image.is_srgb);
            }
            else
            {
                MeanImage4 next{};
                if (level + 1 < image.mip_count)
                {
                    rgba_scratch.width = image.mips[level + 1].width;
                    rgba_scratch.height = image.mips[level + 1].height;
                    next = rgba_scratch;
                }
                if (image.format == Format::BC2)
                    generate_extended_mean_mip<<<groups, threads>>>(rgba_means, destination, current, next,
                                                                    image.is_srgb);
                else
                    generate_extended_mean_mip<<<groups, threads>>>(
                        rgba_means, reinterpret_cast<bc3::Block *>(destination), current, next, image.is_srgb);
                if (next.r != nullptr)
                    std::swap(rgba_means, rgba_scratch);
            }
            if (!cuda_chk(cudaGetLastError(), "Launch BC2/BC3 mip kernel"))
            {
                cudaEventDestroy(end);
                cudaEventDestroy(start);
                return false;
            }
            continue;
        }

        if (image.format == Format::BC7)
        {
            auto *destination = reinterpret_cast<bc7::Block *>(device_data + current.byte_offset);
            if (level == 1)
            {
                const auto *source = reinterpret_cast<const bc7::Block *>(device_data + base_level.byte_offset);
                bc7::generate_base_mip_cuda<<<groups, threads>>>(source, base_level, destination, current, rgba_means);
            }
            else
            {
                MeanImage4 next{};
                if (level + 1 < image.mip_count)
                {
                    rgba_scratch.width = image.mips[level + 1].width;
                    rgba_scratch.height = image.mips[level + 1].height;
                    next = rgba_scratch;
                }
                bc7::generate_mean_mip_cuda<<<groups, threads>>>(rgba_means, destination, current, next);
                if (next.r != nullptr)
                    std::swap(rgba_means, rgba_scratch);
            }
            if (!cuda_chk(cudaGetLastError(), "Launch BC7 mip kernel"))
            {
                cudaEventDestroy(end);
                cudaEventDestroy(start);
                return false;
            }
            continue;
        }

        if (level == 1)
        {
            switch (image.format)
            {
            case Format::BC1:
            {
                const auto *source = reinterpret_cast<const bc1::Block *>(device_data + base_level.byte_offset);
                auto *destination = reinterpret_cast<bc1::Block *>(device_data + current.byte_offset);

                generate_base_mip<<<groups, threads>>>(source, base_level, destination, current, means, image.is_srgb);
                break;
            }

            case Format::BC4_UNORM:
            {
                const auto *source = reinterpret_cast<const bc4::Block *>(device_data + base_level.byte_offset);
                auto *destination = reinterpret_cast<bc4::Block *>(device_data + current.byte_offset);

                bc4::generate_base_mip_cuda(source, base_level, destination, current, means, groups, threads);
                break;
            }

            case Format::BC5_UNORM:
            case Format::BC5_SNORM:
            {
                const auto *source = reinterpret_cast<const bc5::Block *>(device_data + base_level.byte_offset);
                auto *destination = reinterpret_cast<bc5::Block *>(device_data + current.byte_offset);
                generate_extended_base_mip<<<groups, threads>>>(source, base_level, destination, current, means,
                                                                image.format == Format::BC5_SNORM);
                break;
            }

            case Format::BC6H_UF16:
            {
                const auto *source = reinterpret_cast<const bc6h::Block *>(device_data + base_level.byte_offset);
                auto *destination = reinterpret_cast<bc6h::Block *>(device_data + current.byte_offset);
                generate_extended_base_mip<<<groups, threads>>>(source, base_level, destination, current, means, false);
                break;
            }

            default:
                break; // Unsupported formats were rejected above.
            }
        }
        else
        {
            MeanImage next_means{};

            if (level + 1 < image.mip_count)
            {
                scratch.width = image.mips[level + 1].width;
                scratch.height = image.mips[level + 1].height;
                next_means = scratch;
            }

            switch (image.format)
            {
            case Format::BC1:
            {
                auto *destination = reinterpret_cast<bc1::Block *>(device_data + current.byte_offset);
                generate_mean_mip<<<groups, threads>>>(means, destination, current, next_means, image.is_srgb);
                break;
            }

            case Format::BC4_UNORM:
            {
                auto *destination = reinterpret_cast<bc4::Block *>(device_data + current.byte_offset);
                bc4::generate_mean_mip_cuda(means, destination, current, next_means, groups, threads);
                break;
            }

            case Format::BC5_UNORM:
            case Format::BC5_SNORM:
            {
                auto *destination = reinterpret_cast<bc5::Block *>(device_data + current.byte_offset);
                generate_extended_mean_mip<<<groups, threads>>>(means, destination, current, next_means,
                                                                image.format == Format::BC5_SNORM);
                break;
            }

            case Format::BC6H_UF16:
            {
                auto *destination = reinterpret_cast<bc6h::Block *>(device_data + current.byte_offset);
                generate_extended_mean_mip<<<groups, threads>>>(means, destination, current, next_means, false);
                break;
            }

            default:
                break; // Unsupported formats were rejected above.
            }

            if (next_means.r != nullptr)
            {
                const MeanImage previous = means;
                means = scratch;
                scratch = previous;
            }
        }

        if (!cuda_chk(cudaGetLastError(), "Launch mip kernel"))
        {
            cudaEventDestroy(end);
            cudaEventDestroy(start);
            return false;
        }
    }

    float generation_ms = 0.0f;
    const bool timed = cuda_chk(cudaEventRecord(end), "Record end event") &&
                       cuda_chk(cudaEventSynchronize(end), "Wait for mip generation") &&
                       cuda_chk(cudaEventElapsedTime(&generation_ms, start, end), "Measure mip generation");

    cudaEventDestroy(end);
    cudaEventDestroy(start);

    if (!timed)
        return false;

    if (generation_time_ms)
        *generation_time_ms = generation_ms;
    else
        std::printf("Total generation time (cuda): %.3f ms\n", generation_ms);

    // Wait for completion and copy the generated mip levels back.
    const size_t offset = image.mips[1].byte_offset;
    const MipLevel &last = image.mips[image.mip_count - 1];
    const size_t bytes = last.byte_offset + last.byte_size - offset;

    return cuda_chk(cudaMemcpy(image.data + offset, device_data + offset, bytes, cudaMemcpyDeviceToHost),
                    "Download mip chain");
}
