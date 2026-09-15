#include "mip.h"
#include "bc1.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace
{
    using Color = Float3;
    using MeanImage = bc1::MeanImage;
    static_assert(sizeof(Block64) == 8);

    __constant__ float g_srgb_to_linear[256];
    __constant__ float g_threshold31[31];
    __constant__ float g_threshold63[63];

    static_assert(bc1::texel_index(0) == 0 && bc1::texel_index(3) == 5
        && bc1::texel_index(10) == 12 && bc1::texel_index(15) == 15);

    __device__ __forceinline__ uint32_t clamp_index(uint32_t value, uint32_t limit)
    {
        return value < limit ? value : limit - 1;
    }

    template <bool Srgb>
    __device__ __forceinline__ float decode_channel(uint32_t value)
    {
        if constexpr (Srgb) return g_srgb_to_linear[value];
        else return float(value) * (1.0f / 255.0f);
    }

    template <bool Srgb, bool Opaque = false>
    __device__ __forceinline__ Color palette_color(uint16_t c0, uint16_t c1, uint32_t selector)
    {
        bc1::Rgb8 color = bc1::palette_color_rgb8<Opaque>(c0, c1, selector);
        return {decode_channel<Srgb>(color.r), decode_channel<Srgb>(color.g), decode_channel<Srgb>(color.b)};
    }

    template <bool Srgb>
    __device__ __forceinline__ Color quadrant_mean(const Block64 &block, uint32_t quadrant)
    {
        const unsigned mask = 0xffffu << (threadIdx.x & 16);
        const uint32_t group_start = (threadIdx.x & 15u) & ~3u;
        Color color = palette_color<Srgb>(block.c0, block.c1, quadrant);
        uint32_t counts = bc1::selector_region_counts(block.indices, quadrant);
        uint32_t shift = ((quadrant & 1u) << 2) | ((quadrant >> 1) << 4);
        Color result{};
        for (uint32_t selector = 0; selector < 4; ++selector)
        {
            uint32_t source_lane = group_start + selector;
            Color entry = {__shfl_sync(mask, color.r, source_lane, 16),
                           __shfl_sync(mask, color.g, source_lane, 16),
                           __shfl_sync(mask, color.b, source_lane, 16)};
            uint32_t entry_counts = __shfl_sync(mask, counts, source_lane, 16);
            float weight = float((entry_counts >> shift) & 15u) * 0.25f;
            result += entry * weight;
        }
        return result;
    }

    __device__ __forceinline__ unsigned half_warp_mask()
    {
        return 0xffffu << (threadIdx.x & 16);
    }

    __device__ __forceinline__ float sum4(float value, unsigned mask)
    {
        value += __shfl_xor_sync(mask, value, 1, 4);
        value += __shfl_xor_sync(mask, value, 2, 4);
        return value;
    }

    __device__ __forceinline__ float sum16(float value, unsigned mask)
    {
        for (int offset = 8; offset; offset >>= 1)
            value += __shfl_down_sync(mask, value, offset, 16);
        return __shfl_sync(mask, value, 0, 16);
    }

    __device__ __forceinline__ float min16(float value, unsigned mask)
    {
        for (int offset = 8; offset; offset >>= 1)
            value = fminf(value, __shfl_down_sync(mask, value, offset, 16));
        return __shfl_sync(mask, value, 0, 16);
    }

    __device__ __forceinline__ float max16(float value, unsigned mask)
    {
        for (int offset = 8; offset; offset >>= 1)
            value = fmaxf(value, __shfl_down_sync(mask, value, offset, 16));
        return __shfl_sync(mask, value, 0, 16);
    }

    __device__ __forceinline__ uint32_t or16(uint32_t value, unsigned mask)
    {
        for (int offset = 8; offset; offset >>= 1)
            value |= __shfl_down_sync(mask, value, offset, 16);
        return __shfl_sync(mask, value, 0, 16);
    }

    template <uint32_t MaxValue>
    __device__ __forceinline__ uint32_t quantize_channel(float value)
    {
        const float *thresholds = MaxValue == 31 ? g_threshold31 : g_threshold63;
        return bc1::quantize_with_thresholds<MaxValue>(value, thresholds);
    }

    template <bool Srgb>
    __device__ __forceinline__ uint16_t encode_rgb565(Color color)
    {
        uint32_t r, g, b;
        if constexpr (Srgb)
        {
            r = quantize_channel<31>(color.r);
            g = quantize_channel<63>(color.g);
            b = quantize_channel<31>(color.b);
        }
        else
        {
            r = uint32_t(fminf(fmaxf(color.r, 0.0f), 1.0f) * 31.0f + 0.5f);
            g = uint32_t(fminf(fmaxf(color.g, 0.0f), 1.0f) * 63.0f + 0.5f);
            b = uint32_t(fminf(fmaxf(color.b, 0.0f), 1.0f) * 31.0f + 0.5f);
        }
        return uint16_t((r << 11) | (g << 5) | b);
    }

    template <bool Srgb>
    __device__ __forceinline__ void encode_block_half_warp(Color sample, Color parent_mean,
                                                            uint32_t sample_index, Block64 *output)
    {
        const unsigned mask = half_warp_mask();
        Color mean = {sum16(sample.r, mask) * (1.0f / 16.0f),
                      sum16(sample.g, mask) * (1.0f / 16.0f),
                      sum16(sample.b, mask) * (1.0f / 16.0f)};

        Color within = sample - parent_mean;
        Color between = parent_mean - mean;
        float rr = sum16(within.r * within.r + between.r * between.r, mask) * (1.0f / 16.0f);
        float gg = sum16(within.g * within.g + between.g * between.g, mask) * (1.0f / 16.0f);
        float bb = sum16(within.b * within.b + between.b * between.b, mask) * (1.0f / 16.0f);
        float rg = sum16(within.r * within.g + between.r * between.g, mask) * (1.0f / 16.0f);
        float rb = sum16(within.r * within.b + between.r * between.b, mask) * (1.0f / 16.0f);
        float gb = sum16(within.g * within.b + between.g * between.b, mask) * (1.0f / 16.0f);

        Color axis{};
        if (sample_index == 0)
            axis = compute_principal_axis(SymMat3{rr, gg, bb, rg, rb, gb});
        axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16),
                __shfl_sync(mask, axis.b, 0, 16)};

        float projection = dot(sample - mean, axis);
        float minimum = min16(projection, mask);
        float maximum = max16(projection, mask);
        Color p0 = mean + axis * minimum;
        Color p1 = mean + axis * maximum;
        Color direction = p1 - p0;
        float inverse_length_sq = 1.0f / (dot(direction, direction) + 1e-12f);
        float t = fminf(fmaxf(dot(sample - p0, direction) * inverse_length_sq, 0.0f), 1.0f);
        float weight = rintf(t * 3.0f) * (1.0f / 3.0f);
        float weight_sum = sum16(weight, mask);
        float weight_sq_sum = sum16(weight * weight, mask);
        Color sum = {sum16(sample.r, mask), sum16(sample.g, mask), sum16(sample.b, mask)};
        Color weighted = {sum16(sample.r * weight, mask), sum16(sample.g * weight, mask), sum16(sample.b * weight, mask)};
        if (sample_index == 0)
            solve_least_squares_endpoints(weight_sum, weight_sq_sum, sum, weighted, mean, p0, p1);
        p0 = {__shfl_sync(mask, p0.r, 0, 16), __shfl_sync(mask, p0.g, 0, 16), __shfl_sync(mask, p0.b, 0, 16)};
        p1 = {__shfl_sync(mask, p1.r, 0, 16), __shfl_sync(mask, p1.g, 0, 16), __shfl_sync(mask, p1.b, 0, 16)};

        uint32_t c0 = 0, c1 = 0;
        if ((sample_index & 15u) == 0)
        {
            c0 = encode_rgb565<Srgb>(p0);
            c1 = encode_rgb565<Srgb>(p1);
            if (c0 < c1) { uint32_t swap = c0; c0 = c1; c1 = swap; }
            else if (c0 == c1) { if (c1) --c1; else ++c0; }
        }
        c0 = __shfl_sync(mask, c0, 0, 16);
        c1 = __shfl_sync(mask, c1, 0, 16);

        Color owned_color = sample_index < 4
            ? palette_color<Srgb, true>((uint16_t)c0, (uint16_t)c1, sample_index)
            : Color{};
        float best_distance = 1e30f;
        uint32_t selector = 0;
        for (uint32_t i = 0; i < 4; ++i)
        {
            Color entry = {__shfl_sync(mask, owned_color.r, i, 16),
                           __shfl_sync(mask, owned_color.g, i, 16),
                           __shfl_sync(mask, owned_color.b, i, 16)};
            Color delta = sample - entry;
            float distance = dot(delta, delta);
            if (distance < best_distance) { best_distance = distance; selector = i; }
        }
        uint32_t indices = or16(selector << (2 * bc1::texel_index(sample_index)), mask);
        if ((sample_index & 15u) == 0)
            *output = {(uint16_t)c0, (uint16_t)c1, indices};
    }

    template <bool Srgb>
    __global__ void generate_base_mip(const Block64 *source, uint32_t source_width, uint32_t source_height,
                                      Block64 *destination, uint32_t destination_width, uint32_t destination_height,
                                      MeanImage means)
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
        Color sample = quadrant_mean<Srgb>(source[source_y * source_width + source_x], quadrant);
        Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};

        bool unique_parent = parent == 0 || (parent == 1 && x1 != x0)
            || (parent == 2 && y1 != y0) || (parent == 3 && x1 != x0 && y1 != y0);
        if (quadrant == 0 && unique_parent)
        {
            size_t index = (size_t)source_y * means.width + source_x;
            means.r[index] = parent_mean.r;
            means.g[index] = parent_mean.g;
            means.b[index] = parent_mean.b;
        }
        encode_block_half_warp<Srgb>(sample, parent_mean, sample_index, destination + output_index);
    }

    template <bool Srgb>
    __global__ void generate_mean_mip(MeanImage source, Block64 *destination,
                                      uint32_t destination_width, uint32_t destination_height)
    {
        uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
        uint32_t output_count = destination_width * destination_height;
        if (output_index >= output_count)
            return;

        uint32_t sample_index = threadIdx.x & 15u;
        unsigned mask = half_warp_mask();
        uint32_t local = bc1::texel_index(sample_index);
        uint32_t output_x = sample_index == 0 ? output_index % destination_width : 0;
        uint32_t output_y = sample_index == 0 ? output_index / destination_width : 0;
        output_x = __shfl_sync(mask, output_x, 0, 16);
        output_y = __shfl_sync(mask, output_y, 0, 16);
        uint32_t x = clamp_index(output_x * 4 + (local & 3u), source.width);
        uint32_t y = clamp_index(output_y * 4 + (local >> 2), source.height);
        size_t index = (size_t)y * source.width + x;
        Color sample = {source.r[index], source.g[index], source.b[index]};
        Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};
        encode_block_half_warp<Srgb>(sample, parent_mean, sample_index, destination + output_index);
    }

    __global__ void downsample_mean_image(MeanImage source, MeanImage destination)
    {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= destination.width || y >= destination.height)
            return;
        size_t index = (size_t)y * destination.width + x;
        uint32_t x0 = x * 2;
        uint32_t x1 = clamp_index(x0 + 1, source.width);
        uint32_t y0 = y * 2;
        uint32_t y1 = clamp_index(y0 + 1, source.height);
        size_t i00 = (size_t)y0 * source.width + x0;
        size_t i10 = (size_t)y0 * source.width + x1;
        size_t i01 = (size_t)y1 * source.width + x0;
        size_t i11 = (size_t)y1 * source.width + x1;
        destination.r[index] = (source.r[i00] + source.r[i10] + source.r[i01] + source.r[i11]) * 0.25f;
        destination.g[index] = (source.g[i00] + source.g[i10] + source.g[i01] + source.g[i11]) * 0.25f;
        destination.b[index] = (source.b[i00] + source.b[i10] + source.b[i01] + source.b[i11]) * 0.25f;
    }

    template <typename T>
    class DeviceBuffer
    {
    public:
        ~DeviceBuffer() { if (data_) cudaFree(data_); }
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
        T *get() const { return data_; }
    private:
        T *data_ = nullptr;
        size_t capacity_ = 0;
    };

    bool initialize_tables()
    {
        static bool initialized = false;
        if (initialized)
            return true;
        static const std::array<float, 256> srgb = [] {
            std::array<float, 256> values{};
            for (size_t i = 0; i < values.size(); ++i)
            {
                float s = float(i) * (1.0f / 255.0f);
                values[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
            }
            return values;
        }();
        static const std::array<float, 31> threshold31 = [] {
            std::array<float, 31> values{};
            for (size_t i = 0; i < values.size(); ++i)
            {
                float s = (float(i) + 0.5f) / 31.0f;
                values[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
            }
            return values;
        }();
        static const std::array<float, 63> threshold63 = [] {
            std::array<float, 63> values{};
            for (size_t i = 0; i < values.size(); ++i)
            {
                float s = (float(i) + 0.5f) / 63.0f;
                values[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
            }
            return values;
        }();
        initialized = cudaMemcpyToSymbol(g_srgb_to_linear, srgb.data(), sizeof(srgb)) == cudaSuccess
            && cudaMemcpyToSymbol(g_threshold31, threshold31.data(), sizeof(threshold31)) == cudaSuccess
            && cudaMemcpyToSymbol(g_threshold63, threshold63.data(), sizeof(threshold63)) == cudaSuccess;
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
            const size_t scratch_count = (size_t)((image->mips[0].block_count_x + 1) / 2)
                * ((image->mips[0].block_count_y + 1) / 2);
            return data.allocate(image->data_size) && base.allocate(base_count * 3) && scratch.allocate(scratch_count * 3);
        }
    };

    CudaWorkspace g_workspace;

    template <bool Srgb>
    bool generate_cuda_impl(Image *image, uint8_t *device_data, MeanImage means, MeanImage scratch)
    {
        constexpr uint32_t threads = 256;
        for (uint32_t level = 1; level < image->mip_count; ++level)
        {
            const MipLevel &previous = image->mips[level - 1];
            const MipLevel &current = image->mips[level];
            if (level >= 3)
            {
                scratch.width = (means.width + 1) / 2;
                scratch.height = (means.height + 1) / 2;
                dim3 thread_grid(32, 8);
                dim3 block_grid((scratch.width + thread_grid.x - 1) / thread_grid.x,
                                (scratch.height + thread_grid.y - 1) / thread_grid.y);
                downsample_mean_image<<<block_grid, thread_grid>>>(means, scratch);
                std::swap(means, scratch);
            }

            auto *destination = reinterpret_cast<Block64 *>(device_data + current.byte_offset);
            uint32_t output_count = current.block_count_x * current.block_count_y;
            constexpr uint32_t outputs_per_block = threads / 16;
            uint32_t blocks = (output_count + outputs_per_block - 1) / outputs_per_block;
            if (level == 1)
            {
                const auto *source = reinterpret_cast<const Block64 *>(device_data + previous.byte_offset);
                generate_base_mip<Srgb><<<blocks, threads>>>(source, previous.block_count_x, previous.block_count_y,
                    destination, current.block_count_x, current.block_count_y, means);
            }
            else
            {
                generate_mean_mip<Srgb><<<blocks, threads>>>(means, destination,
                    current.block_count_x, current.block_count_y);
            }
            if (cudaGetLastError() != cudaSuccess)
                return false;
        }
        return true;
    }
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
    if (cudaMemcpy(g_workspace.data.get(), image->data, image->mips[0].byte_size, cudaMemcpyHostToDevice) != cudaSuccess)
        return false;

    MeanImage means = {g_workspace.base.get(), g_workspace.base.get() + base_count,
                       g_workspace.base.get() + base_count * 2, base_width, base_height};
    MeanImage scratch = {g_workspace.scratch.get(), g_workspace.scratch.get() + scratch_count,
                         g_workspace.scratch.get() + scratch_count * 2, scratch_width, scratch_height};
    bool generated = image->is_srgb
        ? generate_cuda_impl<true>(image, g_workspace.data.get(), means, scratch)
        : generate_cuda_impl<false>(image, g_workspace.data.get(), means, scratch);
    const size_t generated_offset = image->mips[1].byte_offset;
    return generated && cudaMemcpy(image->data + generated_offset, g_workspace.data.get() + generated_offset,
                                   image->data_size - generated_offset, cudaMemcpyDeviceToHost) == cudaSuccess;
}
