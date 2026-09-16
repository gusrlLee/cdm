#include "mip.h"
#include "bc1.h"
#include "bc6h.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

using Color = Float3;
using MeanImage = bc1::MeanImage;
static_assert(sizeof(Block64) == 8);

__constant__ float g_srgb_to_linear[256];
__constant__ float g_threshold31[31];
__constant__ float g_threshold63[63];

static_assert(bc1::texel_index(0) == 0 && bc1::texel_index(3) == 5
    && bc1::texel_index(10) == 12 && bc1::texel_index(15) == 15);

// ---------------------------------------------------------------------
// CUDA utilities
// ---------------------------------------------------------------------

static __device__ __forceinline__ uint32_t clamp_index(uint32_t value, uint32_t limit)
{
    return value < limit ? value : limit - 1;
}

template <bool Srgb>
static __device__ __forceinline__ float decode_channel(uint32_t value)
{
    if constexpr (Srgb) return g_srgb_to_linear[value];
    else return float(value) * (1.0f / 255.0f);
}

template <bool Srgb, bool Opaque = false>
static __device__ __forceinline__ Color palette_color(uint16_t c0, uint16_t c1, uint32_t selector)
{
    bc1::Rgb8 color = bc1::palette_color_rgb8<Opaque>(c0, c1, selector);
    return {decode_channel<Srgb>(color.r), decode_channel<Srgb>(color.g), decode_channel<Srgb>(color.b)};
}

// ---------------------------------------------------------------------
// CUDA-only reduction / warp helpers
// ---------------------------------------------------------------------

template <bool Srgb>
static __device__ __forceinline__ Color quadrant_mean(const Block64 &block, uint32_t quadrant)
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

static __device__ __forceinline__ unsigned half_warp_mask()
{
    return 0xffffu << (threadIdx.x & 16);
}

static __device__ __forceinline__ float sum4(float value, unsigned mask)
{
    value += __shfl_xor_sync(mask, value, 1, 4);
    value += __shfl_xor_sync(mask, value, 2, 4);
    return value;
}

static __device__ __forceinline__ float sum16(float value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset, 16);
    return __shfl_sync(mask, value, 0, 16);
}

static __device__ __forceinline__ float min16(float value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value = fminf(value, __shfl_down_sync(mask, value, offset, 16));
    return __shfl_sync(mask, value, 0, 16);
}

static __device__ __forceinline__ float max16(float value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(mask, value, offset, 16));
    return __shfl_sync(mask, value, 0, 16);
}

static __device__ __forceinline__ uint32_t or16(uint32_t value, unsigned mask)
{
    for (int offset = 8; offset; offset >>= 1)
        value |= __shfl_down_sync(mask, value, offset, 16);
    return __shfl_sync(mask, value, 0, 16);
}

template <uint32_t MaxValue>
static __device__ __forceinline__ uint32_t quantize_channel(float value)
{
    const float *thresholds = MaxValue == 31 ? g_threshold31 : g_threshold63;
    return bc1::quantize_with_thresholds<MaxValue>(value, thresholds);
}

template <bool Srgb>
static __device__ __forceinline__ uint16_t encode_rgb565(Color color)
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

// Encode one child BC1 block from 16 per-lane samples spread across a half-warp.
static __device__ __forceinline__ SymMat3 compute_half_warp_moments(Color sample, Color &mean)
{
    const unsigned mask = half_warp_mask();
    mean = {sum16(sample.r, mask) * (1.0f / 16.0f),
                   sum16(sample.g, mask) * (1.0f / 16.0f),
                   sum16(sample.b, mask) * (1.0f / 16.0f)};

    Color delta = sample - mean;
    float rr = sum16(delta.r * delta.r, mask) * (1.0f / 16.0f);
    float gg = sum16(delta.g * delta.g, mask) * (1.0f / 16.0f);
    float bb = sum16(delta.b * delta.b, mask) * (1.0f / 16.0f);
    float rg = sum16(delta.r * delta.g, mask) * (1.0f / 16.0f);
    float rb = sum16(delta.r * delta.b, mask) * (1.0f / 16.0f);
    float gb = sum16(delta.g * delta.b, mask) * (1.0f / 16.0f);

    return {rr, gg, bb, rg, rb, gb};
}

template <bool Srgb>
static __device__ __forceinline__ void encode_block_half_warp(Color sample,
                                                             uint32_t sample_index, Block64 *output)
{
    const unsigned mask = half_warp_mask();
    Color mean;
    SymMat3 cov = compute_half_warp_moments(sample, mean);
    float rr = cov.rr, gg = cov.gg, bb = cov.bb, rg = cov.rg, rb = cov.rb, gb = cov.gb;
    bool flat = rr + gg + bb < bc1::kFlatVarianceEpsilon;
    Color p0, p1;
    Color axis = {1.0f, 0.0f, 0.0f};
    float sample_projection = 0.0f;
    if (flat)
    {
        // Flat blocks use their mean as both endpoints.
        p0 = mean;
        p1 = mean;
    }
    else
    {
        if (sample_index == 0)
            axis = compute_principal_axis(SymMat3{rr, gg, bb, rg, rb, gb});
        axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16),
                __shfl_sync(mask, axis.b, 0, 16)};

        sample_projection = dot(sample - mean, axis);
        float minimum = min16(sample_projection, mask);
        float maximum = max16(sample_projection, mask);
        p0 = mean + axis * minimum;
        p1 = mean + axis * maximum;
    }

    uint32_t c0 = 0, c1 = 0;
    if ((sample_index & 15u) == 0)
    {
        c0 = encode_rgb565<Srgb>(p0);
        c1 = encode_rgb565<Srgb>(p1);
        if (c0 < c1) { uint32_t swap = c0; c0 = c1; c1 = swap; }
    }
    c0 = __shfl_sync(mask, c0, 0, 16);
    c1 = __shfl_sync(mask, c1, 0, 16);

    Color owned_color = sample_index < 4
        ? palette_color<Srgb, true>((uint16_t)c0, (uint16_t)c1, sample_index) : Color{};
    uint32_t selector = 0;
    float best_distance = 1e30f;
    if (flat)
    {
        for (uint32_t i = 0; i < 4; ++i)
        {
            Color entry = {__shfl_sync(mask, owned_color.r, i, 16),
                           __shfl_sync(mask, owned_color.g, i, 16),
                           __shfl_sync(mask, owned_color.b, i, 16)};
            Color delta = sample - entry;
            float distance = dot(delta, delta);
            if (distance < best_distance) { best_distance = distance; selector = i; }
        }
    }
    else
    {
        float owned_projection = sample_index < 4 ? dot(owned_color - mean, axis) : 0.0f;
        for (uint32_t i = 0; i < 4; ++i)
        {
            float delta = sample_projection - __shfl_sync(mask, owned_projection, i, 16);
            float distance = delta * delta;
            if (distance < best_distance) { best_distance = distance; selector = i; }
        }
    }
    uint32_t indices = or16(selector << (2 * bc1::texel_index(sample_index)), mask);
    if ((sample_index & 15u) == 0)
        *output = {(uint16_t)c0, (uint16_t)c1, indices};
}

// ---------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------

// Level 1: derive child blocks directly from the base mip's BC1 blocks, and record
// the mean pyramid's base level along the way.
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
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t lane = bc1::repeat_small_sample(sample_index, valid_width, valid_height);
        sample = {__shfl_sync(mask, sample.r, lane, 16), __shfl_sync(mask, sample.g, lane, 16),
                  __shfl_sync(mask, sample.b, lane, 16)};
    }
    encode_block_half_warp<Srgb>(sample, sample_index, destination + output_index);
}

// Level 2+: derive child blocks from the mean pyramid AND emit next mip's means on the fly!
template <bool Srgb>
static __global__ void generate_mean_mip(MeanImage source, Block64 *destination,
                                  uint32_t destination_width, uint32_t destination_height,
                                  MeanImage next_means, uint32_t valid_width, uint32_t valid_height)
{
    uint32_t sample_index = threadIdx.x & 15u;
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    uint32_t output_count = destination_width * destination_height;

    if (output_index >= output_count)
        return;

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

    // Repeat only encoder inputs; the next linear means above retain their original box calculation.
    if (valid_width < 4 || valid_height < 4)
    {
        x = clamp_index(bc1::repeat_small_coordinate(output_x * 4 + (local & 3u), valid_width), source.width);
        y = clamp_index(bc1::repeat_small_coordinate(output_y * 4 + (local >> 2), valid_height), source.height);
        index = (size_t)y * source.width + x;
        sample = {source.r[index], source.g[index], source.b[index]};
    }
    encode_block_half_warp<Srgb>(sample, sample_index, destination + output_index);
}

static __device__ __forceinline__ Color bc6h_quadrant_mean(const bc6h::Block &block, uint32_t quadrant)
{
    unsigned mask = half_warp_mask();
    uint32_t source_lane = (threadIdx.x & 15u) & ~3u;
    Color means[4] = {};
    if (quadrant == 0)
    {
        Color pixels[16];
        bc6h::decode_block(block, pixels);
        for (uint32_t i = 0; i < 16; ++i) pixels[i] = bc6h::sanitize_hdr(pixels[i]);
        for (uint32_t q = 0; q < 4; ++q)
        {
            uint32_t x = (q & 1u) * 2u, y = (q >> 1u) * 2u;
            means[q] = (pixels[y * 4 + x] + pixels[y * 4 + x + 1] + pixels[(y + 1) * 4 + x]
                + pixels[(y + 1) * 4 + x + 1]) * 0.25f;
        }
    }
    Color result{};
    for (uint32_t q = 0; q < 4; ++q)
    {
        Color value = means[q];
        value = {__shfl_sync(mask, value.r, source_lane, 16), __shfl_sync(mask, value.g, source_lane, 16),
            __shfl_sync(mask, value.b, source_lane, 16)};
        if (quadrant == q) result = value;
    }
    return result;
}

static __device__ __forceinline__ void encode_bc6h_half_warp(Color sample, uint32_t lane, bc6h::Block *output)
{
    unsigned mask = half_warp_mask();
    sample = bc6h::sanitize_hdr(sample);
    Color mean;
    SymMat3 cov = compute_half_warp_moments(sample, mean);
    Color axis = {1.0f, 0.0f, 0.0f};
    if (lane == 0) axis = compute_principal_axis(cov);
    axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16),
        __shfl_sync(mask, axis.b, 0, 16)};
    float projection = dot(sample - mean, axis);
    float minimum = min16(projection, mask), maximum = max16(projection, mask);
    Color p0 = mean + axis * minimum, p1 = mean + axis * maximum;
    bool degenerate = false;
    if (lane == 0) degenerate = bc6h::hdr_covariance_degenerate(cov, mean);
    degenerate = __shfl_sync(mask, degenerate, 0, 16);
    if (degenerate) p0 = p1 = mean;
    uint32_t ep[2][3] = {};
    if (lane == 0)
    {
        ep[0][0] = bc6h::quantize_endpoint(p0.r); ep[0][1] = bc6h::quantize_endpoint(p0.g); ep[0][2] = bc6h::quantize_endpoint(p0.b);
        ep[1][0] = bc6h::quantize_endpoint(p1.r); ep[1][1] = bc6h::quantize_endpoint(p1.g); ep[1][2] = bc6h::quantize_endpoint(p1.b);
    }
    for (int e = 0; e < 2; ++e) for (int c = 0; c < 3; ++c) ep[e][c] = __shfl_sync(mask, ep[e][c], 0, 16);
    Color owned = bc6h::palette_color(ep, lane);
    float best = 1e30f;
    uint32_t selector = 0;
    for (uint32_t s = 0; s < 16; ++s)
    {
        Color palette = {__shfl_sync(mask, owned.r, s, 16), __shfl_sync(mask, owned.g, s, 16),
            __shfl_sync(mask, owned.b, s, 16)};
        float error = length_sq(sample - palette);
        if (error < best) { best = error; selector = s; }
    }
    bool reverse = __shfl_sync(mask, selector >= 8, 0, 16);
    if (reverse)
    {
        selector = 15u - selector;
        for (int c = 0; c < 3; ++c) { uint32_t t = ep[0][c]; ep[0][c] = ep[1][c]; ep[1][c] = t; }
    }
    uint32_t texel = bc1::texel_index(lane);
    uint64_t selector_bits = texel == 0 ? uint64_t(selector) << 1 : uint64_t(selector) << (4 * texel);
    selector_bits |= uint64_t(or16(uint32_t(selector_bits), mask));
    uint32_t high_half = or16(uint32_t(selector_bits >> 32), mask);
    if (lane == 0)
    {
        uint64_t low = 3u | (uint64_t(ep[0][0]) << 5) | (uint64_t(ep[0][1]) << 15)
            | (uint64_t(ep[0][2]) << 25) | (uint64_t(ep[1][0]) << 35)
            | (uint64_t(ep[1][1]) << 45) | (uint64_t(ep[1][2] & 0x1ffu) << 55);
        uint64_t high = uint64_t(ep[1][2] >> 9) | uint32_t(selector_bits) | (uint64_t(high_half) << 32);
        *output = {low, high};
    }
}

static __global__ void generate_bc6h_base_mip(const bc6h::Block *source, uint32_t source_width, uint32_t source_height,
    bc6h::Block *destination, uint32_t destination_width, uint32_t destination_height, MeanImage means,
    uint32_t valid_width, uint32_t valid_height)
{
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    if (output_index >= destination_width * destination_height) return;
    uint32_t lane = threadIdx.x & 15u;
    unsigned mask = half_warp_mask();
    uint32_t ox = lane == 0 ? output_index % destination_width : 0;
    uint32_t oy = lane == 0 ? output_index / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16); oy = __shfl_sync(mask, oy, 0, 16);
    uint32_t parent = lane >> 2, quadrant = lane & 3u;
    uint32_t x0 = clamp_index(ox * 2, source_width), x1 = clamp_index(x0 + 1, source_width);
    uint32_t y0 = clamp_index(oy * 2, source_height), y1 = clamp_index(y0 + 1, source_height);
    uint32_t sx = parent & 1u ? x1 : x0, sy = parent & 2u ? y1 : y0;
    Color sample = bc6h_quadrant_mean(source[sy * source_width + sx], quadrant);
    Color parent_mean = {sum4(sample.r, mask) * 0.25f, sum4(sample.g, mask) * 0.25f, sum4(sample.b, mask) * 0.25f};
    bool unique = parent == 0 || (parent == 1 && x1 != x0) || (parent == 2 && y1 != y0)
        || (parent == 3 && x1 != x0 && y1 != y0);
    if (quadrant == 0 && unique)
    {
        size_t i = size_t(sy) * means.width + sx;
        means.r[i] = parent_mean.r; means.g[i] = parent_mean.g; means.b[i] = parent_mean.b;
    }
    if (valid_width < 4 || valid_height < 4)
    {
        uint32_t source_lane = bc1::repeat_small_sample(lane, valid_width, valid_height);
        sample = {__shfl_sync(mask, sample.r, source_lane, 16), __shfl_sync(mask, sample.g, source_lane, 16),
            __shfl_sync(mask, sample.b, source_lane, 16)};
    }
    encode_bc6h_half_warp(sample, lane, destination + output_index);
}

static __global__ void generate_bc6h_mean_mip(MeanImage source, bc6h::Block *destination,
    uint32_t destination_width, uint32_t destination_height, MeanImage next_means,
    uint32_t valid_width, uint32_t valid_height)
{
    uint32_t lane = threadIdx.x & 15u;
    uint32_t output_index = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    if (output_index >= destination_width * destination_height) return;
    unsigned mask = half_warp_mask();
    uint32_t local = bc1::texel_index(lane);
    uint32_t ox = lane == 0 ? output_index % destination_width : 0;
    uint32_t oy = lane == 0 ? output_index / destination_width : 0;
    ox = __shfl_sync(mask, ox, 0, 16); oy = __shfl_sync(mask, oy, 0, 16);
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
            next_means.r[ni] = parent_mean.r; next_means.g[ni] = parent_mean.g; next_means.b[ni] = parent_mean.b;
        }
    }
    if (valid_width < 4 || valid_height < 4)
    {
        x = clamp_index(bc1::repeat_small_coordinate(ox * 4 + (local & 3u), valid_width), source.width);
        y = clamp_index(bc1::repeat_small_coordinate(oy * 4 + (local >> 2), valid_height), source.height);
        index = size_t(y) * source.width + x;
        sample = {source.r[index], source.g[index], source.b[index]};
    }
    encode_bc6h_half_warp(sample, lane, destination + output_index);
}

// ---------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------

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

template <size_t MaxValue>
static std::array<float, MaxValue> make_srgb_quantize_thresholds()
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
                destination, current.block_count_x, current.block_count_y, means, current.width, current.height);
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

            generate_mean_mip<Srgb><<<blocks, threads>>>(means, destination, current.block_count_x, current.block_count_y,
                next_means, current.width, current.height);

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
                destination, current.block_count_x, current.block_count_y, means, current.width, current.height);
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
                current.block_count_y, next_means, current.width, current.height);
            if (level + 1 < image->mip_count) std::swap(means, scratch);
        }
        if (cudaGetLastError() != cudaSuccess) return false;
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

    if (cudaMemcpy(g_workspace.data.get(), image->data, image->mips[0].byte_size, cudaMemcpyHostToDevice) != cudaSuccess)
        return false;

    MeanImage means = {g_workspace.base.get(), g_workspace.base.get() + base_count,
                       g_workspace.base.get() + base_count * 2, base_width, base_height};
    MeanImage scratch = {g_workspace.scratch.get(), g_workspace.scratch.get() + scratch_count,
                         g_workspace.scratch.get() + scratch_count * 2, scratch_width, scratch_height};

    bool generated = image->format == Format::BC6H_UF16
        ? generate_bc6h_cuda_impl(image, g_workspace.data.get(), means, scratch)
        : (image->is_srgb ? generate_cuda_impl<true>(image, g_workspace.data.get(), means, scratch)
                          : generate_cuda_impl<false>(image, g_workspace.data.get(), means, scratch));
    const size_t generated_offset = image->mips[1].byte_offset;
    return generated && cudaMemcpy(image->data + generated_offset, g_workspace.data.get() + generated_offset,
                                   image->data_size - generated_offset, cudaMemcpyDeviceToHost) == cudaSuccess;
}
