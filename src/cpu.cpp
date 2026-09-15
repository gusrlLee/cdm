#include "mip.h"
#include "bc1.h"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

bool generate_mipmaps_cuda(Image *image);
bool prepare_mipmaps_cuda(const Image *image);

// Lightweight 4-lane float vector wrapper for SSE
struct v4f
{
    __m128 v;

    CDM_INLINE v4f() = default;
    CDM_INLINE v4f(__m128 x) : v(x) {}
    CDM_INLINE v4f(float s) : v(_mm_set1_ps(s)) {}

    // Basic arithmetic operators (Vertical SIMD)
    CDM_INLINE v4f operator+(const v4f &o) const { return _mm_add_ps(v, o.v); }
    CDM_INLINE v4f operator-(const v4f &o) const { return _mm_sub_ps(v, o.v); }
    CDM_INLINE v4f operator*(const v4f &o) const { return _mm_mul_ps(v, o.v); }

    CDM_INLINE v4f &operator+=(const v4f &o)
    {
        v = _mm_add_ps(v, o.v);
        return *this;
    }
};

CDM_INLINE v4f clamp01(const v4f &a)
{
    return _mm_min_ps(_mm_max_ps(a.v, _mm_setzero_ps()), _mm_set1_ps(1.0f));
}

CDM_INLINE v4f round(const v4f &a)
{
    // Round to nearest integer (SSE 4.1)
    return _mm_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

CDM_INLINE v4f reciprocal(const v4f &a)
{
    __m128 estimate = _mm_rcp_ps(a.v);
    return _mm_mul_ps(estimate, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(a.v, estimate)));
}

// 4-block parallel types using our existing templates
using Float3x4 = Vec3T<v4f>;
using SymMat3x4 = SymMat3T<v4f>;

// Principal axis for four blocks in parallel
CDM_INLINE Float3x4 compute_principal_axis(const SymMat3x4 &cov)
{
    const v4f inv_sqrt3 = v4f(0.57735027f);
    Float3x4 axis0 = {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    Float3x4 next0 = cov.multiply(axis0);
    v4f lsq0 = length_sq(next0);

    // The normal path needs one matrix-vector product. Alternate seeds are only
    // evaluated for the rare covariance orthogonal to (1,1,1).
    __m128 degenerate0 = _mm_cmplt_ps(lsq0.v, _mm_set1_ps(1e-9f));
    if (_mm_movemask_ps(degenerate0) == 0)
    {
        v4f inv_len = _mm_rsqrt_ps(_mm_max_ps(lsq0.v, _mm_set1_ps(1e-20f)));
        return next0 * inv_len;
    }

    const v4f inv_sqrt2 = v4f(0.70710678f);
    Float3x4 axis1 = {inv_sqrt2, v4f(-0.70710678f), v4f(0.0f)};
    Float3x4 next1 = cov.multiply(axis1);
    v4f lsq1 = length_sq(next1);

    // Branchless blend: if lsq0 < 1e-9f, use Candidate 1
    __m128 mask0 = degenerate0;
    Float3x4 next;
    next.r = _mm_blendv_ps(next0.r.v, next1.r.v, mask0);
    next.g = _mm_blendv_ps(next0.g.v, next1.g.v, mask0);
    next.b = _mm_blendv_ps(next0.b.v, next1.b.v, mask0);
    v4f lsq = _mm_blendv_ps(lsq0.v, lsq1.v, mask0);

    // Candidate 2: (0, 1, -1) / sqrt(2)
    __m128 mask1 = _mm_cmplt_ps(lsq.v, _mm_set1_ps(1e-9f));
    Float3x4 axis2 = {v4f(0.0f), inv_sqrt2, v4f(-0.70710678f)};
    Float3x4 next2 = cov.multiply(axis2);
    v4f lsq2 = length_sq(next2);

    next.r = _mm_blendv_ps(next.r.v, next2.r.v, mask1);
    next.g = _mm_blendv_ps(next.g.v, next2.g.v, mask1);
    next.b = _mm_blendv_ps(next.b.v, next2.b.v, mask1);
    lsq = _mm_blendv_ps(lsq.v, lsq2.v, mask1);

    // Normalize
    v4f inv_len = _mm_rsqrt_ps(_mm_max_ps(lsq.v, _mm_set1_ps(1e-20f)));
    Float3x4 result = next * inv_len;

    // Default flat blocks to (1, 0, 0)
    __m128 flat_mask = _mm_cmplt_ps(lsq.v, _mm_set1_ps(1e-20f));
    result.r = _mm_blendv_ps(result.r.v, _mm_set1_ps(1.0f), flat_mask);
    result.g = _mm_blendv_ps(result.g.v, _mm_setzero_ps(), flat_mask);
    result.b = _mm_blendv_ps(result.b.v, _mm_setzero_ps(), flat_mask);
    return result;
}

namespace bc1
{
    // Four-block SIMD decode path
    struct PaletteBatch
    {
        Float3x4 colors[4];
    };

    CDM_INLINE __m128i load_block_field_x4(const Block64 &b0, const Block64 &b1, const Block64 &b2, const Block64 &b3, bool color1)
    {
        return _mm_set_epi32(color1 ? b3.c1 : b3.c0, color1 ? b2.c1 : b2.c0,
                             color1 ? b1.c1 : b1.c0, color1 ? b0.c1 : b0.c0);
    }

    template <bool IsSrgb>
    CDM_INLINE v4f rgb8_to_float(__m128i value)
    {
        if constexpr (!IsSrgb)
            return _mm_mul_ps(_mm_cvtepi32_ps(value), _mm_set1_ps(1.0f / 255.0f));

        alignas(16) uint32_t lanes[4];
        _mm_store_si128((__m128i *)lanes, value);
        return _mm_set_ps(c_srgb_to_linear[lanes[3]], c_srgb_to_linear[lanes[2]], c_srgb_to_linear[lanes[1]], c_srgb_to_linear[lanes[0]]);
    }

    template <bool IsSrgb, bool Opaque = false>
    CDM_INLINE PaletteBatch get_palette_x4(__m128i c0, __m128i c1)
    {
        const __m128i mask5 = _mm_set1_epi32(31);
        const __m128i mask6 = _mm_set1_epi32(63);
        auto expand5 = [mask5](__m128i c, int shift) {
            __m128i v = _mm_and_si128(_mm_srl_epi32(c, _mm_cvtsi32_si128(shift)), mask5);
            return _mm_or_si128(_mm_slli_epi32(v, 3), _mm_srli_epi32(v, 2));
        };
        auto expand6 = [mask6](__m128i c) {
            __m128i v = _mm_and_si128(_mm_srli_epi32(c, 5), mask6);
            return _mm_or_si128(_mm_slli_epi32(v, 2), _mm_srli_epi32(v, 4));
        };
        auto div3 = [](__m128i v) { return _mm_srli_epi32(_mm_mullo_epi32(v, _mm_set1_epi32(683)), 11); };

        __m128i channel0[3] = {expand5(c0, 11), expand6(c0), expand5(c0, 0)};
        __m128i channel1[3] = {expand5(c1, 11), expand6(c1), expand5(c1, 0)};
        __m128i four_color;
        if constexpr (Opaque) four_color = _mm_set1_epi32(-1);
        else four_color = _mm_cmpgt_epi32(c0, c1);
        PaletteBatch result;
        for (int channel = 0; channel < 3; ++channel)
        {
            __m128i c2_4 = div3(_mm_add_epi32(_mm_slli_epi32(channel0[channel], 1), channel1[channel]));
            __m128i c3_4 = div3(_mm_add_epi32(channel0[channel], _mm_slli_epi32(channel1[channel], 1)));
            __m128i c2, c3;
            if constexpr (Opaque)
            {
                c2 = c2_4;
                c3 = c3_4;
            }
            else
            {
                __m128i c2_3 = _mm_srli_epi32(_mm_add_epi32(channel0[channel], channel1[channel]), 1);
                c2 = _mm_blendv_epi8(c2_3, c2_4, four_color);
                c3 = _mm_and_si128(c3_4, four_color);
            }
            v4f values[4] = {rgb8_to_float<IsSrgb>(channel0[channel]), rgb8_to_float<IsSrgb>(channel1[channel]),
                             rgb8_to_float<IsSrgb>(c2), rgb8_to_float<IsSrgb>(c3)};
            for (int selector = 0; selector < 4; ++selector)
            {
                if (channel == 0) result.colors[selector].r = values[selector];
                else if (channel == 1) result.colors[selector].g = values[selector];
                else result.colors[selector].b = values[selector];
            }
        }
        return result;
    }

    CDM_INLINE __m128i count_2x2_regions_x4(__m128i flags)
    {
        const __m128i horizontal_mask = _mm_set1_epi32(0x11111111u);
        __m128i horizontal = _mm_add_epi32(_mm_and_si128(flags, horizontal_mask),
                                           _mm_and_si128(_mm_srli_epi32(flags, 2), horizontal_mask));
        const __m128i vertical_mask = _mm_set1_epi32(0x00ff00ffu);
        return _mm_add_epi32(_mm_and_si128(horizontal, vertical_mask),
                             _mm_and_si128(_mm_srli_epi32(horizontal, 8), vertical_mask));
    }

    template <int Shift>
    CDM_INLINE Float3x4 quadrant_mean_x4(const PaletteBatch &palette, const __m128i hist[4])
    {
        Float3x4 result = Float3x4::zero();
        const __m128 scale = _mm_set1_ps(0.25f);
        const __m128i mask = _mm_set1_epi32(15);
        for (int selector = 0; selector < 4; ++selector)
        {
            __m128i count = _mm_and_si128(_mm_srli_epi32(hist[selector], Shift), mask);
            v4f weight = _mm_mul_ps(_mm_cvtepi32_ps(count), scale);
            result += palette.colors[selector] * weight;
        }
        return result;
    }

    template <bool IsSrgb>
    CDM_INLINE void get_quadrant_means_x4(const Block64 &b0, const Block64 &b1, const Block64 &b2, const Block64 &b3,
                                          Float3x4 out[4])
    {
        __m128i c0 = load_block_field_x4(b0, b1, b2, b3, false);
        __m128i c1 = load_block_field_x4(b0, b1, b2, b3, true);
        PaletteBatch palette = get_palette_x4<IsSrgb>(c0, c1);
        __m128i indices = _mm_set_epi32(b3.indices, b2.indices, b1.indices, b0.indices);
        
        const __m128i low_mask = _mm_set1_epi32(0x55555555u);
        __m128i low = _mm_and_si128(indices, low_mask);
        __m128i high = _mm_and_si128(_mm_srli_epi32(indices, 1), low_mask);
        __m128i hist[4] = {
            count_2x2_regions_x4(_mm_andnot_si128(_mm_or_si128(low, high), low_mask)),
            count_2x2_regions_x4(_mm_andnot_si128(high, low)),
            count_2x2_regions_x4(_mm_andnot_si128(low, high)),
            count_2x2_regions_x4(_mm_and_si128(low, high))};
        out[0] = quadrant_mean_x4<0>(palette, hist);
        out[1] = quadrant_mean_x4<4>(palette, hist);
        out[2] = quadrant_mean_x4<16>(palette, hist);
        out[3] = quadrant_mean_x4<20>(palette, hist);
    }

    // Four-block SIMD encoder
    template <bool IsSrgb>
    CDM_INLINE void pack_child_blocks_x4(const Float3x4 samples[16], const Float3x4 &p0, const Float3x4 &p1, Block64 out[4])
    {
        alignas(16) float p0r[4], p0g[4], p0b[4], p1r[4], p1g[4], p1b[4];
        _mm_store_ps(p0r, p0.r.v); _mm_store_ps(p0g, p0.g.v); _mm_store_ps(p0b, p0.b.v);
        _mm_store_ps(p1r, p1.r.v); _mm_store_ps(p1g, p1.g.v); _mm_store_ps(p1b, p1.b.v);

        alignas(16) uint32_t c0[4], c1[4];
        for (int lane = 0; lane < 4; ++lane)
        {
            c0[lane] = encode_rgb565_fast<IsSrgb>({p0r[lane], p0g[lane], p0b[lane]});
            c1[lane] = encode_rgb565_fast<IsSrgb>({p1r[lane], p1g[lane], p1b[lane]});
            if (c0[lane] < c1[lane]) std::swap(c0[lane], c1[lane]);
            else if (c0[lane] == c1[lane])
            {
                if (c1[lane] > 0) --c1[lane];
                else ++c0[lane];
            }
        }

        PaletteBatch palette = get_palette_x4<IsSrgb, true>(_mm_load_si128((const __m128i *)c0), _mm_load_si128((const __m128i *)c1));
        __m128i packed_indices = _mm_setzero_si128();
        for (int i = 0; i < 16; ++i)
        {
            v4f best_distance(1e30f);
            __m128i best_selector = _mm_setzero_si128();
            for (int selector = 0; selector < 4; ++selector)
            {
                v4f distance = length_sq(samples[i] - palette.colors[selector]);
                __m128 mask = _mm_cmplt_ps(distance.v, best_distance.v);
                best_distance = _mm_blendv_ps(best_distance.v, distance.v, mask);
                best_selector = _mm_blendv_epi8(best_selector, _mm_set1_epi32(selector), _mm_castps_si128(mask));
            }
            packed_indices = _mm_or_si128(packed_indices,
                _mm_sll_epi32(best_selector, _mm_cvtsi32_si128(2 * texel_map[i])));
        }

        alignas(16) uint32_t indices[4];
        _mm_store_si128((__m128i *)indices, packed_indices);
        for (int lane = 0; lane < 4; ++lane)
            out[lane] = Block64{(uint16_t)c0[lane], (uint16_t)c1[lane], indices[lane]};
    }

    template <bool IsSrgb>
    CDM_INLINE void encode_samples_x4(const Float3x4 samples[16], Block64 *out_blocks)
    {
        constexpr float inv3 = 1.0f / 3.0f;
        Float3x4 mean;
        SymMat3x4 cov;
        compute_child_moments(samples, mean, cov);

        // 4. Principal Axis (PCA) for 4 blocks at once!
        Float3x4 axis = compute_principal_axis(cov);

        v4f min_proj = v4f(1e30f);
        v4f max_proj = v4f(-1e30f);
        for (int i = 0; i < 16; ++i)
        {
            v4f proj = dot(samples[i] - mean, axis);
            min_proj = _mm_min_ps(min_proj.v, proj.v);
            max_proj = _mm_max_ps(max_proj.v, proj.v);
        }

        Float3x4 p0 = mean + axis * min_proj;
        Float3x4 p1 = mean + axis * max_proj;

        // 5. Refine endpoints using Least-Squares in parallel
        Float3x4 dir = p1 - p0;
        v4f len_sq = length_sq(dir) + v4f(1e-12f);
        v4f inv_len_sq = reciprocal(len_sq);

        v4f sum_w = v4f(0.0f);
        v4f sum_w2 = v4f(0.0f);
        Float3x4 sum_y = Float3x4::zero();
        Float3x4 sum_wy = Float3x4::zero();

        for (int i = 0; i < 16; ++i)
        {
            v4f t = dot(samples[i] - p0, dir) * inv_len_sq;
            t = clamp01(t);
            v4f w = round(t * v4f(3.0f)) * v4f(inv3);
            sum_w += w;
            sum_w2 += w * w;
            sum_y += samples[i];
            sum_wy += samples[i] * w;
        }

        // Solve 2x2 normal equations for 4 blocks simultaneously
        v4f det = v4f(16.0f) * sum_w2 - sum_w * sum_w;
        __m128 mask = _mm_cmplt_ps(det.v, _mm_set1_ps(1e-6f));
        v4f safe_det = _mm_blendv_ps(det.v, _mm_set1_ps(1.0f), mask);
        v4f inv_det = reciprocal(safe_det);

        Float3x4 solved_p0 = (sum_y * sum_w2 - sum_wy * sum_w) * inv_det;
        Float3x4 solve_dir = (sum_wy * v4f(16.0f) - sum_y * sum_w) * inv_det;
        Float3x4 solved_p1 = solved_p0 + solve_dir;

        // Fall back to mean if det is near zero
        p0.r = _mm_blendv_ps(solved_p0.r.v, mean.r.v, mask);
        p0.g = _mm_blendv_ps(solved_p0.g.v, mean.g.v, mask);
        p0.b = _mm_blendv_ps(solved_p0.b.v, mean.b.v, mask);

        p1.r = _mm_blendv_ps(solved_p1.r.v, mean.r.v, mask);
        p1.g = _mm_blendv_ps(solved_p1.g.v, mean.g.v, mask);
        p1.b = _mm_blendv_ps(solved_p1.b.v, mean.b.v, mask);

        pack_child_blocks_x4<IsSrgb>(samples, p0, p1, out_blocks);
    }

    CDM_INLINE Float3x4 block_means_x4(const Float3x4 quadrants[4])
    {
        return (quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3]) * v4f(0.25f);
    }

    CDM_INLINE void store_block_mean_pairs_x4(const Float3x4 even_quadrants[4], const Float3x4 odd_quadrants[4], const MeanImage &destination, uint32_t x, uint32_t y)
    {
        Float3x4 even = block_means_x4(even_quadrants);
        Float3x4 odd = block_means_x4(odd_quadrants);
        size_t index = (size_t)y * destination.width + x;
        _mm_storeu_ps(destination.r + index, _mm_unpacklo_ps(even.r.v, odd.r.v));
        _mm_storeu_ps(destination.r + index + 4, _mm_unpackhi_ps(even.r.v, odd.r.v));
        _mm_storeu_ps(destination.g + index, _mm_unpacklo_ps(even.g.v, odd.g.v));
        _mm_storeu_ps(destination.g + index + 4, _mm_unpackhi_ps(even.g.v, odd.g.v));
        _mm_storeu_ps(destination.b + index, _mm_unpacklo_ps(even.b.v, odd.b.v));
        _mm_storeu_ps(destination.b + index + 4, _mm_unpackhi_ps(even.b.v, odd.b.v));
    }

    template <bool IsSrgb>
    CDM_INLINE void generate_child_blocks_x4_impl(const Block64 *src0, const Block64 *src1, Block64 *out_blocks,
                                                   MeanImage *means, uint32_t source_x, uint32_t source_y0, uint32_t source_y1)
    {
        Float3x4 samples[16];
        get_quadrant_means_x4<IsSrgb>(src0[0], src0[2], src0[4], src0[6], samples + 0);
        get_quadrant_means_x4<IsSrgb>(src0[1], src0[3], src0[5], src0[7], samples + 4);
        get_quadrant_means_x4<IsSrgb>(src1[0], src1[2], src1[4], src1[6], samples + 8);
        get_quadrant_means_x4<IsSrgb>(src1[1], src1[3], src1[5], src1[7], samples + 12);
        if (means)
        {
            store_block_mean_pairs_x4(samples + 0, samples + 4, *means, source_x, source_y0);
            store_block_mean_pairs_x4(samples + 8, samples + 12, *means, source_x, source_y1);
        }
        encode_samples_x4<IsSrgb>(samples, out_blocks);
    }

    CDM_INLINE void generate_child_blocks_x4(const Block64 *src0, const Block64 *src1, Block64 *out_blocks, bool is_srgb,
                                              MeanImage *means = nullptr, uint32_t source_x = 0,
                                              uint32_t source_y0 = 0, uint32_t source_y1 = 0)
    {
        if (is_srgb) generate_child_blocks_x4_impl<true>(src0, src1, out_blocks, means, source_x, source_y0, source_y1);
        else generate_child_blocks_x4_impl<false>(src0, src1, out_blocks, means, source_x, source_y0, source_y1);
    }

    template <bool IsSrgb>
    CDM_INLINE void generate_child_blocks_from_means_x4(const MeanImage &source,
                                                         uint32_t block_x, uint32_t block_y, uint32_t valid_lanes,
                                                         Block64 *destination)
    {
        Float3x4 samples[16];
        if (valid_lanes == 4 && (block_x + 4) * 4 <= source.width && (block_y + 1) * 4 <= source.height)
        {
            for (uint32_t i = 0; i < 16; ++i)
            {
                uint32_t local = texel_map[i];
                size_t index = (size_t)(block_y * 4 + (local >> 2)) * source.width + block_x * 4 + (local & 3u);
                samples[i] = {
                    _mm_set_ps(source.r[index + 12], source.r[index + 8], source.r[index + 4], source.r[index]),
                    _mm_set_ps(source.g[index + 12], source.g[index + 8], source.g[index + 4], source.g[index]),
                    _mm_set_ps(source.b[index + 12], source.b[index + 8], source.b[index + 4], source.b[index])};
            }
        }
        else
        {
            for (uint32_t i = 0; i < 16; ++i)
            {
                uint32_t local = texel_map[i];
                Float3 lane[4];
                for (uint32_t j = 0; j < 4; ++j)
                {
                    uint32_t lane_x = block_x + std::min(j, valid_lanes - 1);
                    lane[j] = source.get(lane_x * 4 + (local & 3u), block_y * 4 + (local >> 2));
                }
                samples[i] = {
                    _mm_set_ps(lane[3].r, lane[2].r, lane[1].r, lane[0].r),
                    _mm_set_ps(lane[3].g, lane[2].g, lane[1].g, lane[0].g),
                    _mm_set_ps(lane[3].b, lane[2].b, lane[1].b, lane[0].b)};
            }
        }
        Block64 encoded[4];
        encode_samples_x4<IsSrgb>(samples, encoded);
        for (uint32_t lane = 0; lane < valid_lanes; ++lane)
            destination[lane] = encoded[lane];
    }

    CDM_INLINE void generate_child_blocks_from_means_x4(const MeanImage &source,
                                                         uint32_t block_x, uint32_t block_y, uint32_t valid_lanes,
                                                         Block64 *destination, bool is_srgb)
    {
        if (is_srgb) generate_child_blocks_from_means_x4<true>(source, block_x, block_y, valid_lanes, destination);
        else generate_child_blocks_from_means_x4<false>(source, block_x, block_y, valid_lanes, destination);
    }
}

namespace
{
    class TaskDispatcher
    {
    public:
        TaskDispatcher()
        {
            const size_t hardware_threads = std::max(1u, std::thread::hardware_concurrency());
            workers_.reserve(hardware_threads - 1);
            for (size_t i = 1; i < hardware_threads; ++i)
                workers_.emplace_back([this] { worker_loop(); });
        }

        ~TaskDispatcher()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            work_ready_.notify_all();
            for (std::thread &worker : workers_)
                worker.join();
        }

        template <typename Function>
        void parallel_rows(uint32_t rows, size_t work_items, Function function)
        {
            constexpr size_t parallel_threshold = 2048;
            if (work_items < parallel_threshold || workers_.empty())
            {
                function(0, rows);
                return;
            }

            const uint32_t task_count = std::min<uint32_t>(rows, (uint32_t)workers_.size() + 1);
            const uint32_t rows_per_task = (rows + task_count - 1) / task_count;
            for (uint32_t begin = 0; begin < rows; begin += rows_per_task)
            {
                const uint32_t end = std::min(begin + rows_per_task, rows);
                enqueue([=] { function(begin, end); });
            }
            sync();
        }

    private:
        void enqueue(std::function<void()> task)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.emplace_back(std::move(task));
            if (queue_.size() > 1)
                work_ready_.notify_one();
        }

        void sync()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!queue_.empty())
            {
                std::function<void()> task = std::move(queue_.back());
                queue_.pop_back();
                ++active_jobs_;
                lock.unlock();
                task();
                lock.lock();
                --active_jobs_;
            }
            all_done_.wait(lock, [this] { return queue_.empty() && active_jobs_ == 0; });
        }

        void worker_loop()
        {
            for (;;)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty())
                    return;

                std::function<void()> task = std::move(queue_.back());
                queue_.pop_back();
                ++active_jobs_;
                lock.unlock();
                task();
                lock.lock();
                --active_jobs_;
                if (queue_.empty() && active_jobs_ == 0)
                    all_done_.notify_all();
            }
        }

        std::vector<std::thread> workers_;
        std::vector<std::function<void()>> queue_;
        std::mutex mutex_;
        std::condition_variable work_ready_;
        std::condition_variable all_done_;
        size_t active_jobs_ = 0;
        bool stopping_ = false;
    };

    TaskDispatcher g_dispatcher;

    bc1::MeanImage make_mean_image(float *storage, size_t capacity, uint32_t width, uint32_t height)
    {
        return {storage, storage + capacity, storage + capacity * 2, width, height};
    }

    void downsample_channel_scalar(const float *source, uint32_t source_width, uint32_t source_height,
                                   float *destination, uint32_t destination_width, uint32_t destination_height)
    {
        for (uint32_t y = 0; y < destination_height; ++y)
        {
            const uint32_t y0 = y * 2;
            const uint32_t y1 = std::min(y0 + 1, source_height - 1);
            const float *row0 = source + (size_t)y0 * source_width;
            const float *row1 = source + (size_t)y1 * source_width;
            for (uint32_t x = 0; x < destination_width; ++x)
            {
                const uint32_t x0 = x * 2;
                const uint32_t x1 = std::min(x0 + 1, source_width - 1);
                destination[(size_t)y * destination_width + x] =
                    (row0[x0] + row0[x1] + row1[x0] + row1[x1]) * 0.25f;
            }
        }
    }

    void downsample_channel_simd(const float *source, uint32_t source_width, uint32_t source_height,
                                 float *destination, uint32_t destination_width, uint32_t destination_height)
    {
        for (uint32_t y = 0; y < destination_height; ++y)
        {
            const uint32_t y0 = y * 2;
            const uint32_t y1 = std::min(y0 + 1, source_height - 1);
            const float *row0 = source + (size_t)y0 * source_width;
            const float *row1 = source + (size_t)y1 * source_width;
            float *output = destination + (size_t)y * destination_width;
            uint32_t x = 0;
            for (; x + 3 < destination_width && x * 2 + 7 < source_width; x += 4)
            {
                const uint32_t sx = x * 2;
                __m128 top = _mm_hadd_ps(_mm_loadu_ps(row0 + sx), _mm_loadu_ps(row0 + sx + 4));
                __m128 bottom = _mm_hadd_ps(_mm_loadu_ps(row1 + sx), _mm_loadu_ps(row1 + sx + 4));
                _mm_storeu_ps(output + x, _mm_mul_ps(_mm_add_ps(top, bottom), _mm_set1_ps(0.25f)));
            }
            for (; x < destination_width; ++x)
            {
                const uint32_t x0 = x * 2;
                const uint32_t x1 = std::min(x0 + 1, source_width - 1);
                output[x] = (row0[x0] + row0[x1] + row1[x0] + row1[x1]) * 0.25f;
            }
        }
    }

    template <bool Simd>
    void downsample_means(const bc1::MeanImage &source, const bc1::MeanImage &destination)
    {
        auto downsample = Simd ? downsample_channel_simd : downsample_channel_scalar;
        downsample(source.r, source.width, source.height, destination.r, destination.width, destination.height);
        downsample(source.g, source.width, source.height, destination.g, destination.width, destination.height);
        downsample(source.b, source.width, source.height, destination.b, destination.width, destination.height);
    }

    template <bool Simd>
    bool generate_cpu(Image *image)
    {
        const uint32_t base_width = image->mips[0].block_count_x;
        const uint32_t base_height = image->mips[0].block_count_y;
        const size_t base_capacity = (size_t)base_width * base_height;
        
        const uint32_t scratch_width = (base_width + 1) / 2;
        const uint32_t scratch_height = (base_height + 1) / 2;
        const size_t scratch_capacity = (size_t)scratch_width * scratch_height;

        std::unique_ptr<float[]> base_storage(new (std::nothrow) float[base_capacity * 3]);
        std::unique_ptr<float[]> scratch_storage(new (std::nothrow) float[scratch_capacity * 3]);
        if (!base_storage || !scratch_storage)
            return false;

        bc1::MeanImage means = make_mean_image(base_storage.get(), base_capacity, base_width, base_height);
        bc1::MeanImage scratch = make_mean_image(scratch_storage.get(), scratch_capacity, scratch_width, scratch_height);

        for (uint32_t level = 1; level < image->mip_count; ++level)
        {
            const MipLevel previous = image->mips[level - 1];
            const MipLevel current = image->mips[level];

            if (level >= 3)
            {
                scratch.width = (means.width + 1) / 2;
                scratch.height = (means.height + 1) / 2;
                downsample_means<Simd>(means, scratch);
                std::swap(means, scratch);
            }

            const Block64 *source = reinterpret_cast<const Block64 *>(image->data + previous.byte_offset);
            Block64 *destination = reinterpret_cast<Block64 *>(image->data + current.byte_offset);
            auto process_rows = [=, &means](uint32_t begin, uint32_t end)
            {
                for (uint32_t by = begin; by < end; ++by)
                {
                    Block64 *output = destination + (size_t)by * current.block_count_x;
                    if (level >= 2)
                    {
                        if constexpr (Simd)
                        {
                            for (uint32_t bx = 0; bx < current.block_count_x; bx += 4)
                            {
                                const uint32_t lanes = std::min(4u, current.block_count_x - bx);
                                bc1::generate_child_blocks_from_means_x4(means, bx, by, lanes, output + bx, image->is_srgb);
                            }
                        }
                        else
                        {
                            for (uint32_t bx = 0; bx < current.block_count_x; ++bx)
                                output[bx] = bc1::generate_child_block_from_means_scalar(means, bx, by, image->is_srgb);
                        }
                        continue;
                    }

                    const uint32_t py0 = by * 2;
                    const uint32_t py1 = std::min(py0 + 1, previous.block_count_y - 1);
                    const Block64 *row0 = source + (size_t)py0 * previous.block_count_x;
                    const Block64 *row1 = source + (size_t)py1 * previous.block_count_x;
                    uint32_t bx = 0;
                    if constexpr (Simd)
                    {
                        for (; bx + 3 < current.block_count_x && (bx + 3) * 2 + 1 < previous.block_count_x; bx += 4)
                            bc1::generate_child_blocks_x4(row0 + bx * 2, row1 + bx * 2, output + bx,
                                image->is_srgb, &means, bx * 2, py0, py1);
                    }
                    for (; bx < current.block_count_x; ++bx)
                    {
                        const uint32_t px0 = bx * 2;
                        const uint32_t px1 = std::min(px0 + 1, previous.block_count_x - 1);
                        Float3 parent_means[4];
                        output[bx] = bc1::generate_child_block_scalar(row0[px0], row0[px1], row1[px0], row1[px1],
                                                               image->is_srgb, parent_means);
                        means.set(px0, py0, parent_means[0]);
                        means.set(px1, py0, parent_means[1]);
                        means.set(px0, py1, parent_means[2]);
                        means.set(px1, py1, parent_means[3]);
                    }
                }
            };

            g_dispatcher.parallel_rows(current.block_count_y,
                (size_t)current.block_count_x * current.block_count_y, process_rows);
        }
        return true;
    }
}

bool generate_mipmaps(Image *image, const Options &options)
{
    if (!image || !image->data || image->format != Format::BC1)
        return false;

    switch (options.backend)
    {
    case Backend::CPU:
        return generate_cpu<false>(image);
    case Backend::CPU_SIMD:
        return generate_cpu<true>(image);
    case Backend::CUDA:
        return generate_mipmaps_cuda(image);
    }
    return false;
}

bool prepare_mipmap_backend(const Image *image, Backend backend)
{
    return backend != Backend::CUDA || prepare_mipmaps_cuda(image);
}
