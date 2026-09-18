#include "bc1.h"
#include "bc6h.h"
#include "bc7.h"
#include "mip.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

bool generate_mipmaps_cuda(Image *image);
bool prepare_mipmaps_cuda(const Image *image);

void downsample_scalar(const float *source, uint32_t source_width, uint32_t source_height, float *destination,
                       uint32_t destination_width, uint32_t begin, uint32_t end)
{
    for (uint32_t y = begin; y < end; ++y)
    {
        const uint32_t y0 = y * 2, y1 = std::min(y0 + 1, source_height - 1);
        const float *row0 = source + size_t(y0) * source_width;
        const float *row1 = source + size_t(y1) * source_width;
        for (uint32_t x = 0; x < destination_width; ++x)
        {
            const uint32_t x0 = x * 2, x1 = std::min(x0 + 1, source_width - 1);
            destination[size_t(y) * destination_width + x] = (row0[x0] + row0[x1] + row1[x0] + row1[x1]) * 0.25f;
        }
    }
}

void downsample_simd(const float *source, uint32_t source_width, uint32_t source_height, float *destination,
                     uint32_t destination_width, uint32_t begin, uint32_t end)
{
    for (uint32_t y = begin; y < end; ++y)
    {
        const uint32_t y0 = y * 2, y1 = std::min(y0 + 1, source_height - 1);
        const float *row0 = source + size_t(y0) * source_width;
        const float *row1 = source + size_t(y1) * source_width;
        float *output = destination + size_t(y) * destination_width;
        uint32_t x = 0;
        for (; x + 3 < destination_width && x * 2 + 7 < source_width; x += 4)
        {
            const uint32_t sx = x * 2;
            const __m128 top = _mm_hadd_ps(_mm_loadu_ps(row0 + sx), _mm_loadu_ps(row0 + sx + 4));
            const __m128 bottom = _mm_hadd_ps(_mm_loadu_ps(row1 + sx), _mm_loadu_ps(row1 + sx + 4));
            _mm_storeu_ps(output + x, _mm_mul_ps(_mm_add_ps(top, bottom), _mm_set1_ps(0.25f)));
        }
        for (; x < destination_width; ++x)
        {
            const uint32_t x0 = x * 2, x1 = std::min(x0 + 1, source_width - 1);
            output[x] = (row0[x0] + row0[x1] + row1[x0] + row1[x1]) * 0.25f;
        }
    }
}

// The calling thread consumes queued work too, so small machines need no
// special path.
class TaskDispatcher
{
  public:
    TaskDispatcher()
    {
        const size_t count = std::max(1u, std::thread::hardware_concurrency());
        workers_.reserve(count - 1);
        for (size_t i = 1; i < count; ++i)
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

    template <typename Function> void parallel_rows(uint32_t rows, size_t work_items, Function function)
    {
        if (work_items < 2048 || workers_.empty())
        {
            function(0, rows);
            return;
        }
        const uint32_t tasks = std::min<uint32_t>(rows, uint32_t(workers_.size()) + 1);
        const uint32_t rows_per_task = (rows + tasks - 1) / tasks;
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
        work_ready_.notify_one();
    }

    void finish(std::unique_lock<std::mutex> &lock, std::function<void()> task)
    {
        ++active_jobs_;
        lock.unlock();
        task();
        lock.lock();
        --active_jobs_;
        if (queue_.empty() && active_jobs_ == 0)
            all_done_.notify_all();
    }

    void sync()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty())
        {
            auto task = std::move(queue_.back());
            queue_.pop_back();
            finish(lock, std::move(task));
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
            auto task = std::move(queue_.back());
            queue_.pop_back();
            finish(lock, std::move(task));
        }
    }

    std::vector<std::thread> workers_;
    std::vector<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable work_ready_, all_done_;
    size_t active_jobs_ = 0;
    bool stopping_ = false;
};

TaskDispatcher g_dispatcher;

template <typename MeanImage>
MeanImage make_mean_image(float *storage, size_t capacity, uint32_t width, uint32_t height)
{
    if constexpr (MeanImage::channel_count == 3)
        return {storage, storage + capacity, storage + capacity * 2, width, height};
    else
        return {storage, storage + capacity, storage + capacity * 2, storage + capacity * 3, width, height};
}

template <bool Simd, typename MeanImage> void downsample_means(const MeanImage &source, const MeanImage &destination)
{
    auto downsample = Simd ? downsample_simd : downsample_scalar;
    auto rows = [&](uint32_t begin, uint32_t end) {
        for (uint32_t channel = 0; channel < MeanImage::channel_count; ++channel)
            downsample(source.channel(channel), source.width, source.height, destination.channel(channel),
                       destination.width, begin, end);
    };
    g_dispatcher.parallel_rows(destination.height,
                               size_t(destination.width) * destination.height * MeanImage::channel_count, rows);
}

struct Bc1Codec
{
    using Block = Block64;
    using Color = Float3;
    using MeanImage = MeanImage3;
    static Block child(const Block &a, const Block &b, const Block &c, const Block &d, bool srgb, Color means[4],
                       uint32_t w, uint32_t h)
    {
        return bc1::generate_child_block_scalar(a, b, c, d, srgb, means, w, h);
    }
    static Block from_means(const MeanImage &m, uint32_t x, uint32_t y, bool srgb, uint32_t w, uint32_t h)
    {
        return bc1::generate_child_block_from_means_scalar(m, x, y, srgb, w, h);
    }
    static void children_x4(const Block *r0, const Block *r1, Block *out, const MeanImage &m, uint32_t x, uint32_t y0,
                            uint32_t y1, bool srgb, uint32_t w, uint32_t h)
    {
        bc1::generate_child_blocks_x4(r0, r1, out, srgb, m, x, y0, y1, w, h);
    }
    static void from_means_x4(const MeanImage &m, uint32_t x, uint32_t y, uint32_t lanes, Block *out, bool srgb,
                              uint32_t w, uint32_t h)
    {
        bc1::generate_child_blocks_from_means_x4(m, x, y, lanes, out, srgb, w, h);
    }
};

struct Bc6hCodec
{
    using Block = bc6h::Block;
    using Color = Float3;
    using MeanImage = MeanImage3;
    static Block child(const Block &a, const Block &b, const Block &c, const Block &d, bool, Color means[4], uint32_t w,
                       uint32_t h)
    {
        return bc6h::generate_child_block_scalar(a, b, c, d, means, w, h);
    }
    static Block from_means(const MeanImage &m, uint32_t x, uint32_t y, bool, uint32_t w, uint32_t h)
    {
        return bc6h::generate_child_block_from_means_scalar(m, x, y, w, h);
    }
    static void children_x4(const Block *r0, const Block *r1, Block *out, const MeanImage &m, uint32_t x, uint32_t y0,
                            uint32_t y1, bool, uint32_t w, uint32_t h)
    {
        bc6h::generate_child_blocks_x4(r0, r1, out, m, x, y0, y1, w, h);
    }
    static void from_means_x4(const MeanImage &m, uint32_t x, uint32_t y, uint32_t lanes, Block *out, bool, uint32_t w,
                              uint32_t h)
    {
        bc6h::generate_child_blocks_from_means_x4(m, x, y, lanes, out, w, h);
    }
};

struct Bc7Codec
{
    using Block = bc7::Block;
    using Color = Float4;
    using MeanImage = MeanImage4;
    static Block child(const Block &a, const Block &b, const Block &c, const Block &d, bool srgb, Color means[4],
                       uint32_t w, uint32_t h)
    {
        return bc7::generate_child(a, b, c, d, srgb, means, w, h);
    }
    static Block from_means(const MeanImage &m, uint32_t x, uint32_t y, bool srgb, uint32_t w, uint32_t h)
    {
        return bc7::generate_from_means(m, x, y, srgb, w, h);
    }
    static void children_x4(const Block *r0, const Block *r1, Block *out, const MeanImage &m, uint32_t x, uint32_t y0,
                            uint32_t y1, bool srgb, uint32_t w, uint32_t h)
    {
        bc7::generate_child_x4(r0, r1, out, m, x, y0, y1, srgb, w, h);
    }
    static void from_means_x4(const MeanImage &m, uint32_t x, uint32_t y, uint32_t lanes, Block *out, bool srgb,
                              uint32_t w, uint32_t h)
    {
        bc7::generate_from_means_x4(m, x, y, lanes, out, srgb, w, h);
    }
};

// Codec adapters keep format-specific calls out of the shared scheduling loop.
template <typename Codec, bool Simd>
void process_rows(const Image *image, const MipLevel &previous, const MipLevel &current, uint32_t level,
                  typename Codec::MeanImage &means, uint32_t begin, uint32_t end)
{
    using Block = typename Codec::Block;
    const Block *source = reinterpret_cast<const Block *>(image->data + previous.byte_offset);
    Block *destination = reinterpret_cast<Block *>(image->data + current.byte_offset);
    for (uint32_t by = begin; by < end; ++by)
    {
        Block *output = destination + size_t(by) * current.block_count_x;
        if (level >= 2)
        {
            if constexpr (Simd)
                for (uint32_t bx = 0; bx < current.block_count_x; bx += 4)
                    Codec::from_means_x4(means, bx, by, std::min(4u, current.block_count_x - bx), output + bx,
                                         image->is_srgb, current.width, current.height);
            else
                for (uint32_t bx = 0; bx < current.block_count_x; ++bx)
                    output[bx] = Codec::from_means(means, bx, by, image->is_srgb, current.width, current.height);
            continue;
        }

        const uint32_t y0 = by * 2, y1 = std::min(y0 + 1, previous.block_count_y - 1);
        const Block *row0 = source + size_t(y0) * previous.block_count_x;
        const Block *row1 = source + size_t(y1) * previous.block_count_x;
        uint32_t bx = 0;
        if constexpr (Simd)
            for (; bx + 3 < current.block_count_x && (bx + 3) * 2 + 1 < previous.block_count_x; bx += 4)
                Codec::children_x4(row0 + bx * 2, row1 + bx * 2, output + bx, means, bx * 2, y0, y1, image->is_srgb,
                                   current.width, current.height);
        for (; bx < current.block_count_x; ++bx)
        {
            const uint32_t x0 = bx * 2, x1 = std::min(x0 + 1, previous.block_count_x - 1);
            typename Codec::Color parent_means[4];
            output[bx] = Codec::child(row0[x0], row0[x1], row1[x0], row1[x1], image->is_srgb, parent_means,
                                      current.width, current.height);
            means.set(x0, y0, parent_means[0]);
            means.set(x1, y0, parent_means[1]);
            means.set(x0, y1, parent_means[2]);
            means.set(x1, y1, parent_means[3]);
        }
    }
}

template <typename Codec, bool Simd> bool generate_cpu(Image *image)
{
    using MeanImage = typename Codec::MeanImage;
    const uint32_t bw = image->mips[0].block_count_x, bh = image->mips[0].block_count_y;
    const uint32_t sw = (bw + 1) / 2, sh = (bh + 1) / 2;
    const size_t base_count = size_t(bw) * bh, scratch_count = size_t(sw) * sh;
    std::unique_ptr<float[]> base(new (std::nothrow) float[base_count * MeanImage::channel_count]);
    std::unique_ptr<float[]> scratch_data(new (std::nothrow) float[scratch_count * MeanImage::channel_count]);
    if (!base || !scratch_data)
        return false;
    MeanImage means = make_mean_image<MeanImage>(base.get(), base_count, bw, bh);
    MeanImage scratch = make_mean_image<MeanImage>(scratch_data.get(), scratch_count, sw, sh);
    for (uint32_t level = 1; level < image->mip_count; ++level)
    {
        const MipLevel previous = image->mips[level - 1], current = image->mips[level];
        if (level >= 3)
        {
            scratch.width = (means.width + 1) / 2;
            scratch.height = (means.height + 1) / 2;
            downsample_means<Simd>(means, scratch);
            std::swap(means, scratch);
        }
        auto rows = [image, previous, current, level, &means](uint32_t begin, uint32_t end) {
            process_rows<Codec, Simd>(image, previous, current, level, means, begin, end);
        };
        g_dispatcher.parallel_rows(current.block_count_y, size_t(current.block_count_x) * current.block_count_y, rows);
    }
    return true;
}

template <bool Simd> bool dispatch_cpu(Image *image)
{
    switch (image->format)
    {
    case Format::BC1:
        return generate_cpu<Bc1Codec, Simd>(image);
    case Format::BC6H_UF16:
        return generate_cpu<Bc6hCodec, Simd>(image);
    case Format::BC7:
        return generate_cpu<Bc7Codec, Simd>(image);
    default:
        return false;
    }
}
bool generate_mipmaps(Image *image, const Options &options)
{
    if (!image || !image->data || image->format == Format::Unknown)
        return false;
    switch (options.backend)
    {
    case Backend::CPU:
        return dispatch_cpu<false>(image);
    case Backend::CPU_SIMD:
        return dispatch_cpu<true>(image);
    case Backend::CUDA:
        return generate_mipmaps_cuda(image);
    }
    return false;
}

bool prepare_mipmap_backend(const Image *image, Backend backend)
{
    return backend != Backend::CUDA || prepare_mipmaps_cuda(image);
}
