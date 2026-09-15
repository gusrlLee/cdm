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

// ---------------------------------------------------------------------
// CPU utilities
// ---------------------------------------------------------------------

static bc1::MeanImage make_mean_image(float *storage, size_t capacity, uint32_t width, uint32_t height)
{
    return {storage, storage + capacity, storage + capacity * 2, width, height};
}

static void downsample_channel_scalar(const float *source, uint32_t source_width, uint32_t source_height,
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

static void downsample_channel_simd(const float *source, uint32_t source_width, uint32_t source_height,
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
static void downsample_means(const bc1::MeanImage &source, const bc1::MeanImage &destination)
{
    auto downsample = Simd ? downsample_channel_simd : downsample_channel_scalar;
    downsample(source.r, source.width, source.height, destination.r, destination.width, destination.height);
    downsample(source.g, source.width, source.height, destination.g, destination.width, destination.height);
    downsample(source.b, source.width, source.height, destination.b, destination.width, destination.height);
}

// ---------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------

// Persistent thread pool. Work is split into block-row ranges (not per-block),
// so each task amortizes its scheduling cost over many blocks.
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

static TaskDispatcher g_dispatcher;

// ---------------------------------------------------------------------
// Mip-level processing (dispatches into bc1.h's scalar / SIMD algorithms)
// ---------------------------------------------------------------------

// Process one row range of a mip level. Level 1 derives blocks directly from the
// previous mip's BC1 blocks (and records the mean pyramid's base level); level 2+
// reads its 16 samples from the mean pyramid instead.
template <bool Simd>
static void process_mip_rows(const Image *image, const MipLevel &previous, const MipLevel &current,
                       uint32_t level, bc1::MeanImage &means, uint32_t begin, uint32_t end)
{
    const Block64 *source = reinterpret_cast<const Block64 *>(image->data + previous.byte_offset);
    Block64 *destination = reinterpret_cast<Block64 *>(image->data + current.byte_offset);

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
}

template <bool Simd>
static bool generate_cpu(Image *image)
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

        auto process_rows = [image, previous, current, level, &means](uint32_t begin, uint32_t end)
        { process_mip_rows<Simd>(image, previous, current, level, means, begin, end); };
        g_dispatcher.parallel_rows(current.block_count_y,
            (size_t)current.block_count_x * current.block_count_y, process_rows);
    }
    return true;
}

// ---------------------------------------------------------------------
// Public CPU entry point
// ---------------------------------------------------------------------

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
