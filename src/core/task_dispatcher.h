#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

class TaskDispatcher
{
public:
    TaskDispatcher()
    {
        const size_t count = std::max(1u, std::thread::hardware_concurrency());
        workers_.reserve(count - 1);
        for (size_t i = 1; i < count; ++i)
            workers_.emplace_back([this]
                                  { worker_loop(); });
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
            enqueue([=]
                    { function(begin, end); });
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
            std::function<void()> task = std::move(queue_.back());
            queue_.pop_back();
            finish(lock, std::move(task));
        }
        all_done_.wait(lock, [this]
                       { return queue_.empty() && active_jobs_ == 0; });
    }

    void worker_loop()
    {
        for (;;)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_ready_.wait(lock, [this]
                             { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty())
                return;
            std::function<void()> task = std::move(queue_.back());
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