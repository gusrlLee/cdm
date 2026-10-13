#pragma once

#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    // Prevent two buffers from owning the same GPU memory.
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer()
    {
        if (data_)
            cudaFree(data_);
    }

    bool allocate(size_t count)
    {
        // Reuse the allocation when it is large enough.
        if (count <= capacity_)
            return true;

        T *replacement = nullptr;

        if (cudaMalloc(
                reinterpret_cast<void **>(&replacement),
                count * sizeof(T)) != cudaSuccess)
        {
            return false;
        }

        if (data_)
            cudaFree(data_);

        data_ = replacement;
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