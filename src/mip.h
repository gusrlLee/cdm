#pragma once

#include <cstdint>
#include <cstddef>

enum class Backend
{
    CPU,
    CPU_SIMD,
    CUDA,
};

enum class Format
{
    Unknown = 0,
    BC1,
    BC2,
    BC3,
    BC4,
    BC5,
    BC6H,
    BC7,
};

struct Options
{
    Backend backend = Backend::CPU_SIMD;
};

struct MipLevel
{
    uint32_t width;
    uint32_t height;
    uint32_t block_count_x;
    uint32_t block_count_y;
    size_t byte_offset;
    size_t byte_size;
};

constexpr uint32_t MAX_MIP_LEVELS = 16;

struct Image
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mip_count = 0;
    Format format = Format::Unknown;

    bool is_srgb = false;
    MipLevel mips[MAX_MIP_LEVELS];
    uint8_t *data = nullptr;
    size_t data_size = 0;

    const uint8_t *get_mip_data(uint32_t level) const
    {
        return (level < mip_count && data) ? data + mips[level].byte_offset : nullptr;
    }
    uint8_t *get_mip_data(uint32_t level)
    {
        return (level < mip_count && data) ? data + mips[level].byte_offset : nullptr;
    }
};

bool generate_mipmaps(Image *image, const Options &options = {});
bool prepare_mipmap_backend(const Image *image, Backend backend);
