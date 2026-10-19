#pragma once

#include <cstddef>
#include <cstdint>

// Describe one mip level inside the image buffer.
struct MipLevel
{
    uint32_t width = 0;
    uint32_t height = 0;

    // Count 4x4 compressed blocks, including partial edge blocks.
    uint32_t block_count_x = 0;
    uint32_t block_count_y = 0;

    // Locate this mip level in the image buffer
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

enum class Format
{
    Unknown = 0,
    BC1,
    BC6H_UF16,
    BC7,
    BC2,
    BC3,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
};

constexpr size_t block_size(Format format)
{
    switch (format)
    {
    case Format::BC1:
    case Format::BC4_UNORM:
    case Format::BC4_SNORM:
        return 8;

    case Format::BC2:
    case Format::BC3:
    case Format::BC5_UNORM:
    case Format::BC5_SNORM:
    case Format::BC6H_UF16:
    case Format::BC7:
        return 16;

    default:
        return 0;
    }
}

constexpr uint32_t MAX_MIP_LEVELS = 16;

struct Image
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mip_count = 0;

    Format format = Format::Unknown;
    bool is_srgb = false;

    MipLevel mips[MAX_MIP_LEVELS]{};

    // Own the buffer shared by all mip levels
    uint8_t *data = nullptr;
    size_t data_size = 0;
};

// Allocate space for the mip chain while preserving mip 0.
bool prepare_mip_chain(Image &image);