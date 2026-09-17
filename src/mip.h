#pragma once

#include <cstdint>
#include <cstddef>

// Execution strategy for mip generation; independent of the block compression format.
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
    BC6H_UF16,
    BC7,
};

struct Options
{
    Backend backend = Backend::CPU_SIMD;
};

// Dimensions and byte range of one mip level within Image::data.
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

// A full mip chain as one flat allocation; mips[level] indexes into it.
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
};

bool generate_mipmaps(Image *image, const Options &options);
bool prepare_mipmap_backend(const Image *image, Backend backend);
