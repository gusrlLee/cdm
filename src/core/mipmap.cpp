#include "mipmap.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

bool prepare_mip_chain(Image &image)
{
    if (!image.data || image.width == 0 || image.height == 0)
    {
        return false;
    }

    uint32_t width = image.width;
    uint32_t height = image.height;
    const size_t block_bytes = block_size(image.format);

    if (block_bytes == 0)
    {
        return false;
    }

    MipLevel levels[MAX_MIP_LEVELS];
    uint32_t count = 0;
    size_t total_bytes = 0;

    // Calculate the dimensions and byte range of each mip.
    while (count < MAX_MIP_LEVELS)
    {
        const uint32_t blocks_x = width / 4 + (width % 4 != 0);
        const uint32_t blocks_y = height / 4 + (height % 4 != 0);

        const size_t bytes = static_cast<size_t>(blocks_x) * blocks_y * block_bytes;
        levels[count++] = {width, height, blocks_x, blocks_y, total_bytes, bytes};
        total_bytes += bytes;

        if (width == 1 && height == 1)
        {
            break;
        }

        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);
    }

    // Reuse the buffer if it already contains the full mip chain
    if (image.mip_count >= count)
        return true;

    uint8_t *data = static_cast<uint8_t *>(std::malloc(total_bytes));
    if (!data)
    {
        return false;
    }

    // Preserve mip 0 before replacing the old buffer.
    std::memcpy(data, image.data, image.mips[0].byte_size);
    std::free(image.data);

    image.data = data;
    image.data_size = total_bytes;
    image.mip_count = count;
    std::copy_n(levels, count, image.mips);
    return true;
}