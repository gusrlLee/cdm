#include "dds.h"
#include "mip.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

static bool prepare_mip_chain(Image &image)
{
    uint32_t width = image.width;
    uint32_t height = image.height;
    const size_t block_bytes = (image.format == Format::BC1 || image.format == Format::BC4) ? 8 : 16;
    MipLevel levels[MAX_MIP_LEVELS];
    size_t total_bytes = 0;
    uint32_t count = 0;

    while (count < MAX_MIP_LEVELS)
    {
        const uint32_t blocks_x = std::max(1u, (width + 3) / 4);
        const uint32_t blocks_y = std::max(1u, (height + 3) / 4);
        const size_t bytes = (size_t)blocks_x * blocks_y * block_bytes;
        levels[count++] = {width, height, blocks_x, blocks_y, total_bytes, bytes};
        total_bytes += bytes;

        if (width == 1 && height == 1)
            break;
        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);
    }

    if (image.mip_count >= count && image.data)
        return true;

    uint8_t *data = static_cast<uint8_t *>(std::malloc(total_bytes));
    if (!data)
        return false;
    std::memcpy(data, image.data, image.mips[0].byte_size);
    std::free(image.data);
    image.data = data;
    image.data_size = total_bytes;
    image.mip_count = count;
    std::copy_n(levels, count, image.mips);
    return true;
}

static bool parse_backend(std::string_view value, Backend &backend)
{
    if (value == "cpu") backend = Backend::CPU;
    else if (value == "simd" || value == "cpu-simd" || value == "cpu_simd") backend = Backend::CPU_SIMD;
    else if (value == "cuda") backend = Backend::CUDA;
    else return false;
    return true;
}

static const char *backend_name(Backend backend)
{
    switch (backend)
    {
    case Backend::CPU: return "CPU";
    case Backend::CPU_SIMD: return "CPU_SIMD";
    case Backend::CUDA: return "CUDA";
    }
    return "Unknown";
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: cdm <input.dds> <output.dds> [--backend cpu|simd|cuda]\n";
        return 1;
    }

    Options options;
    for (int i = 3; i < argc; ++i)
    {
        std::string_view argument = argv[i];
        std::string_view value;
        if (argument == "--backend" && i + 1 < argc)
            value = argv[++i];
        else if (argument.rfind("--backend=", 0) == 0)
            value = argument.substr(10);
        else
        {
            std::cerr << "Unknown option: " << argument << '\n';
            return 1;
        }

        if (!parse_backend(value, options.backend))
        {
            std::cerr << "Unknown backend: " << value << '\n';
            return 1;
        }
    }

    Image image;
    if (!load_dds(argv[1], &image))
    {
        std::cerr << "Failed to load DDS: " << argv[1] << '\n';
        return 1;
    }

    std::cout << "Loaded " << image.width << 'x' << image.height
              << " Format: " << static_cast<int>(image.format) << '\n';
    if (!prepare_mip_chain(image))
    {
        std::cerr << "Failed to allocate mip chain memory.\n";
        free_image(&image);
        return 1;
    }

    if (!prepare_mipmap_backend(&image, options.backend))
    {
        std::cerr << "Failed to initialize " << backend_name(options.backend) << ".\n";
        free_image(&image);
        return 1;
    }

    std::cout << "Generating mipmaps with " << backend_name(options.backend) << "...\n";
    const auto start = std::chrono::high_resolution_clock::now();
    const bool generated = generate_mipmaps(&image, options);
    const auto end = std::chrono::high_resolution_clock::now();
    if (!generated)
    {
        std::cerr << "Failed to generate mipmaps with " << backend_name(options.backend) << ".\n";
        free_image(&image);
        return 1;
    }

    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "====================================\n"
              << "Mipmap generation time: " << elapsed.count() << " ms\n"
              << "====================================\n";

    if (!save_dds(argv[2], &image))
    {
        std::cerr << "Failed to save DDS: " << argv[2] << '\n';
        free_image(&image);
        return 1;
    }

    free_image(&image);
    return 0;
}
