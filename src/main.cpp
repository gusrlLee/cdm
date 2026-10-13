#include "dds.h"
#include "cpu.h"
#include "gpu.h"

#include <iostream>
#include <cstring>
#include <chrono>

enum class Backend
{
    Scalar,
    SIMD,
    CUDA
};

uint64_t GetTime()
{
    return std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
}

int main(int argc, char *argv[])
{
    // Accept an optional processing mode; default to scalar.
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input.dds> <output.dds> [scalar|simd|cuda]" << std::endl;
        return 1;
    }

    // Give each command-line path a clear name
    const char *input_path = argv[1];
    const char *output_path = argv[2];
    const char *mode = argc == 4 ? argv[3] : "scalar";

    Backend backend = Backend::Scalar;

    if (std::strcmp(mode, "scalar") == 0)
        backend = Backend::Scalar;
    else if (std::strcmp(mode, "simd") == 0)
        backend = Backend::SIMD;
    else if (std::strcmp(mode, "cuda") == 0)
        backend = Backend::CUDA;
    else
    {
        std::cerr << "Unknown backend: " << mode << std::endl;
        return 1;
    }

    Image image{};

    // Load the compressed image from the input file.
    if (!load_dds(input_path, &image))
    {
        std::cerr << "Failed to load DDS: " << input_path << std::endl;
        free_image(&image);
        return 1;
    }

    // check dds file format
    if (image.format != Format::BC1)
    {
        std::cerr << "Only BC1 DDS files are supported" << std::endl;
        free_image(&image);
        return 1;
    }

    // Allocate storage for the full mip chain.
    if (!prepare_mip_chain(image))
    {
        std::cerr << "Failed to prepare mip chain" << std::endl;
        free_image(&image);
        return 1;
    }

    // CPU modes measure the full call; CUDA reports its GPU event time.
    const uint64_t start = GetTime();
    bool generated = false;

    switch (backend)
    {
    case Backend::Scalar:
        generated = generate_mipmaps_cpu(image, false);
        break;

    case Backend::SIMD:
        generated = generate_mipmaps_cpu(image, true);
        break;

    case Backend::CUDA:
        generated = generate_mipmaps_cuda(image);
        break;
    }

    const uint64_t end = GetTime();

    if (!generated)
    {
        std::cerr << "Failed to generate mipmaps" << std::endl;
        free_image(&image);
        return 1;
    }

    if (backend != Backend::CUDA)
    {
        const double total_generation_time = static_cast<double>(end - start) / 1000.0;
        std::cout << "Total generation time (" << mode << "): " << total_generation_time << " ms" << std::endl;
    }

    // Save the generated mip chain.
    if (!save_dds(output_path, &image))
    {
        std::cerr << "Failed to save DDS:" << output_path << std::endl;
        free_image(&image);
        return 1;
    }

    // Release the buffer before exiting
    free_image(&image);
    return 0;
}
