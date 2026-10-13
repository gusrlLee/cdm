#include "dds.h"
#include "cpu.h"

#include <iostream>
#include <cstring>
#include <chrono>

uint64_t GetTime()
{
    return std::chrono::time_point_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() ).time_since_epoch().count();
}

int main(int argc, char* argv[])
{
    // Accept an optional CPU mode; default to scalar.
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input.dds> <output.dds> [scalar|simd]" << std::endl;
        return 1;
    }

    // Give each command-line path a clear name
    const char* input_path = argv[1];
    const char* output_path = argv[2];
    const char* mode = argc == 4 ? argv[3] : "scalar";

    if (std::strcmp(mode, "scalar") != 0 && std::strcmp(mode, "simd") != 0)
    {
        std::cerr << "Unknown CPU mode: " << mode << std::endl;
        return 1;
    }

    const bool use_simd = std::strcmp(mode, "simd") == 0;

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

    // Generate all mip levels from the base image.
    const uint64_t start = GetTime();
    const bool generated = generate_mipmaps_cpu(image, use_simd);
    const uint64_t end = GetTime();

    if (!generated)
    {
        std::cerr << "Failed to generate mipmaps" << std::endl;
        free_image(&image);
        return 1;
    }

    const double total_generation_time = static_cast<double>(end - start) / 1000.0;
    std::cout << "Total generation time (" << mode <<  "): " << total_generation_time << " ms" << std::endl;

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