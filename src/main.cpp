#include "dds.h"
#include "cpu.h"

#include <iostream>

int main(int argc, char* argv[])
{
    // Expect an input DDS path and an output DDS path.
    if (argc != 3)
    {
        std::cerr << "Usage : " << argv[0] << " <input.dds> <output.dds>" << std::endl;
        return 1;
    }

    // Give each command-line path a clear name
    const char* input_path = argv[1];
    const char* output_path = argv[2];

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
    if (!generate_mipmaps_cpu(image))
    {
        std::cerr << "Failed to generate mipmaps" << std::endl;
        return 1;
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