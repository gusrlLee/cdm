#include "dds.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

static bool prepare_mip_chain(Image &image)
{
    uint32_t width = image.width;
    uint32_t height = image.height;
    const size_t block_bytes = image.format == Format::BC1 ? 8 : 16;
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
    if (value == "cpu")
        backend = Backend::CPU;
    else if (value == "simd" || value == "cpu-simd" || value == "cpu_simd")
        backend = Backend::CPU_SIMD;
    else if (value == "cuda")
        backend = Backend::CUDA;
    else
        return false;
    return true;
}

static const char *backend_name(Backend backend)
{
    switch (backend)
    {
    case Backend::CPU:
        return "CPU";
    case Backend::CPU_SIMD:
        return "CPU_SIMD";
    case Backend::CUDA:
        return "CUDA";
    }
    return "Unknown";
}

static bool wildcard_match(std::string_view pattern, std::string_view value)
{
    size_t pattern_index = 0;
    size_t value_index = 0;
    size_t star_index = std::string_view::npos;
    size_t star_value_index = 0;

    while (value_index < value.size())
    {
        const bool same = pattern_index < pattern.size() &&
                          std::tolower(static_cast<unsigned char>(pattern[pattern_index])) ==
                              std::tolower(static_cast<unsigned char>(value[value_index]));
        if (same || (pattern_index < pattern.size() && pattern[pattern_index] == '?'))
        {
            ++pattern_index;
            ++value_index;
        }
        else if (pattern_index < pattern.size() && pattern[pattern_index] == '*')
        {
            star_index = pattern_index++;
            star_value_index = value_index;
        }
        else if (star_index != std::string_view::npos)
        {
            pattern_index = star_index + 1;
            value_index = ++star_value_index;
        }
        else
            return false;
    }

    while (pattern_index < pattern.size() && pattern[pattern_index] == '*')
        ++pattern_index;
    return pattern_index == pattern.size();
}

static std::vector<fs::path> find_input_files(const fs::path &input)
{
    std::vector<fs::path> files;
    std::error_code error;
    if (fs::is_directory(input, error))
    {
        for (const auto &entry : fs::directory_iterator(input, error))
            if (entry.is_regular_file() && wildcard_match("*.dds", entry.path().filename().string()))
                files.push_back(entry.path());
    }
    else if (input.filename().string().find_first_of("*?") != std::string::npos)
    {
        fs::path directory = input.parent_path();
        if (directory.empty())
            directory = ".";
        const std::string pattern = input.filename().string();
        for (const auto &entry : fs::directory_iterator(directory, error))
            if (entry.is_regular_file() && wildcard_match(pattern, entry.path().filename().string()))
                files.push_back(entry.path());
    }
    else if (fs::is_regular_file(input, error))
        files.push_back(input);

    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: cdm <input.dds|input_directory|pattern> <output.dds|output_directory> "
                     "[--backend cpu|simd|cuda]\n";
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

    const std::vector<fs::path> input_files = find_input_files(fs::path(argv[1]));
    if (input_files.empty())
    {
        std::cerr << "No DDS files found: " << argv[1] << '\n';
        return 1;
    }

    const fs::path input_argument(argv[1]);
    const fs::path output_argument(argv[2]);
    const bool has_wildcard = input_argument.filename().string().find_first_of("*?") != std::string::npos;
    const bool batch = input_files.size() > 1 || fs::is_directory(input_argument) || has_wildcard;
    if (batch)
    {
        std::error_code error;
        fs::create_directories(output_argument, error);
        if (error || !fs::is_directory(output_argument))
        {
            std::cerr << "Failed to create output directory: " << output_argument.string() << '\n';
            return 1;
        }
    }

    const auto process_start = std::chrono::steady_clock::now();
    double total_generation_ms = 0.0;
    size_t processed_count = 0;

    for (const fs::path &input_file : input_files)
    {
        Image image;
        if (!load_dds(input_file.string().c_str(), &image))
        {
            std::cerr << "Failed to load DDS: " << input_file.string() << '\n';
            return 1;
        }

        if (!prepare_mip_chain(image))
        {
            std::cerr << "Failed to allocate mip chain memory: " << input_file.string() << '\n';
            free_image(&image);
            return 1;
        }

        if (!prepare_mipmap_backend(&image, options.backend))
        {
            std::cerr << "Failed to initialize " << backend_name(options.backend) << ": " << input_file.string()
                      << '\n';
            free_image(&image);
            return 1;
        }

        const auto generation_start = std::chrono::steady_clock::now();
        const bool generated = generate_mipmaps(&image, options);
        const auto generation_end = std::chrono::steady_clock::now();
        if (!generated)
        {
            std::cerr << "Failed to generate mipmaps with " << backend_name(options.backend) << ": "
                      << input_file.string() << '\n';
            free_image(&image);
            return 1;
        }
        total_generation_ms +=
            std::chrono::duration<double, std::milli>(generation_end - generation_start).count();

        const fs::path output_file = batch ? output_argument / input_file.filename() : output_argument;
        if (!save_dds(output_file.string().c_str(), &image))
        {
            std::cerr << "Failed to save DDS: " << output_file.string() << '\n';
            free_image(&image);
            return 1;
        }
        free_image(&image);
        ++processed_count;
    }

    const double total_process_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - process_start).count();
    const double average_generation_ms = total_generation_ms / static_cast<double>(processed_count);

    std::cout << std::fixed << std::setprecision(3)
              << "========================================================================\n"
              << "                  BENCHMARK PURE COMPUTE SUMMARY\n"
              << "========================================================================\n"
              << " Total Processed Images: " << processed_count << '\n'
              << " Method: [Our Method - Compression Domain Mipmap Generation / " << backend_name(options.backend)
              << "]\n"
              << "   * Total Mipmap Generation Time: " << total_generation_ms << " ms\n"
              << "   * Average Time per Image:       " << average_generation_ms << " ms\n"
              << "   (Disk Read/Write I/O strictly excluded from compute time)\n"
              << "   --------------------------------------------------------------------\n"
              << " Total Process Time (including Disk Read/Write): " << total_process_ms << " ms\n"
              << "========================================================================\n";
    return 0;
}
