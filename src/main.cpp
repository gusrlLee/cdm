#include "dds.h"
#include "cpu.h"
#include "gpu.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

enum class Backend
{
    Scalar,
    SIMD,
    CUDA
};

static bool process_image(const fs::path &input, const fs::path &output, Backend backend, double &generation_ms)
{
    Image image{};
    if (!load_dds(input.string().c_str(), &image))
    {
        std::cerr << "Failed to load DDS: " << input << std::endl;
        free_image(&image);
        return false;
    }

    switch (image.format)
    {
    case Format::BC1:
    case Format::BC2:
    case Format::BC3:
    case Format::BC4_UNORM:
    case Format::BC5_UNORM:
    case Format::BC5_SNORM:
    case Format::BC6H_UF16:
        break;
    case Format::BC7:
        if (image.is_srgb)
            break;
        std::cerr << "BC7 mip generation requires BC7_UNORM_SRGB: " << input << std::endl;
        free_image(&image);
        return false;
    default:
        std::cerr << "Unsupported mip generation format: " << input << std::endl;
        free_image(&image);
        return false;
    }

    if (!prepare_mip_chain(image))
    {
        std::cerr << "Failed to prepare mip chain: " << input << std::endl;
        free_image(&image);
        return false;
    }

    // Include backend setup, transfers, and completion in the generation call time.
    const auto start = std::chrono::steady_clock::now();
    bool generated;
    if (backend == Backend::CUDA)
    {
        float gpu_ms = 0.0f;
        generated = generate_mipmaps_cuda(image, &gpu_ms);
    }
    else
    {
        generated = generate_mipmaps_cpu(image, backend == Backend::SIMD);
    }
    generation_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    if (!generated)
    {
        std::cerr << "Failed to generate mipmaps: " << input << std::endl;
        free_image(&image);
        return false;
    }

    const bool saved = save_dds(output.string().c_str(), &image);
    free_image(&image);
    if (!saved)
        std::cerr << "Failed to save DDS: " << output << std::endl;
    return saved;
}

int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.dds|folder/*.dds> <output.dds|output_folder> [scalar|simd|cuda]" << std::endl;
        return 1;
    }

    const char *mode = argc == 4 ? argv[3] : "scalar";
    Backend backend;
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

    try
    {
        const fs::path input = argv[1];
        const fs::path output = argv[2];

        std::string pattern = input.filename().string();
        std::transform(pattern.begin(), pattern.end(), pattern.begin(), 
            [](unsigned char c) { return static_cast<char>(std::tolower(c));});
        
        const bool batch = pattern == "*.dds";
        std::vector<fs::path> files;
        if (batch)
        {
            const fs::path directory = input.has_parent_path() ? input.parent_path() : fs::path(".");
            for (const auto &entry : fs::directory_iterator(directory))
            {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                if (entry.is_regular_file() && extension == ".dds")
                    files.push_back(entry.path());
            }

            std::sort(files.begin(), files.end());
        }
        else
        {
            if (!fs::is_regular_file(input))
            {
                std::cerr << "Input file not found: " << input << std::endl;
                return 1;
            }
            files.push_back(input);
        }
        if (files.empty())
        {
            std::cerr << "No DDS files found: " << input << std::endl;
            return 1;
        }

        const fs::path output_directory = batch ? output : (output.has_parent_path() ? output.parent_path() : fs::path("."));
        fs::create_directories(output_directory);

        const fs::path csv_path = output_directory / "benchmark.csv";

        for (const fs::path &file : files)
        {
            const fs::path destination = batch ? output / file.filename() : output;
            if ((fs::exists(destination) && fs::equivalent(file, destination)) ||
                fs::absolute(destination).lexically_normal() == fs::absolute(csv_path).lexically_normal() ||
                (fs::exists(destination) && fs::exists(csv_path) && fs::equivalent(destination, csv_path)) ||
                (fs::exists(csv_path) && fs::equivalent(file, csv_path)))
            {
                std::cerr << "Input, output DDS, and CSV paths must be distinct" << std::endl;
                return 1;
            }
        }

        if (backend == Backend::CUDA)
        {
            if (!initialize_tables_cuda())
            {
                std::cerr << "Failed to initialize CUDA tables" << std::endl;
                return 1;
            }
        }
        else // cpu
        {
            initialize_tables_cpu();
        }

        std::ofstream csv(csv_path);
        if (!csv)
        {
            std::cerr << "Failed to open CSV: " << csv_path << std::endl;
            return 1;
        }
        csv << "file name,total generation time (ms)\n" << std::fixed << std::setprecision(6);

        size_t succeeded = 0, failed = 0;
        double total_ms = 0.0;

        for (const fs::path &file : files)
        {
            std::cout << "Processing: " << file.filename().string() << std::endl;
            double generation_ms = 0.0;
            if (!process_image(file, batch ? output / file.filename() : output, backend, generation_ms))
            {
                ++failed;
                continue;
            }
            ++succeeded;

            total_ms += generation_ms;
            csv << '"';

            for (char c : file.filename().string())
            {
                if (c == '"')
                    csv << '"';
                csv << c;
            }

            csv << "\"," << generation_ms << '\n';
            if (!csv)
            {
                std::cerr << "Failed to write CSV: " << csv_path << std::endl;
                return 1;
            }
        }

        csv << "Average,";
        if (succeeded != 0)
            csv << total_ms / succeeded;
        csv << '\n';
        csv.close();
        if (!csv)
        {
            std::cerr << "Failed to write CSV: " << csv_path << std::endl;
            return 1;
        }
        
        std::cout << "Completed: " << succeeded << " succeeded, " << failed << " failed" << std::endl;
        return failed == 0 ? 0 : 1;
    }
    catch (const fs::filesystem_error &error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
