// cl /std:c++17 /EHsc /O2 /fp:precise tests\compare_mips.cpp src\dds.cpp /Fe:compare_mips.exe
// compare_mips.exe before.dds after.dds [other_backend.dds]
#include "../src/bc1.h"
#include "../src/dds.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

using Pixel = std::array<double, 3>;

static double linear(double code)
{
    return code <= .04045 ? code / 12.92 : std::pow((code + .055) / 1.055, 2.4);
}

static double code(double value)
{
    return std::clamp(value <= .0031308 ? 12.92 * value : 1.055 * std::pow(value, 1.0 / 2.4) - .055, 0.0, 1.0);
}

static std::vector<Pixel> decode(const Image &image, unsigned level)
{
    const auto &mip = image.mips[level];
    auto blocks = reinterpret_cast<const Block64 *>(image.get_mip_data(level));
    std::vector<Pixel> result(size_t(mip.width) * mip.height);
    for (unsigned by = 0; by < mip.block_count_y; ++by)
        for (unsigned bx = 0; bx < mip.block_count_x; ++bx)
        {
            const auto &block = blocks[size_t(by) * mip.block_count_x + bx];
            Pixel palette[4];
            for (unsigned i = 0; i < 4; ++i)
            {
                auto rgb = bc1::palette_color_rgb8<true>(block.c0, block.c1, i);
                palette[i] = {rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0};
            }
            for (unsigned y = 0; y < 4 && by * 4 + y < mip.height; ++y)
                for (unsigned x = 0; x < 4 && bx * 4 + x < mip.width; ++x)
                    result[size_t(by * 4 + y) * mip.width + bx * 4 + x] = palette[(block.indices >> (2 * (y * 4 + x))) & 3];
        }
    return result;
}

static size_t differing_bytes(const char *a, const char *b)
{
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) throw std::runtime_error("Cannot open comparison files");
    std::vector<char> va{std::istreambuf_iterator<char>(fa), {}}, vb{std::istreambuf_iterator<char>(fb), {}};
    size_t count = std::max(va.size(), vb.size()) - std::min(va.size(), vb.size());
    for (size_t i = 0; i < std::min(va.size(), vb.size()); ++i) count += va[i] != vb[i];
    return count;
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4) return 2;
    Image before, after;
    if (!load_dds(argv[1], &before) || !load_dds(argv[2], &after)) return 2;
    if (before.format != Format::BC1 || after.format != Format::BC1 || before.is_srgb != after.is_srgb
        || before.width != after.width || before.height != after.height || before.mip_count != after.mip_count
        || before.mips[0].byte_size != after.mips[0].byte_size
        || std::memcmp(before.data, after.data, before.mips[0].byte_size) != 0)
        throw std::runtime_error("Expected matching BC1 formats, dimensions and identical mip-0");
    bool failed = false;
    if (!before.is_srgb)
    {
        size_t differences = differing_bytes(argv[1], argv[2]);
        std::cout << "T2 differing file bytes: " << differences << '\n';
        failed = differences != 0;
    }
    else
    {
        auto base = decode(before, 0);
        for (auto &pixel : base) for (double &channel : pixel) channel = linear(channel);
        std::cout << "| Level | Code SSE before | Code SSE after | Linear SSE before | Linear SSE after |\n"
                     "|---:|---:|---:|---:|---:|\n" << std::setprecision(12);
        for (unsigned level = 1; level < before.mip_count; ++level)
        {
            auto old_pixels = decode(before, level), new_pixels = decode(after, level);
            unsigned scale = 1u << level;
            const auto &mip = before.mips[level];
            double old_code = 0, new_code = 0, old_linear = 0, new_linear = 0;
            for (unsigned y = 0; y < mip.height; ++y)
                for (unsigned x = 0; x < mip.width; ++x)
                {
                    // Direct 2^level box from mip-0, accumulated in float64.
                    Pixel target{};
                    unsigned end_y = std::min((y + 1) * scale, before.height);
                    unsigned end_x = std::min((x + 1) * scale, before.width);
                    size_t count = size_t(end_y - y * scale) * (end_x - x * scale);
                    for (unsigned sy = y * scale; sy < end_y; ++sy)
                        for (unsigned sx = x * scale; sx < end_x; ++sx)
                            for (int c = 0; c < 3; ++c) target[c] += base[size_t(sy) * before.width + sx][c];
                    size_t i = size_t(y) * mip.width + x;
                    for (int c = 0; c < 3; ++c)
                    {
                        target[c] /= double(count);
                        double old_delta = old_pixels[i][c] - code(target[c]);
                        double new_delta = new_pixels[i][c] - code(target[c]);
                        old_code += old_delta * old_delta;
                        new_code += new_delta * new_delta;
                        old_delta = linear(old_pixels[i][c]) - target[c];
                        new_delta = linear(new_pixels[i][c]) - target[c];
                        old_linear += old_delta * old_delta;
                        new_linear += new_delta * new_delta;
                    }
                }
            std::cout << "| " << level << " | " << old_code << " | " << new_code << " | "
                      << old_linear << " | " << new_linear << " |\n";
        }
    }
    if (argc == 4) std::cout << "T4 differing file bytes: " << differing_bytes(argv[2], argv[3]) << '\n';
    free_image(&before);
    free_image(&after);
    return failed ? 1 : 0;
}
