#include "dds.h"
#include <fstream>
#include <algorithm>
#include <cstring>

#pragma pack(push, 1)
struct DdsPixelFormat
{
    uint32_t size;
    uint32_t flags;
    uint32_t four_cc;
    uint32_t rgb_bit_count;
    uint32_t r_bit_mask;
    uint32_t g_bit_mask;
    uint32_t b_bit_mask;
    uint32_t a_bit_mask;
};

struct DdsHeader
{
    uint32_t size;
    uint32_t flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitch_or_linear_size;
    uint32_t depth;
    uint32_t mip_map_count;
    uint32_t reserved1[11];
    DdsPixelFormat ddspf;
    uint32_t caps;
    uint32_t caps2;
    uint32_t caps3;
    uint32_t caps4;
    uint32_t reserved2;
};

struct DdsHeaderDxt10
{
    uint32_t dxgi_format;
    uint32_t resource_dimension;
    uint32_t misc_flag;
    uint32_t array_size;
    uint32_t misc_flags2;
};
#pragma pack(pop)

constexpr uint32_t MakeFourCC(char ch0, char ch1, char ch2, char ch3)
{
    return (uint32_t)(uint8_t)(ch0) |
           ((uint32_t)(uint8_t)(ch1) << 8) |
           ((uint32_t)(uint8_t)(ch2) << 16) |
           ((uint32_t)(uint8_t)(ch3) << 24);
}

static size_t GetBytesPerBlock(TextureFormat format)
{
    switch (format)
    {
    case TextureFormat::BC1:
    case TextureFormat::BC4:
        return 8;
    case TextureFormat::BC5:
    case TextureFormat::BC6H:
    case TextureFormat::BC7:
        return 16;
    default:
        return 0;
    }
}

bool LoadDDS(const std::string &filepath, DdsImage &out_image)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    uint32_t magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (magic != MakeFourCC('D', 'D', 'S', ' '))
    {
        return false;
    }

    DdsHeader header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (header.size != 124 || header.ddspf.size != 32)
    {
        return false;
    }

    out_image.width = header.width;
    out_image.height = header.height;
    out_image.mip_count = (header.mip_map_count == 0) ? 1 : header.mip_map_count;
    out_image.format = TextureFormat::Unknown;
    out_image.is_srgb = false;

    const uint32_t DDPF_FOURCC = 0x4;
    if (header.ddspf.flags & DDPF_FOURCC)
    {
        if (header.ddspf.four_cc == MakeFourCC('D', 'X', 'T', '1'))
        {
            out_image.format = TextureFormat::BC1;
        }
        else if (header.ddspf.four_cc == MakeFourCC('D', 'X', '1', '0'))
        {
            DdsHeaderDxt10 dxt10_header;
            file.read(reinterpret_cast<char *>(&dxt10_header), sizeof(dxt10_header));

            switch (dxt10_header.dxgi_format)
            {
            case 71: // DXGI_FORMAT_BC1_UNORM
                out_image.format = TextureFormat::BC1;
                out_image.is_srgb = false;
                break;
            case 72: // DXGI_FORMAT_BC1_UNORM_SRGB
                out_image.format = TextureFormat::BC1;
                out_image.is_srgb = true;
                break;
            case 80: // DXGI_FORMAT_BC4_UNORM
                out_image.format = TextureFormat::BC4;
                break;
            case 83: // DXGI_FORMAT_BC5_UNORM
                out_image.format = TextureFormat::BC5;
                break;
            case 95: // DXGI_FORMAT_BC6H_UF16
            case 96: // DXGI_FORMAT_BC6H_SF16
                out_image.format = TextureFormat::BC6H;
                break;
            case 98: // DXGI_FORMAT_BC7_UNORM
                out_image.format = TextureFormat::BC7;
                out_image.is_srgb = false;
                break;
            case 99: // DXGI_FORMAT_BC7_UNORM_SRGB
                out_image.format = TextureFormat::BC7;
                out_image.is_srgb = true;
                break;
            default:
                return false;
            }
        }
    }

    if (out_image.format == TextureFormat::Unknown)
    {
        return false;
    }

    // 4. 각 밉맵 레벨의 크기 및 오프셋 계산
    size_t block_bytes = GetBytesPerBlock(out_image.format);
    out_image.mips.clear();

    uint32_t curr_w = out_image.width;
    uint32_t curr_h = out_image.height;
    size_t total_data_bytes = 0;

    for (uint32_t i = 0; i < out_image.mip_count; ++i)
    {
        uint32_t bx = std::max(1u, (curr_w + 3) / 4);
        uint32_t by = std::max(1u, (curr_h + 3) / 4);
        size_t mip_bytes = (size_t)bx * by * block_bytes;

        DdsMipLevel mip;
        mip.width = curr_w;
        mip.height = curr_h;
        mip.block_count_x = bx;
        mip.block_count_y = by;
        mip.byte_offset = total_data_bytes;
        mip.byte_size = mip_bytes;
        out_image.mips.push_back(mip);

        total_data_bytes += mip_bytes;

        curr_w = std::max(1u, curr_w / 2);
        curr_h = std::max(1u, curr_h / 2);
    }

    out_image.data.resize(total_data_bytes);
    file.read(reinterpret_cast<char *>(out_image.data.data()), total_data_bytes);

    return !file.bad();
}