#include "dds.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

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

constexpr uint32_t MakeFourCC(char c0, char c1, char c2, char c3)
{
    return (uint32_t)(uint8_t)(c0) | ((uint32_t)(uint8_t)(c1) << 8) | ((uint32_t)(uint8_t)(c2) << 16) |
           ((uint32_t)(uint8_t)(c3) << 24);
}

void free_image(Image *image)
{
    if (image && image->data)
    {
        free(image->data);
        image->data = nullptr;
    }
    if (image)
    {
        image->data_size = 0;
    }
}

bool load_dds(const char *filepath, Image *out_image)
{
    if (!out_image)
        return false;
    free_image(out_image);

    FILE *fp = fopen(filepath, "rb");
    if (!fp)
        return false;

    uint32_t magic = 0;
    if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != MakeFourCC('D', 'D', 'S', ' '))
    {
        fclose(fp);
        return false;
    }

    DdsHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1 || header.size != 124)
    {
        fclose(fp);
        return false;
    }

    // This pipeline accepts only two-dimensional textures.
    if (header.caps2 & 0xFE00)
    {
        fclose(fp);
        return false;
    }

    if (header.flags & 0x800000)
    {
        fclose(fp);
        return false;
    }

    out_image->width = header.width;
    out_image->height = header.height;
    out_image->mip_count = (header.mip_map_count == 0) ? 1 : header.mip_map_count;
    if (out_image->mip_count > MAX_MIP_LEVELS)
    {
        out_image->mip_count = MAX_MIP_LEVELS;
    }
    out_image->format = Format::Unknown;
    out_image->is_srgb = false;

    if (header.ddspf.flags & 0x4)
    {
        if (header.ddspf.four_cc == MakeFourCC('D', 'X', 'T', '1'))
        {
            out_image->format = Format::BC1;
        }
        else if (header.ddspf.four_cc == MakeFourCC('D', 'X', '1', '0'))
        {
            DdsHeaderDxt10 dxt10;
            if (fread(&dxt10, sizeof(dxt10), 1, fp) != 1)
            {
                fclose(fp);
                return false;
            }

            // Texture arrays and non-2D resources need different mip layouts.
            if (dxt10.resource_dimension != 3 || dxt10.array_size > 1)
            {
                fclose(fp);
                return false;
            }

            switch (dxt10.dxgi_format)
            {
            case 71:
                out_image->format = Format::BC1;
                out_image->is_srgb = false;
                break;
            case 72:
                out_image->format = Format::BC1;
                out_image->is_srgb = true;
                break;
            case 95:
                out_image->format = Format::BC6H_UF16;
                out_image->is_srgb = false;
                break;
            case 96: // BC6H_SF16 requires signed endpoint decoding and encoding.
                fclose(fp);
                return false;
            case 98:
                out_image->format = Format::BC7;
                out_image->is_srgb = false;
                break;
            case 99:
                out_image->format = Format::BC7;
                out_image->is_srgb = true;
                break;
            default:
                fclose(fp);
                return false;
            }
        }
    }

    if (out_image->format == Format::Unknown)
    {
        fclose(fp);
        return false;
    }

    const size_t block_bytes = out_image->format == Format::BC1 ? 8 : 16;
    uint32_t curr_w = out_image->width;
    uint32_t curr_h = out_image->height;
    size_t total_bytes = 0;

    for (uint32_t i = 0; i < out_image->mip_count; ++i)
    {
        uint32_t bx = (curr_w + 3) / 4;
        uint32_t by = (curr_h + 3) / 4;
        bx = (bx == 0) ? 1 : bx;
        by = (by == 0) ? 1 : by;
        size_t mip_bytes = (size_t)bx * by * block_bytes;

        out_image->mips[i] = {curr_w, curr_h, bx, by, total_bytes, mip_bytes};
        total_bytes += mip_bytes;

        curr_w = std::max(1u, curr_w / 2);
        curr_h = std::max(1u, curr_h / 2);
    }

    out_image->data = (uint8_t *)malloc(total_bytes);
    out_image->data_size = total_bytes;

    if (!out_image->data)
    {
        fclose(fp);
        return false;
    }

    if (fread(out_image->data, 1, total_bytes, fp) != total_bytes)
    {
        free_image(out_image);
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

bool save_dds(const char *filepath, const Image *image)
{
    if (!image || !image->data)
        return false;

    FILE *fp = fopen(filepath, "wb");
    if (!fp)
        return false;

    // Every DDS file starts with the four-byte magic value.
    uint32_t magic = MakeFourCC('D', 'D', 'S', ' ');
    fwrite(&magic, sizeof(magic), 1, fp);

    // Write the legacy DDS header followed by the DX10 extension.
    DdsHeader header = {};
    header.size = 124;
    header.flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000; // CAPS, HEIGHT, WIDTH, PIXELFORMAT, MIPMAPCOUNT
    header.height = image->height;
    header.width = image->width;
    header.pitch_or_linear_size = (uint32_t)image->mips[0].byte_size;
    header.mip_map_count = image->mip_count;
    header.caps = 0x8 | 0x1000 | 0x400000; // COMPLEX, TEXTURE, MIPMAP

    header.ddspf.size = 32;
    header.ddspf.flags = 0x4; // DDPF_FOURCC
    header.ddspf.four_cc = MakeFourCC('D', 'X', '1', '0');

    fwrite(&header, sizeof(header), 1, fp);

    DdsHeaderDxt10 dxt10 = {};
    dxt10.resource_dimension = 3; // TEXTURE2D
    dxt10.array_size = 1;

    if (image->format == Format::Unknown)
    {
        fclose(fp);
        return false;
    }
    dxt10.dxgi_format = image->format == Format::BC6H_UF16
                            ? 95
                            : (image->format == Format::BC7 ? (image->is_srgb ? 99 : 98) : (image->is_srgb ? 72 : 71));

    fwrite(&dxt10, sizeof(dxt10), 1, fp);

    if (fwrite(image->data, 1, image->data_size, fp) != image->data_size)
    {
        fclose(fp);
        return false;
    }

    if (fclose(fp) != 0)
        return false;

    return true;
}
