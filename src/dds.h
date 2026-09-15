#pragma once

#include <cstdint>
#include <vector>
#include <string>

enum class TextureFormat {
    Unknown,
    BC1,
    BC4,
    BC5,
    BC6H,
    BC7
};

struct DdsMipLevel {
    uint32_t width;
    uint32_t height;
    uint32_t block_count_x;
    uint32_t block_count_y;
    size_t byte_offset; // image.data 내에서의 시작 위치
    size_t byte_size;
};

struct DdsImage {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mip_count = 0;
    TextureFormat format = TextureFormat::Unknown;
    bool is_srgb = false;
    
    std::vector<DdsMipLevel> mips;
    std::vector<uint8_t> data;

    const uint8_t* GetMipData(uint32_t level) const {
        if (level >= mips.size()) return nullptr;
        return data.data() + mips[level].byte_offset;
    }
};

bool LoadDDS(const std::string& filepath, DdsImage& out_image);