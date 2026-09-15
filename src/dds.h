#pragma once

#include "mip.h"

#include <cstdint>
#include <cstddef>

bool load_dds(const char* filepath, Image* out_image);
bool save_dds(const char* filepath, const Image* image);
void free_image(Image* image);