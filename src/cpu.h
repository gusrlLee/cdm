#pragma once 

#include "core/mipmap.h"

// Generate mipmaps using scalar or SIMD calculations.
bool generate_mipmaps_cpu(Image& image, bool use_simd = false);