#pragma once

#include "core/mipmap.h"

// Initialize all CPU lookup tables before benchmarking.
void initialize_tables_cpu();

// Generate mipmaps using scalar or SIMD calculations.
bool generate_mipmaps_cpu(Image &image, bool use_simd = false);