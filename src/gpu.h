#pragma once

#include "core/mipmap.h"

// Initialize and upload all CUDA lookup tables before benchmarking.
bool initialize_tables_cuda();

// Generate the mip chain on the GPU and copy the result back.
// Return GPU event time when requested instead of printing it.
bool generate_mipmaps_cuda(Image &image, float *generation_time_ms = nullptr);
