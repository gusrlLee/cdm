#pragma once

#include "core/mipmap.h"

// Generate the mip chain on the GPU and copy the result back.
bool generate_mipmaps_cuda(Image &image);