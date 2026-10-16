# Compression-Domain Texture Mipmap Generation

## Features 
* BC1
  * Support BC1_UNORM and BC1_UNORM_SRGB. 
  * Input: Support both 3-color and 4-color modes.
  * Output: 4-color mode preferred (no 3-color mode optimization).
  * Alpha channel / 1-bit transparency preservation is not supported.
  * Support Scalar, SIMD, and CUDA backends.
  * Support Compression-Domain Mipmap Generation.
* BC2
  * Not implemented yet.
* BC3
  * Not implemented yet.
* BC4
  * Support BC4_UNORM only.
  * BC4_SNORM is not implemented yet.
  * Input: Support both 8-value and 6-value interpolation modes.
  * Output: 8-value mode preferred (no adaptive interpolation mode selection).
  * Scalar, SIMD, and CUDA backends are implemented and build successfully.
  * SIMD processes four full blocks at a time, with scalar fallback for remaining and partial blocks.
  * Output correctness, edge cases, CUDA memory checks, and performance validation are pending experimental setup.
  * Support Compression-Domain Mipmap Generation.
* BC5
  * Not implemented yet.
* BC6H
  * Not implemented yet.
* BC7
  * Not implemented yet.

## Performance 
* BC1
  * Total generation time (scalar): 10.368 ms
  * Total generation time (simd): 4.524 ms
  * Total generation time (cuda): 0.357 ms
* BC2
* BC3
* BC4
  * Not measured yet; experimental setup is pending.
* BC5
* BC6H
* BC7
