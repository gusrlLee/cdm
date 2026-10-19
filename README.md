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
  * Scalar, SIMD, and CUDA support BC7_UNORM_SRGB input Modes 0-7 and output Modes 1, 6, and 7.
  * Uses integer palette reconstruction, linear RGB box filtering, and linear alpha means.
  * Preserves unquantized RGBA means across the full mip chain, including NPOT and small mips.
  * Opaque blocks fit RGB with SymMat3/Float3 and compare Modes 1/6; blocks with alpha below 1 fit RGBA with SymMat4/Float4 and compare Modes 7/6.
  * Matches src2: selects one partition by PCA mask distance, then selects the mode using estimated SSE.
  * Mode 1 is excluded when any input sample is not opaque. Opaque output alpha is preserved.
  * SIMD processes four texels per vector; CUDA uses one half-warp per block.
  * BC7_UNORM generation is not implemented. Floating-point reduction order can change encoded bytes between backends.
  * Assigns selectors by PCA projection distance; only the selected mode and partition are encoded.

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
