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
  * Support BC2_UNORM and BC2_UNORM_SRGB.
  * Input: RGB always uses the opaque 4-color palette, regardless of endpoint order.
  * Output: Reuse BC1 RGB fitting and pack explicit 4-bit alpha per texel.
  * Average sRGB RGB in linear space and alpha in linear UNORM space.
  * Support Scalar, SIMD, and CUDA backends.
  * SIMD encodes four full blocks at a time, with scalar fallback for remaining and partial blocks.
  * Preserve unquantized RGBA means for subsequent mip levels.
  * Support Compression-Domain Mipmap Generation, including NPOT and small mips.
* BC3
  * Support BC3_UNORM and BC3_UNORM_SRGB.
  * Input: RGB always uses the opaque 4-color palette; alpha supports both 6-value and 8-value modes.
  * Reconstruct alpha using integer interpolation, separately from BC4 normalized interpolation.
  * Output: Reuse BC1 RGB fitting and fit min/max alpha endpoints without adaptive alpha-mode search.
  * Average sRGB RGB in linear space and alpha in linear UNORM space.
  * Support Scalar, SIMD, and CUDA backends.
  * SIMD encodes four full blocks at a time, with scalar fallback for remaining and partial blocks.
  * Preserve unquantized RGBA means for subsequent mip levels.
  * Support Compression-Domain Mipmap Generation, including NPOT and small mips.
* BC4
  * Support BC4_UNORM only.
  * BC4_SNORM is not implemented yet.
  * Input: Support both 8-value and 6-value interpolation modes.
  * Output: 8-value mode preferred (no adaptive interpolation mode selection).
  * Scalar, SIMD, and CUDA backends are implemented and build successfully.
  * SIMD processes four full blocks at a time, with scalar fallback for remaining and partial blocks.
  * Support Compression-Domain Mipmap Generation.
* BC5
  * Support BC5_UNORM and BC5_SNORM.
  * Process red and green as two independent BC4 channels.
  * Input: Support both 6-value and 8-value palettes for each channel.
  * SNORM supports signed interpolation and treats endpoint -128 as -127.
  * Output: Fit min/max endpoints independently for each channel without adaptive interpolation-mode search.
  * Average channels in their linear UNORM or SNORM domain.
  * Support Scalar, SIMD, and CUDA backends.
  * SIMD encodes four full blocks at a time, with scalar fallback for remaining and partial blocks.
  * Preserve unquantized red/green means for subsequent mip levels.
  * Normal-map Z reconstruction and normal renormalization are not performed.
  * Support Compression-Domain Mipmap Generation, including NPOT and small mips.
* BC6H
  * Support BC6H_UF16 only; BC6H_SF16 is not implemented.
  * Input: Support all 14 valid unsigned BC6H modes.
  * Output: Mode 11 only, with 10-bit RGB endpoints and 4-bit selectors.
  * Fit endpoints using RGB covariance and PCA, then assign selectors by projected palette distance.
  * Quantize endpoints against reconstructed half-float values and correct the anchor selector before packing.
  * Average and fit RGB in HDR linear space without clamping to the LDR range.
  * Map negative and non-finite fitting samples to zero and clamp finite overflow to 65504.
  * Support Scalar, SIMD, and CUDA backends.
  * SIMD encodes four full blocks at a time and normalizes its approximate PCA axis again before HDR fitting.
  * Partial and remaining SIMD blocks use scalar fallback.
  * Preserve unquantized HDR RGB means for subsequent mip levels.
  * Support Compression-Domain Mipmap Generation, including NPOT and small mips.
  * Multi-mode output search is not implemented.
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