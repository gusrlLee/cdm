# Storage-space sRGB encoder checks

From an MSVC/CUDA developer prompt at the repository root:

```bat
cl /nologo /std:c++17 /EHsc /O2 /arch:AVX2 /fp:fast tests\scalar_srgb.cpp /Fe:scalar_srgb_test.exe
scalar_srgb_test.exe
nvcc -std=c++17 -O3 --use_fast_math tests\cuda_srgb.cu -o cuda_srgb_test.exe
cuda_srgb_test.exe
cl /nologo /std:c++17 /EHsc /O2 /fp:precise tests\compare_mips.cpp src\dds.cpp /Fe:compare_mips.exe
```

T1 tests all three encoders using linear samples decoded from the exact code values 0, 85, 170, 255. The decoded palette is compared against those original code values; expected endpoints are FFFF/0000 and code SSE is zero. The CPU test also checks 10,000 batches for equivalence to explicit E followed by decoder-midpoint code-space encoding, including mixed flat SIMD lanes.

For T2/T3, save DDS outputs from the original build and the modified build using the same base input. Then run:

```bat
compare_mips.exe before_linear.dds after_linear.dds
compare_mips.exe before_cpu_srgb.dds after_cpu_srgb.dds after_simd_srgb.dds
compare_mips.exe before_simd_srgb.dds after_simd_srgb.dds
compare_mips.exe before_cuda_srgb.dds after_cuda_srgb.dds
```

T2 compares complete non-sRGB files byte for byte; repeat for every backend. T3 reports both code and linear SSE per generated mip. Its ideal target is a direct 2^level box average of integer-decoded mip-0 converted to linear, using double precision. Only valid texels are scored. Neither metric is required to improve with the new objective. The optional third DDS supplies T4's complete-file differing-byte count.

See [results.md](results.md) for measured build, correctness, SSE and timing results. Before measurements use the original linear encoder at Git HEAD 882dea3, not the intermediate weighted-candidate implementation.

Latest decoder-midpoint changes: see [decoder-results.md](decoder-results.md) for T1-T5, including all backend pairs. The CUDA test compares 10,000 random blocks against scalar compute_child_moments; the CPU test also checks quantization against exhaustive decoded values.
