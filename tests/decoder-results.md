# Decoder D midpoint quantization results

Before: the working encoder at the start of this request (code-space fitting with round(31*s)/round(63*s)), saved as build/decoder-review/before.exe. After: decoder-midpoint quantization, before the separate small-level padding change. Release build, test.dds 2048x2048.

- T1 scalar/SIMD/CUDA: FFFF/0000, code SSE = 0.
- T2 CPU/SIMD/CUDA: 0 differing bytes for the non-sRGB copy of test.dds (DXGI 71). Whole-file byte comparison, equivalent to cmp.
- T5: 10,000 random 16-sample blocks, all six covariance components; maximum absolute GPU/scalar error = 2.98023224e-08 (limit 1e-5).
- Quantizer: 100,003 values per bit depth checked against an exhaustive nearest-decoded-value reference; all midpoint ties and immediately preceding floats checked. Midpoint ties choose the higher code.

## T4: differing file bytes

| Pair | Before | After |
|---|---:|---:|
| cpu / simd | 126 | 109 |
| cpu / cuda | 633 | 618 |
| simd / cuda | 723 | 705 |

## T3: direct float64 ideal box targets

Mip-0 is decoded with bit replication and opaque integer floor interpolation (D), then converted to linear in float64. Each target is a direct 2^L box average. Code SSE compares E(target) with D/255; linear SSE compares target with L(D/255). Only valid texels are scored. No improvement assertion is imposed.

### cpu

| Level | Code SSE before | Code SSE after | Linear SSE before | Linear SSE after |
|---:|---:|---:|---:|---:|
| 1 | 272.64870415 | 270.412933044 | 77.6860664138 | 77.1836212902 |
| 2 | 92.4665941328 | 91.8422176652 | 24.7942448324 | 24.6369414297 |
| 3 | 22.510661049 | 22.2921768454 | 5.46463947348 | 5.41322789076 |
| 4 | 5.97420094441 | 5.91864139576 | 1.39532077476 | 1.38363704257 |
| 5 | 1.7083723607 | 1.67412531169 | 0.39784167721 | 0.392799894712 |
| 6 | 0.486668300879 | 0.481404912177 | 0.112805201473 | 0.111169306887 |
| 7 | 0.153097116797 | 0.152617211432 | 0.0371708591747 | 0.036914937447 |
| 8 | 0.0419392400381 | 0.0390931796924 | 0.0116602290549 | 0.0108392432895 |
| 9 | 0.00473299307755 | 0.00473299307755 | 0.00119049629219 | 0.00119049629219 |
| 10 | 0.000999641178594 | 0.000999641178594 | 0.000288340321429 | 0.000288340321429 |
| 11 | 0.000127832863998 | 0.000127832863998 | 3.57933458856e-05 | 3.57933458856e-05 |

### simd

| Level | Code SSE before | Code SSE after | Linear SSE before | Linear SSE after |
|---:|---:|---:|---:|---:|
| 1 | 272.656494927 | 270.416040371 | 77.6898155733 | 77.1854225136 |
| 2 | 92.4685317992 | 91.8394840698 | 24.7947448151 | 24.6361084348 |
| 3 | 22.5122914326 | 22.293807229 | 5.46536514531 | 5.41395356258 |
| 4 | 5.97420094441 | 5.91864139576 | 1.39532077476 | 1.38363704257 |
| 5 | 1.7083723607 | 1.67412531169 | 0.39784167721 | 0.392799894712 |
| 6 | 0.486668300879 | 0.481404912177 | 0.112805201473 | 0.111169306887 |
| 7 | 0.153097116797 | 0.152617211432 | 0.0371708591747 | 0.036914937447 |
| 8 | 0.0419392400381 | 0.0390931796924 | 0.0116602290549 | 0.0108392432895 |
| 9 | 0.00473299307755 | 0.00473299307755 | 0.00119049629219 | 0.00119049629219 |
| 10 | 0.000999641178594 | 0.000999641178594 | 0.000288340321429 | 0.000288340321429 |
| 11 | 0.000127832863998 | 0.000127832863998 | 3.57933458856e-05 | 3.57933458856e-05 |

### cuda

| Level | Code SSE before | Code SSE after | Linear SSE before | Linear SSE after |
|---:|---:|---:|---:|---:|
| 1 | 272.646464175 | 270.411129977 | 77.6914408441 | 77.1876906959 |
| 2 | 92.4654519577 | 91.841162976 | 24.7937731806 | 24.6367781664 |
| 3 | 22.5106610476 | 22.2921768441 | 5.46464344214 | 5.41323185942 |
| 4 | 5.97420094441 | 5.91864139576 | 1.39532077476 | 1.38363704257 |
| 5 | 1.7083723607 | 1.67412531169 | 0.39784167721 | 0.392799894712 |
| 6 | 0.486668300879 | 0.481404912177 | 0.112805201473 | 0.111169306887 |
| 7 | 0.153097116797 | 0.152617211432 | 0.0371708591747 | 0.036914937447 |
| 8 | 0.0419392400381 | 0.0390931796924 | 0.0116602290549 | 0.0108392432895 |
| 9 | 0.00473299307755 | 0.00473299307755 | 0.00119049629219 | 0.00119049629219 |
| 10 | 0.000999641178594 | 0.000999641178594 | 0.000288340321429 | 0.000288340321429 |
| 11 | 0.000127832863998 | 0.000127832863998 | 3.57933458856e-05 | 3.57933458856e-05 |
