#pragma once

// Allow shared functions to run on both CPU and GPU.
#if defined(__CUDACC__)
#define CDM_INLINE __host__ __device__ inline
#else
#define CDM_INLINE inline
#endif