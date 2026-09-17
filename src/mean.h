#pragma once

#include "base.h"

#include <cstddef>
#include <cstdint>

CDM_INLINE constexpr uint32_t texel_index(uint32_t sample) {
  const uint32_t parent = sample >> 2;
  const uint32_t quadrant = sample & 3u;
  const uint32_t x = ((parent & 1u) << 1) | (quadrant & 1u);
  const uint32_t y = (parent & 2u) | (quadrant >> 1);
  return y * 4 + x;
}

constexpr uint8_t texel_map[16] = {0, 1, 4,  5,  2,  3,  6,  7,
                                   8, 9, 12, 13, 10, 11, 14, 15};

CDM_INLINE uint32_t repeat_small_coordinate(uint32_t value, uint32_t valid) {
  return valid < 4 ? value % valid : value;
}

CDM_INLINE uint32_t repeat_small_sample(uint32_t sample, uint32_t width,
                                        uint32_t height) {
  const uint32_t texel = texel_index(sample);
  const uint32_t x = repeat_small_coordinate(texel & 3u, width);
  const uint32_t y = repeat_small_coordinate(texel >> 2, height);
  return ((y >> 1) * 2 + (x >> 1)) * 4 + (y & 1u) * 2 + (x & 1u);
}

template <typename T>
CDM_INLINE void repeat_small_samples(Vec3T<T> samples[16], uint32_t width,
                                     uint32_t height) {
  if (width >= 4 && height >= 4)
    return;
  Vec3T<T> original[16];
  for (uint32_t i = 0; i < 16; ++i)
    original[i] = samples[i];
  for (uint32_t i = 0; i < 16; ++i)
    samples[i] = original[repeat_small_sample(i, width, height)];
}

struct MeanImage3 {
  static constexpr uint32_t channel_count = 3;
  float *r = nullptr, *g = nullptr, *b = nullptr;
  uint32_t width = 0, height = 0;

  CDM_INLINE float *channel(uint32_t index) const {
    return index == 0 ? r : (index == 1 ? g : b);
  }

  CDM_INLINE Float3 get(uint32_t x, uint32_t y) const {
    x = x < width ? x : width - 1;
    y = y < height ? y : height - 1;
    const size_t index = size_t(y) * width + x;
    return {r[index], g[index], b[index]};
  }

  CDM_INLINE void set(uint32_t x, uint32_t y, const Float3 &value) const {
    const size_t index = size_t(y) * width + x;
    r[index] = value.r;
    g[index] = value.g;
    b[index] = value.b;
  }
};

struct MeanImage4 {
  static constexpr uint32_t channel_count = 4;
  float *r = nullptr, *g = nullptr, *b = nullptr, *a = nullptr;
  uint32_t width = 0, height = 0;

  CDM_INLINE float *channel(uint32_t index) const {
    return index == 0 ? r : (index == 1 ? g : (index == 2 ? b : a));
  }

  CDM_INLINE Float4 get(uint32_t x, uint32_t y) const {
    x = x < width ? x : width - 1;
    y = y < height ? y : height - 1;
    const size_t index = size_t(y) * width + x;
    return {r[index], g[index], b[index], a[index]};
  }

  CDM_INLINE void set(uint32_t x, uint32_t y, const Float4 &value) const {
    const size_t index = size_t(y) * width + x;
    r[index] = value.r;
    g[index] = value.g;
    b[index] = value.b;
    a[index] = value.a;
  }
};
