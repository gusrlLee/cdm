// cl /nologo /std:c++17 /EHsc /O2 /arch:AVX2 tests\scalar_srgb.cpp /Fe:scalar_srgb_test.exe
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/bc1.h"
#include <cassert>
#include <cstdio>
#include <cstring>

static float code_sse(const Float3 code[16], const Block64 &block)
{
    Float3 palette[4];
    bc1::get_opaque_palette_scalar(block.c0, block.c1, false, palette);
    float error = 0;
    for (int i = 0; i < 16; ++i)
        error += length_sq(code[i] - palette[(block.indices >> (2 * bc1::texel_map[i])) & 3]);
    return error;
}

template <unsigned Bits>
static void check_quantizer()
{
    constexpr unsigned maximum = (1u << Bits) - 1;
    for (int i = -1; i <= 100001; ++i)
    {
        float value = i / 100000.0f;
        unsigned actual = bc1::quantize_code_channel<Bits>(value);
        double best = 1e30, chosen = 0;
        for (unsigned q = 0; q <= maximum; ++q)
        {
            double decoded = ((q << (8 - Bits)) | (q >> (2 * Bits - 8))) / 255.0;
            double distance = std::abs(double(value) - decoded);
            best = std::min(best, distance);
            if (q == actual) chosen = distance;
        }
        assert(chosen <= best + 1e-7);
    }
    for (unsigned q = 0; q < maximum; ++q)
    {
        unsigned a = (q << (8 - Bits)) | (q >> (2 * Bits - 8));
        unsigned b = ((q + 1) << (8 - Bits)) | ((q + 1) >> (2 * Bits - 8));
        float midpoint = float(a + b) * (1.0f / 510.0f);
        assert(bc1::quantize_code_channel<Bits>(midpoint) == q + 1);
        assert(bc1::quantize_code_channel<Bits>(std::nextafter(midpoint, 0.0f)) == q);
    }
}

int main()
{
    check_quantizer<5>();
    check_quantizer<6>();
    Float3 samples[16], code[16];
    Float3x4 batch[16];
    for (int i = 0; i < 16; ++i)
    {
        unsigned value = (i / 4) * 85;
        float linear = bc1::c_srgb_to_linear[value];
        float storage = value * (1.0f / 255.0f);
        samples[i] = {linear, linear, linear};
        code[i] = {storage, storage, storage};
        batch[i] = {v4f(linear), v4f(linear), v4f(linear)};
    }
    Block64 scalar = bc1::encode_samples_scalar(samples, true), simd[4];
    bc1::encode_samples_x4<true>(batch, simd);
    float error = code_sse(code, scalar);
    std::printf("T1 scalar: %04x %04x, code SSE %.9g\n", scalar.c0, scalar.c1, error);
    assert(scalar.c0 == 0xffff && scalar.c1 == 0 && error == 0);
    for (const Block64 &block : simd)
        assert(block.c0 == 0xffff && block.c1 == 0 && code_sse(code, block) == 0);
    assert(linear_to_srgb_code(-1) == 0 && linear_to_srgb_code(2) == 1);
    uint32_t state = 1;
    auto next = [&]() { state = state * 1664525u + 1013904223u; return float(state >> 8) / 16777215.0f; };
    for (int block = 0; block < 10000; ++block)
    {
        for (int i = 0; i < 16; ++i)
        {
            samples[i] = {next(), next(), next()};
            code[i] = {linear_to_srgb_code(samples[i].r), linear_to_srgb_code(samples[i].g), linear_to_srgb_code(samples[i].b)};
            batch[i] = {_mm_set_ps(0, .5f, samples[i].g, samples[i].r),
                        _mm_set_ps(0, .5f, samples[i].b, samples[i].g),
                        _mm_set_ps(0, .5f, samples[i].r, samples[i].b)};
        }
        // The sRGB entry must do exactly E followed by the unchanged storage-space encoder.
        Block64 expected = bc1::encode_samples_code_scalar<true>(code);
        Block64 actual = bc1::encode_samples_scalar(samples, true);
        assert(std::memcmp(&expected, &actual, sizeof(actual)) == 0);
        Float3x4 converted[16];
        for (int i = 0; i < 16; ++i)
            converted[i] = {linear_to_srgb_code(batch[i].r), linear_to_srgb_code(batch[i].g), linear_to_srgb_code(batch[i].b)};
        Block64 expected_batch[4], actual_batch[4];
        bc1::encode_samples_code_x4<true>(converted, expected_batch);
        bc1::encode_samples_x4<true>(batch, actual_batch);
        assert(std::memcmp(expected_batch, actual_batch, sizeof(actual_batch)) == 0);
    }
    std::puts("PASS: scalar/SIMD T1 and 10000 storage-space encoding checks");
}
