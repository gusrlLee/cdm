// nvcc -std=c++17 -O3 --use_fast_math tests/cuda_srgb.cu -o cuda_srgb_test.exe
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/gpu.cu"
#include <cassert>
#include <cstdio>
#include <vector>

static __global__ void check_moments(const Color *input, SymMat3 *output)
{
    unsigned lane = threadIdx.x & 15u;
    unsigned block = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    Color sample = input[block * 16 + lane];
    unsigned mask = half_warp_mask();
    Color parent{sum4(sample.r, mask) * .25f, sum4(sample.g, mask) * .25f, sum4(sample.b, mask) * .25f};
    Color mean;
    SymMat3 covariance = compute_half_warp_moments<true>(sample, parent, mean);
    if (lane == 0) output[block] = covariance;
}

static __global__ void check_encoder(Block64 *output, float *error)
{
    unsigned i = threadIdx.x;
    unsigned value = (i / 4) * 85;
    float linear = g_srgb_to_linear[value];
    Color sample{linear, linear, linear};
    unsigned mask = half_warp_mask();
    Color mean{sum4(sample.r, mask) * .25f, sum4(sample.g, mask) * .25f, sum4(sample.b, mask) * .25f};
    Block64 block{};
    encode_block_half_warp<true>(sample, mean, i, &block);
    block.c0 = (uint16_t)__shfl_sync(mask, unsigned(block.c0), 0, 16);
    block.c1 = (uint16_t)__shfl_sync(mask, unsigned(block.c1), 0, 16);
    block.indices = __shfl_sync(mask, block.indices, 0, 16);
    // Compare exact decoded bytes first, so normalization rounding cannot hide SSE=0.
    bc1::Rgb8 decoded = bc1::palette_color_rgb8<true>(block.c0, block.c1, (block.indices >> (2 * bc1::texel_index(i))) & 3);
    Color delta{float(int(value) - int(decoded.r)), float(int(value) - int(decoded.g)), float(int(value) - int(decoded.b))};
    float sse = sum16(length_sq(delta), mask) / (255.0f * 255.0f);
    if (i == 0) { *output = block; *error = sse; }
}

int main()
{
    assert(initialize_tables());
    DeviceBuffer<Block64> block;
    DeviceBuffer<float> error;
    assert(block.allocate(1) && error.allocate(1));
    check_encoder<<<1, 16>>>(block.get(), error.get());
    assert(cudaGetLastError() == cudaSuccess);
    Block64 result;
    float sse;
    assert(cudaMemcpy(&result, block.get(), sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&sse, error.get(), sizeof(sse), cudaMemcpyDeviceToHost) == cudaSuccess);
    std::printf("T1 CUDA: %04x %04x, code SSE %.9g\n", result.c0, result.c1, sse);
    std::fflush(stdout);
    assert(result.c0 == 0xffff && result.c1 == 0 && sse == 0);

    constexpr unsigned count = 10000;
    std::vector<Color> samples(count * 16);
    std::vector<SymMat3> actual(count);
    uint32_t state = 1;
    auto next = [&]() { state = state * 1664525u + 1013904223u; return float(state >> 8) / 16777215.0f; };
    for (auto &sample : samples) sample = {next(), next(), next()};
    DeviceBuffer<Color> input;
    DeviceBuffer<SymMat3> output;
    assert(input.allocate(samples.size()) && output.allocate(count));
    assert(cudaMemcpy(input.get(), samples.data(), samples.size() * sizeof(Color), cudaMemcpyHostToDevice) == cudaSuccess);
    check_moments<<<count / 16, 256>>>(input.get(), output.get());
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaMemcpy(actual.data(), output.get(), count * sizeof(SymMat3), cudaMemcpyDeviceToHost) == cudaSuccess);
    float maximum = 0;
    for (unsigned block_index = 0; block_index < count; ++block_index)
    {
        Color code[16], mean;
        for (unsigned i = 0; i < 16; ++i)
        {
            Color s = samples[block_index * 16 + i];
            code[i] = {linear_to_srgb_code(s.r), linear_to_srgb_code(s.g), linear_to_srgb_code(s.b)};
        }
        SymMat3 expected;
        bc1::compute_child_moments(code, mean, expected);
        SymMat3 got = actual[block_index];
        float differences[] = {got.rr - expected.rr, got.gg - expected.gg, got.bb - expected.bb,
                               got.rg - expected.rg, got.rb - expected.rb, got.gb - expected.gb};
        for (float difference : differences) maximum = std::max(maximum, std::abs(difference));
    }
    std::printf("T5: 10000 random blocks, 6 covariance components, max error %.9g\n", maximum);
    assert(maximum <= 1e-5f);
}
