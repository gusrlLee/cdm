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

template <bool Srgb>
static __global__ void reference_small_mean(MeanImage source, unsigned w, unsigned h, Block64 *output)
{
    unsigned i = threadIdx.x;
    unsigned local = bc1::texel_index(i);
    unsigned index = ((local / 4) % h) * source.width + ((local % 4) % w);
    Color sample{source.r[index], source.g[index], source.b[index]};
    unsigned mask = half_warp_mask();
    Color parent{sum4(sample.r, mask) * .25f, sum4(sample.g, mask) * .25f, sum4(sample.b, mask) * .25f};
    encode_block_half_warp<Srgb>(sample, parent, i, output);
}

template <bool Srgb>
static __global__ void reference_small_base(const Block64 *source, unsigned w, unsigned h, Block64 *output)
{
    unsigned i = threadIdx.x, local = bc1::texel_index(i);
    unsigned x = (local % 4) % w, y = (local / 4) % h;
    Block64 block = source[(y / 2) * 2 + x / 2];
    unsigned shift = (y % 2) * 16 + (x % 2) * 4;
    Color sample{};
    for (unsigned selector = 0; selector < 4; ++selector)
        sample += palette_color<Srgb>(block.c0, block.c1, selector)
            * (float((bc1::selector_region_counts(block.indices, selector) >> shift) & 15) * .25f);
    unsigned mask = half_warp_mask();
    Color parent{sum4(sample.r, mask) * .25f, sum4(sample.g, mask) * .25f, sum4(sample.b, mask) * .25f};
    encode_block_half_warp<Srgb>(sample, parent, i, output);
}

static void check_small_levels()
{
    float values[48];
    for (unsigned i = 0; i < 48; ++i) values[i] = ((i * 7) % 47) / 47.f;
    DeviceBuffer<float> input, next;
    DeviceBuffer<Block64> blocks;
    DeviceBuffer<Block64> base;
    Block64 parents[4] = {{0xffff, 0, 0x1b1b1b1b}, {0xf800, 0, 0x12345678},
                         {0x07e0, 0, 0x76543210}, {0x001f, 0, 0xabcdef01}};
    assert(input.allocate(48) && next.allocate(12) && blocks.allocate(2) && base.allocate(4));
    assert(cudaMemcpy(base.get(), parents, sizeof(parents), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(input.get(), values, sizeof(values), cudaMemcpyHostToDevice) == cudaSuccess);
    MeanImage source{input.get(), input.get()+16, input.get()+32, 4, 4};
    MeanImage destination{next.get(), next.get()+4, next.get()+8, 2, 2};
    for (unsigned h = 1; h <= 4; ++h)
        for (unsigned w = 1; w <= 4; ++w)
            for (bool srgb : {false, true})
            {
                if (srgb)
                {
                    generate_mean_mip<true><<<1,16>>>(source, blocks.get(), 1, 1, destination, w, h);
                    reference_small_mean<true><<<1,16>>>(source, w, h, blocks.get()+1);
                }
                else
                {
                    generate_mean_mip<false><<<1,16>>>(source, blocks.get(), 1, 1, destination, w, h);
                    reference_small_mean<false><<<1,16>>>(source, w, h, blocks.get()+1);
                }
                assert(cudaGetLastError() == cudaSuccess);
                Block64 result[2];
                assert(cudaMemcpy(result, blocks.get(), sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess);
                assert(result[0].c0 == result[1].c0 && result[0].c1 == result[1].c1 && result[0].indices == result[1].indices);
                float stored[12];
                assert(cudaMemcpy(stored, next.get(), sizeof(stored), cudaMemcpyDeviceToHost) == cudaSuccess);
                for (unsigned c = 0; c < 3; ++c)
                    for (unsigned q = 0; q < 4; ++q)
                    {
                        unsigned i = c*16 + (q/2)*8 + (q%2)*2;
                        float expected = (values[i]+values[i+1]+values[i+4]+values[i+5])*.25f;
                        assert(std::abs(stored[c*4+q]-expected) < 1e-7f);
                    }
                if (srgb)
                {
                    generate_base_mip<true><<<1,16>>>(base.get(), 2, 2, blocks.get(), 1, 1, destination, w, h);
                    reference_small_base<true><<<1,16>>>(base.get(), w, h, blocks.get()+1);
                }
                else
                {
                    generate_base_mip<false><<<1,16>>>(base.get(), 2, 2, blocks.get(), 1, 1, destination, w, h);
                    reference_small_base<false><<<1,16>>>(base.get(), w, h, blocks.get()+1);
                }
                assert(cudaGetLastError() == cudaSuccess);
                assert(cudaMemcpy(result, blocks.get(), sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess);
                assert(result[0].c0 == result[1].c0 && result[0].c1 == result[1].c1 && result[0].indices == result[1].indices);
            }
    std::puts("PASS: CUDA small levels 1..4 in both modes; linear next means unchanged");
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
    check_small_levels();
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
