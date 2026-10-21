Set-Location -LiteralPath $PSScriptRoot

# Build current sources before measuring; stop rather than run stale executables.
cmake --build build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Release build failed." }

# BC7 inputs must be BC7_UNORM_SRGB. BC5 supports UNORM and SNORM; BC6H supports UF16.
# Each output folder contains generated DDS files and benchmark.csv; reruns overwrite them.

# CDM - scalar (7 commands)
./build/Release/cdm.exe "D:/space/input/bc1/*.dds" "D:/space/output/cdm/scalar/bc1" scalar
./build/Release/cdm.exe "D:/space/input/bc2/*.dds" "D:/space/output/cdm/scalar/bc2" scalar
./build/Release/cdm.exe "D:/space/input/bc3/*.dds" "D:/space/output/cdm/scalar/bc3" scalar
./build/Release/cdm.exe "D:/space/input/bc4/*.dds" "D:/space/output/cdm/scalar/bc4" scalar
./build/Release/cdm.exe "D:/space/input/bc5/*.dds" "D:/space/output/cdm/scalar/bc5" scalar
./build/Release/cdm.exe "D:/space/input/bc6h/*.dds" "D:/space/output/cdm/scalar/bc6h" scalar
./build/Release/cdm.exe "D:/space/input/bc7/*.dds" "D:/space/output/cdm/scalar/bc7" scalar

# CDM - SIMD (7 commands)
./build/Release/cdm.exe "D:/space/input/bc1/*.dds" "D:/space/output/cdm/simd/bc1" simd
./build/Release/cdm.exe "D:/space/input/bc2/*.dds" "D:/space/output/cdm/simd/bc2" simd
./build/Release/cdm.exe "D:/space/input/bc3/*.dds" "D:/space/output/cdm/simd/bc3" simd
./build/Release/cdm.exe "D:/space/input/bc4/*.dds" "D:/space/output/cdm/simd/bc4" simd
./build/Release/cdm.exe "D:/space/input/bc5/*.dds" "D:/space/output/cdm/simd/bc5" simd
./build/Release/cdm.exe "D:/space/input/bc6h/*.dds" "D:/space/output/cdm/simd/bc6h" simd
./build/Release/cdm.exe "D:/space/input/bc7/*.dds" "D:/space/output/cdm/simd/bc7" simd

# CDM - CUDA (7 commands)
./build/Release/cdm.exe "D:/space/input/bc1/*.dds" "D:/space/output/cdm/cuda/bc1" cuda
./build/Release/cdm.exe "D:/space/input/bc2/*.dds" "D:/space/output/cdm/cuda/bc2" cuda
./build/Release/cdm.exe "D:/space/input/bc3/*.dds" "D:/space/output/cdm/cuda/bc3" cuda
./build/Release/cdm.exe "D:/space/input/bc4/*.dds" "D:/space/output/cdm/cuda/bc4" cuda
./build/Release/cdm.exe "D:/space/input/bc5/*.dds" "D:/space/output/cdm/cuda/bc5" cuda
./build/Release/cdm.exe "D:/space/input/bc6h/*.dds" "D:/space/output/cdm/cuda/bc6h" cuda
./build/Release/cdm.exe "D:/space/input/bc7/*.dds" "D:/space/output/cdm/cuda/bc7" cuda
