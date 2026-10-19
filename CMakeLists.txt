cmake_minimum_required(VERSION 3.30)
project(cdm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    ${PROJECT_NAME}

    src/main.cpp
    src/dds.h
    src/dds.cpp
    src/cpu.h
    src/cpu.cpp
    src/gpu.h
    src/gpu.cu

    src/core/color.h
    src/core/mipmap.h
    src/core/mipmap.cpp
    src/core/math.h
    src/core/mean.h
    src/core/task_dispatcher.h
    src/core/buffer.h
    src/core/platform.h
    src/core/warp.cuh
    src/core/color_cuda.cuh

    src/codecs/bc1.h
    src/codecs/bc1_simd.h
    src/codecs/bc1_cuda.cuh

    src/codecs/bc4.h
    src/codecs/bc4_simd.h
    src/codecs/bc4_cuda.cuh
    src/codecs/bc4_cuda.cu

    src/codecs/bc2.h
    src/codecs/bc2_simd.h
    src/codecs/bc2_cuda.cuh

    src/codecs/bc3.h
    src/codecs/bc3_simd.h
    src/codecs/bc3_cuda.cuh

    src/codecs/bc5.h
    src/codecs/bc5_simd.h
    src/codecs/bc5_cuda.cuh

    src/codecs/bc6h.h
    src/codecs/bc6h_simd.h
    src/codecs/bc6h_cuda.cuh

    src/codecs/bc7.h
    src/codecs/bc7_simd.h
    src/codecs/bc7_cuda.cuh
    src/codecs/bc7_tables.h
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 86 # For 3060, native For 5070
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/Ob3>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/Oi>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/GL>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/arch:AVX2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/fp:fast>
    )
    target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/LTCG>)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-ffast-math>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-flto>
    )
    target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:-flto>)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--extra-device-vectorization>
)