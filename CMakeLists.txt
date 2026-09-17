cmake_minimum_required(VERSION 3.30)
project(cdm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    ${PROJECT_NAME}
    src/dds.h
    src/mip.h
    src/base.h
    src/mean.h
    src/bc1.h
    src/bc6h.h
    src/bc7.h
    src/main.cpp
    src/dds.cpp
    src/cpu.cpp
    src/gpu.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES native
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

include(CTest)
if(BUILD_TESTING)
    add_executable(cdm_pca4_test tests/pca4.cpp)
    target_include_directories(cdm_pca4_test PRIVATE src)
    if(NOT MSVC)
        target_compile_options(cdm_pca4_test PRIVATE -msse4.1)
    endif()
    add_test(NAME pca4 COMMAND cdm_pca4_test)
endif()
