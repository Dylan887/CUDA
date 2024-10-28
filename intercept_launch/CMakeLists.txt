cmake_minimum_required(VERSION 3.5.0)
project(intercept_launch VERSION 0.1.0 LANGUAGES C CXX CUDA)

add_executable(intercept_launch 

    main.cu
    kernels_launch.cu 
    CUDA_init_destory.cu 
    CUPTI_init_unscribe.cu 
    load_model.cu 
    streams.cu 
    callbackfunc.cu 


    )

include(CTest)
enable_testing()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
# 禁用 C 优化
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
# 禁用 CUDA 优化
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0")

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 11)
target_link_libraries(intercept_launch cuda cudart cupti)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(intercept_launch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(intercept_launch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_library(CUPTI_LIB cupti PATHS /usr/local/cuda/extras/CUPTI/lib64)


