#include "interceptor.h"
#include <stdio.h>
#include <stdlib.h>

extern CUfunction matrixAdd_func;
extern CUfunction matrixSub_func;
extern CUfunction matrixMul_func;

// 包装函数实现
void launchMatrixAdd(dim3 gridDim, dim3 blockDim, CUstream stream,
                    CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N) {
    currentKernelType = KERNEL_MATRIX_ADD;
    currentKernelFunc = matrixAdd_func;

    // 准备内核参数
    void *args[] = { &d_A, &d_B, &d_C, &N };

    // 启动内核
    CUresult result = cuLaunchKernel(
        currentKernelFunc,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0, // sharedMem
        stream,
        args,
        nullptr // extra
    );

    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        //fprintf(stderr, "Failed to launch matrixAdd: %s\n", errorStr);
    };
}

void launchMatrixSub(dim3 gridDim, dim3 blockDim, CUstream stream,
                    CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N) {
    currentKernelType = KERNEL_MATRIX_SUB;
    currentKernelFunc = matrixSub_func;

    // 准备内核参数
    void *args[] = { &d_A, &d_B, &d_C, &N };

    // 启动内核
    CUresult result = cuLaunchKernel(
        currentKernelFunc,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0, // sharedMem
        stream,
        args,
        nullptr // extra
    );

    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        //fprintf(stderr, "Failed to launch matrixSub: %s\n", errorStr);
    };
}

void launchMatrixMul(dim3 gridDim, dim3 blockDim, CUstream stream,
                    CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N) {
    currentKernelType = KERNEL_MATRIX_MUL;
    currentKernelFunc = matrixMul_func;

    // 准备内核参数
    void *args[] = { &d_A, &d_B, &d_C, &N };

    // 启动内核
    CUresult result = cuLaunchKernel(
        currentKernelFunc,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0, // sharedMem
        stream,
        args,
        nullptr // extra
    );

    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
       // fprintf(stderr, "Failed to launch matrixMul: %s\n", errorStr);
    };
}
