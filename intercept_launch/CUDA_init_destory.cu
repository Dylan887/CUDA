#include "interceptor.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA Driver API 初始化和模块加载
void initializeCUDA() {
    CUresult result;
    CUdevice device;
    int deviceCount;

    // 初始化 CUDA Driver API
    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to initialize CUDA Driver API: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    // 获取系统中 CUDA 设备的数量
    result = cuDeviceGetCount(&deviceCount);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to get CUDA device count: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    // 输出系统中的设备数量
    printf("Number of CUDA devices: %d\n", deviceCount);

    // 逐个尝试获取可用的 CUDA 设备
    for (int i = 0; i < deviceCount; ++i) {
        result = cuDeviceGet(&device, i);
        if (result == CUDA_SUCCESS) {
            printf("Successfully got device %d\n", i);
            // 在这里你可以继续使用这个设备
            // 比如初始化上下文等
                // 创建上下文
            CUcontext context;
            result = cuCtxCreate(&context, 0, device);
            if (result != CUDA_SUCCESS) {
                const char *errorStr;
                cuGetErrorString(result, &errorStr);
                fprintf(stderr, "Failed to create CUDA context: %s\n", errorStr);
                exit(EXIT_FAILURE);
    }
            break;  // 成功获取设备后退出循环
        } else {
            const char *errorStr;
            cuGetErrorString(result, &errorStr);
            fprintf(stderr, "Failed to get CUDA device %d: %s\n", i, errorStr);
        }
    }

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "No CUDA devices available or failed to retrieve them.\n");
        exit(EXIT_FAILURE);
    }
}

// 函数定义：销毁当前 CUDA 上下文
void destroyCurrentContext() {
    CUcontext currentContext;
    CUresult result;

    // 获取当前的 CUDA 上下文
    result = cuCtxGetCurrent(&currentContext);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to get current CUDA context: %s\n", errorStr);
        return;  // 无法获取上下文，直接返回
    }

    // 销毁当前 CUDA 上下文
    result = cuCtxDestroy(currentContext);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to destroy CUDA context: %s\n", errorStr);
    } else {
        printf("CUDA context destroyed successfully.\n");
    }
}