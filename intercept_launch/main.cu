// main.cu

#include "interceptor.h"
#include "globalconfig.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// 定义内核函数指针
CUfunction matrixAdd_func = nullptr;
CUfunction matrixSub_func = nullptr;
CUfunction matrixMul_func = nullptr;

KernelFunction kernelFunctions[] = {
    {&matrixAdd_func, "matrixAdd",4},
    {&matrixSub_func, "matrixSub",4},
    {&matrixMul_func, "matrixMul",4}
};

int kernel_nums = 3;


int main(int argc, char *argv[]) {
    // 定义全局变量
    CUpti_SubscriberHandle subscriber;
    CUptiResult res;

    // 初始化 CUDA Driver API 和加载模块
    initializeCUDA();
    initCUPTI(&subscriber, &res, (CUpti_CallbackFunc)cuptiCallback);

    const char *path = "kernels.ptx";
    loadModuleAndFunctions(path, kernelFunctions,3);

    
    
    // 矩阵尺寸
    int N = 3;
    size_t bytes = N * N * sizeof(int);

    // 分配并初始化主机内存
    int *h_A = (int *)malloc(bytes);
    int *h_B = (int *)malloc(bytes);
    int *h_C_add = (int *)malloc(bytes);
    int *h_C_sub = (int *)malloc(bytes);
    int *h_C_mul = (int *)malloc(bytes);

    // 初始化矩阵数据
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // 分配设备内存
    CUdeviceptr d_A, d_B, d_C_add, d_C_sub, d_C_mul;
    CUresult result;

    result = cuMemAlloc(&d_A, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    result = cuMemAlloc(&d_B, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    result = cuMemAlloc(&d_C_add, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to allocate device memory for C_add: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    result = cuMemAlloc(&d_C_sub, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to allocate device memory for C_sub: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    result = cuMemAlloc(&d_C_mul, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to allocate device memory for C_mul: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    // 将数据从主机拷贝到设备
    result = cuMemcpyHtoD(d_A, h_A, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to copy data from host to device for A: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    result = cuMemcpyHtoD(d_B, h_B, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to copy data from host to device for B: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    // 定义线程块和网格尺寸
    int THREADS = 16;
    dim3 threadsPerBlock(THREADS, THREADS, 1);
    dim3 blocksPerGrid((N + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS, 1);

    // 创建 CUDA 流
    
    cudaStream_t* streams = createStreams(kernel_nums);

    // 启动内核
    printf("Launching matrixAdd kernel\n");
    launchMatrixAdd(blocksPerGrid, threadsPerBlock, streams[0], d_A, d_B, d_C_add, N);
    

    printf("Launching matrixSub kernel\n");
    launchMatrixSub(blocksPerGrid, threadsPerBlock, streams[1], d_A, d_B, d_C_sub, N);

    printf("Launching matrixMul kernel\n");
    launchMatrixMul(blocksPerGrid, threadsPerBlock, streams[2], d_A, d_B, d_C_mul, N);

    // 等待所有内核完成
    result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to synchronize device in main: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    // 将结果从设备拷贝回主机
    result = cuMemcpyDtoH(h_C_add, d_C_add, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to copy result from device to host for C_add: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    result = cuMemcpyDtoH(h_C_sub, d_C_sub, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to copy result from device to host for C_sub: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    result = cuMemcpyDtoH(h_C_mul, d_C_mul, bytes);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to copy result from device to host for C_mul: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }

    // 打印结果
    printf("Matrix Addition Result:\n");
    for (int i = 0; i < N * N; i++) {
        printf("%d ", h_C_add[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }
    printf("\n");

    printf("Matrix Subtraction Result:\n");
    for (int i = 0; i < N * N; i++) {
        printf("%d ", h_C_sub[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }
    printf("\n");

    printf("Matrix Multiplication Result:\n");
    for (int i = 0; i < N * N; i++) {
        printf("%d ", h_C_mul[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }
    printf("\n");

    // 释放设备内存
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C_add);
    cuMemFree(d_C_sub);
    cuMemFree(d_C_mul);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C_add);
    free(h_C_sub);
    free(h_C_mul);

    // 销毁 CUDA 流
    destroyStreams(streams,kernel_nums);

    // 取消订阅 CUPTI
    unsubscribeCUPTI(&subscriber,&res);

    // 销毁 CUDA 上下文
    destroyCurrentContext();
    printf("Program completed successfully.\n");


    return 0;
}
