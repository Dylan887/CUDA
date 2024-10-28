#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// 封装的函数：创建多个 CUDA 流
CUstream* createStreams(int numStreams) {
    // 动态分配 CUstream 数组
    CUstream *streams = (CUstream*) malloc(numStreams * sizeof(CUstream));
    if (streams == NULL) {
        fprintf(stderr, "Failed to allocate memory for streams.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numStreams; i++) {
        CUresult result = cuStreamCreate(&streams[i], CU_STREAM_DEFAULT);
        if (result != CUDA_SUCCESS) {
            const char *errorStr;
            cuGetErrorString(result, &errorStr);
            fprintf(stderr, "Failed to create stream %d: %s\n", i + 1, errorStr);
            exit(EXIT_FAILURE);
        } else {
            printf("Successfully created stream %d\n", i + 1);
        }
    }

    return streams;
}

// 封装销毁 CUDA 流的函数
void destroyStreams(cudaStream_t *streams, int numStreams) {
    for (int i = 0; i < numStreams; i++) {
        cudaError_t err = cudaStreamDestroy(streams[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to destroy stream #%d: %s\n", i + 1, cudaGetErrorString(err));
        } else {
            printf("Successfully destroyed stream #%d\n", i + 1);
        }
    }
}