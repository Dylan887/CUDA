#include "interceptor.h"
#include <stdio.h>
#include <stdlib.h>


// 封装的加载函数
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>



// 封装的加载模块及内核函数的函数
void loadModuleAndFunctions(const char *modulePath, KernelFunction *kernelFunctions,int numKernels) {
    // 加载模块
    CUmodule module;
    CUresult result = cuModuleLoad(&module, modulePath);
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "Failed to load CUDA module: %s\n", errorStr);
        exit(EXIT_FAILURE);
    }
    printf("Successfully loaded module: %s\n", modulePath);
    
    // 加载内核函数指针
    for (int i = 0; i < numKernels; i++) {
        result = cuModuleGetFunction(kernelFunctions[i].funcPtr, module, kernelFunctions[i].funcName);
        if (result != CUDA_SUCCESS) {
            const char *errorStr;
            cuGetErrorString(result, &errorStr);
            fprintf(stderr, "Failed to get function %s: %s\n", kernelFunctions[i].funcName, errorStr);
            exit(EXIT_FAILURE);
        }
        printf("Successfully loaded function: %s\n", kernelFunctions[i].funcName);
    }
}





