// interceptor.cu

#include "interceptor.h"
#include <stdio.h>
#include <stdlib.h>


// 定义全局变量
KernelLaunchParams kernelQueue[MAX_KERNELS];
std::atomic<int> kernelCount(0);
std::atomic<bool> inCallback(false);



// 当前内核类型和函数指针
KernelType currentKernelType = KERNEL_UNKNOWN;
CUfunction currentKernelFunc = nullptr;


extern KernelFunction kernelFunctions[];
extern int  kernel_nums;
std::atomic<int> totalInterceptedKernels(0);

// 假设 params->kernelParams 是内核的参数数组
int getKernelNumArgs(CUfunction func, KernelFunction *kernelFunctions, int numKernels) {
    for (int i = 0; i < numKernels; i++) {
        if (*(kernelFunctions[i].funcPtr) == func) {
            return kernelFunctions[i].numArgs;
        }
    }
    return -1;  // 如果没有找到匹配的内核，返回 -1 表示出错
}


// CUPTI 回调函数实现
void CUPTIAPI cuptiCallback(void *userdata, CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
    if (inCallback.load(std::memory_order_acquire)) {
        // 当前正在处理内核，避免递归调用
        return;
    }

    // 拦截 CUDA Driver API 的 cuLaunchKernel
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API && cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            
            if ((kernel_nums - totalInterceptedKernels.load(std::memory_order_relaxed))  < KERNEL_THRESHOLD) {
                printf("Intercepted kernel launch, but remaining kernels are less than threshold. Launching directly...\n");
                totalInterceptedKernels.fetch_add(1, std::memory_order_relaxed);
                return;  // 不拦截，直接返回，系统自动执行剩余内核
            }
            totalInterceptedKernels.fetch_add(1, std::memory_order_relaxed);
            // 增加内核计数
            int index = kernelCount.fetch_add(1, std::memory_order_relaxed);
            if (index >= MAX_KERNELS) {
                fprintf(stderr, "Kernel queue overflow!\n");
                exit(EXIT_FAILURE);
            }

            // 获取 cuLaunchKernel 的参数
            const cuLaunchKernel_params *params = (const cuLaunchKernel_params *)(cbInfo->functionParams);

            // 存储内核启动参数
            kernelQueue[index].kernelType = currentKernelType;
            kernelQueue[index].func = params->f;
            kernelQueue[index].gridDim = dim3(params->gridDimX, params->gridDimY, params->gridDimZ);
            kernelQueue[index].blockDim = dim3(params->blockDimX, params->blockDimY, params->blockDimZ);

            // 获取内核参数的数量
            //int numArgs = getNumArgs(params->kernelParams);  // 假设有 4 个参数，你可以根据实际情况动态调整
            int numArgs = getKernelNumArgs(params->f, kernelFunctions,  kernel_nums);
            // 为每个参数分配独立的内存并进行深拷贝
            kernelQueue[index].args = (void **)malloc(numArgs * sizeof(void *));
            for (int i = 0; i < numArgs; i++) {
                // 假设所有参数都是指针，按指针大小处理（通用处理）
                size_t argSize = sizeof(void *); // 你可以根据实际类型来调整大小
                kernelQueue[index].args[i] = malloc(argSize);  // 分配内存
                memcpy(kernelQueue[index].args[i], params->kernelParams[i], argSize);  // 拷贝参数
            }

            kernelQueue[index].sharedMem = params->sharedMemBytes;
            kernelQueue[index].stream = params->hStream;  // 使用已有的流，不重复创建
            kernelQueue[index].originalGridDim = kernelQueue[index].gridDim;  // 保存原始的 gridDim

            printf("Intercepted kernel launch #%d\n", index + 1);

            // 修改参数，阻止 kernel 执行（方法二：将函数指针设置为 NULL）
            ((cuLaunchKernel_params *)(cbInfo->functionParams))->f = nullptr;
                

            // 当内核计数达到阈值时，批量启动所有内核
            if (kernelCount.load(std::memory_order_relaxed)  >= KERNEL_THRESHOLD) {
                inCallback.store(true, std::memory_order_release);

                printf("Launching %d kernels together...\n", kernelCount.load());

                for (int i = 0; i < kernelCount.load(); i++) {
                    
                    // 恢复原始的 gridDim
                    kernelQueue[i].gridDim = kernelQueue[i].originalGridDim;

                    // 获取函数指针
                    CUfunction func = kernelQueue[i].func;
                    if (func == nullptr) {
                        fprintf(stderr, "Kernel function pointer is NULL for kernel #%d\n", i + 1);
                        continue;
                    }

                    // 启动内核，确保每个内核使用独立的 CUDA 流
                    CUresult result = cuLaunchKernel(
                        func,
                        kernelQueue[i].gridDim.x,
                        kernelQueue[i].gridDim.y,
                        kernelQueue[i].gridDim.z,
                        kernelQueue[i].blockDim.x,
                        kernelQueue[i].blockDim.y,
                        kernelQueue[i].blockDim.z,
                        kernelQueue[i].sharedMem,
                        kernelQueue[i].stream,  // 确保每个内核的流是独立的
                        kernelQueue[i].args,
                        nullptr  // extra
                    );

                    if (result != CUDA_SUCCESS) {
                        const char *errorStr;
                        cuGetErrorString(result, &errorStr);
                        fprintf(stderr, "Failed to launch kernel #%d: %s\n", i + 1, errorStr);
                    } else {
                        printf("Successfully launched kernel #%d\n", i + 1);
                    }
                }

                // 同步设备
                CUresult syncResult = cuCtxSynchronize();
                if (syncResult != CUDA_SUCCESS) {
                    const char *errorStr;
                    cuGetErrorString(syncResult, &errorStr);
                    fprintf(stderr, "Failed to synchronize device: %s\n", errorStr);
                }

                // 释放所有参数的内存，避免内存泄漏
                for (int i = 0; i < kernelCount.load(); i++) {
                    for (int j = 0; j < numArgs; j++) {
                        free(kernelQueue[i].args[j]);
                    }
                    free(kernelQueue[i].args);
                }

                // 重置标志位和计数器
                inCallback.store(false, std::memory_order_release);
                kernelCount.store(0, std::memory_order_relaxed);
            }
        }
    }
}

