// interceptor.h

#ifndef INTERCEPTOR_H
#define INTERCEPTOR_H

#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include "globalconfig.h"




// 定义内核启动参数结构体
typedef struct {
    KernelType kernelType;
    CUfunction func; // CUfunction 类型函数指针
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    CUstream stream;
    dim3 originalGridDim;
} KernelLaunchParams;

typedef struct {
    CUfunction *funcPtr;
    const char *funcName;
    int numArgs;  // 手动维护参数个数
} KernelFunction;

// 声明全局变量
extern KernelLaunchParams kernelQueue[MAX_KERNELS];
extern std::atomic<int> kernelCount;
extern std::atomic<bool> inCallback;


// 当前内核类型和函数指针
extern KernelType currentKernelType;
extern CUfunction currentKernelFunc;



// CUPTI 回调函数声明
void CUPTIAPI cuptiCallback(void *userdata, CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);

void initializeCUDA();
void initCUPTI(CUpti_SubscriberHandle *subscriber, CUptiResult *res, CUpti_CallbackFunc callbackFunc);
void destroyCurrentContext();
void unsubscribeCUPTI(CUpti_SubscriberHandle *subscriber, CUptiResult *res) ;
CUstream* createStreams(int numStreams);
void destroyStreams(cudaStream_t *streams, int numStreams);
void loadModuleAndFunctions(const char *modulePath, KernelFunction *kernelFunctions,int numKernels);




#endif // INTERCEPTOR_H
