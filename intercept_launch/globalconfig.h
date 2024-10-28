#ifndef GLOBALCONFIG_H
#define GLOBALCONFIG_H


// 定义内核类型枚举
typedef enum {
    KERNEL_UNKNOWN,
    KERNEL_MATRIX_ADD,
    KERNEL_MATRIX_SUB,
    KERNEL_MATRIX_MUL
} KernelType;

// 仅声明全局变量
extern CUfunction matrixAdd_func;
extern CUfunction matrixSub_func;
extern CUfunction matrixMul_func;

// 使用 constexpr 来定义编译时常量
constexpr int KERNEL_THRESHOLD = 2;
constexpr int MAX_KERNELS = 1000;

// 包装函数声明
void launchMatrixAdd(dim3 gridDim, dim3 blockDim, CUstream stream,
                    CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N);
void launchMatrixSub(dim3 gridDim, dim3 blockDim, CUstream stream,
                    CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N);
void launchMatrixMul(dim3 gridDim, dim3 blockDim, CUstream stream,
                    CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N);

#endif