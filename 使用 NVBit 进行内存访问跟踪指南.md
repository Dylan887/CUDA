# **使用 NVBit 进行内存访问跟踪指南**
**NVBit（NVIDIA Binary Instrumentation Tool）** 是 NVIDIA 提供的一款轻量级、灵活的 GPU 二进制插桩框架，可以帮助开发者在不修改源代码的情况下，跟踪 CUDA 程序的内存访问。

具有以下特点：

* **轻量级和高效**：对应用程序的性能影响较小。
* **灵活性**：支持用户自定义插件，实现多种分析功能。
* **透明性**：无需修改应用程序源码或重新编译。
* **适用性**：支持 Maxwell（Compute Capability 5.0）及以上架构的 GPU。

NVBit 通过在程序运行时对 CUDA 内核的二进制代码（SASS）进行动态插桩，实现对内核执行的监控和分析。
### NVBit 的工作原理
1. 直接与 CUDA 驱动程序交互:
NVBit 直接与 CUDA 驱动程序交互，处理已经编译成 SASS 机器代码的程序。它通过 LD_PRELOAD 机制在运行时注入库，使其能够在加载任何其他库之前加载指定的共享库。
2. 应用程序二进制接口 (ABI):
ABI 定义了调用者和被调用者之间的接口属性，例如寄存器的使用、参数传递方式等。NVBit 使用动态汇编器生成符合 ABI 的代码，以便能够将自定义 CUDA 设备函数注入到现有的应用程序中。
3. CUDA API 回调:
NVBit 为所有 CUDA 驱动程序 API 提供回调机制。这使得 NVBit 可以在 CUDA 程序调用任何 CUDA API 时截获这些调用，并插入自定义代码或进行分析。
NVBit 还提供在应用程序启动和终止时的特定回调功能，可以在程序的整个生命周期中进行监控和干预。
### NVBit 工具开发流程
1. 开发 .cu 文件:
使用 NVBit API 编写 CUDA 设备函数和回调函数。例如，在 .cu 文件中定义一个设备函数 incr_counter，用于在每次指令执行时计数。
2. 编译 .cu 文件:

使用 NVCC 编译器将 .cu 文件编译成目标文件。
3. 链接生成共享库:

将编译好的目标文件与静态库 libnvbit.a 链接，生成一个共享库（通常是 .so 文件）。
示例生成的共享库：libmy_nvbit_tool.so。

https://blog.csdn.net/m0_63471305/article/details/139804027

### 1.下载地址：
```bash
https://github.com/NVlabs/NVBit/releases
```
解压 `.bz2`：
```bash
tar -xvjf ./nvbit-Linux-x86_64-1.7.1.tar.bz2
```
**如果直接`git clone https://github.com/NVlabs/NVBit.git`会只有两个txt、md两个文件**

或者`git clone https://github.com/CoffeeBeforeArch/nvbit_tools.git` --->版本较旧

### 2.进入 NVBit 目录并设置环境变量：

```bash
cd nvbit-release
export NVBIT_ROOT=$(pwd)
export LD_LIBRARY_PATH=NVBITROOT:NVBIT_ROOT:LD_LIBRARY_PATH
```
### 3.进入tools里会有mem_trace目录：
mem_trace.cu是一个插槽功能的函数
编译：
```bash
 make ARCH=sm_80 #这里Makefile里是`ARCH?=ALL` 手动指定架构
```

编译成功后，将生成 `mem_trace.so` 插件文件，成为共享库。

### 4.编译cuda程序：
假设是矩阵相乘 matrixmul.cu :

```bash
nvcc -o matrixmul matrixmul.cu
```

```c++
#include <stdio.h>
#include <cuda_runtime.h>

// 矩阵维度
#define N 1024  // 矩阵大小 N x N

// CUDA内核，执行矩阵乘法
__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    // 获取当前线程对应的行和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0;
    
    // 矩阵乘法的核心计算：C(row, col) = A(row, :) * B(:, col)
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;  // 主机内存
    float *d_A, *d_B, *d_C;  // 设备内存

    // 分配主机内存
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // 初始化矩阵A和B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;  // 简单起见，将A和B初始化为全1矩阵
        h_B[i] = 1.0f;
    }

    // 分配设备内存
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 将主机内存拷贝到设备内存
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格维度
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用CUDA内核
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 将结果从设备内存拷贝到主机内存
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != N) {
            success = false;
            break;
        }
    }
    if (success) {
        printf("矩阵相乘成功!\n");
    } else {
        printf("矩阵相乘失败!\n");
    }

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```



### 5. 注入共享库

在运行时通过 LD_PRELOAD 机制将共享库注入到目标应用程序中。
```bash
LD_PRELOAD=nvbit-release/tools/mem_trace.so ./matrixmul
```
