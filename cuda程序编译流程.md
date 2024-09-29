# cuda程序编译流程

本文以cuda example的matrixMul矩阵乘法为例说明cuda程序的编译流程。

### 1. **源代码 `.cu` 文件**

在`matrixMul`示例中，源代码文件 `matrixMul.cu` 是典型的CUDA程序，包含以下部分：
#### 流程图
![alt text](/images/20161225140143964.png)
* **主机代码（Host Code）**：运行在CPU上的代码，用于数据准备、调用CUDA内核（kernel）等。
* **设备代码（Device Code）**：运行在GPU上的CUDA内核，负责矩阵乘法计算。

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA内核：用于执行矩阵乘法
__global__ void MatrixMulKernel(float* C, const float* A, const float* B, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 矩阵的列索引
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 矩阵的行索引

    if (x < width && y < width) {
        float sum = 0;
        for (int i = 0; i < width; ++i) {
            sum += A[y * width + i] * B[i * width + x];  // A的行与B的列对应相乘累加
        }
        C[y * width + x] = sum;  // 结果存储在C矩阵中
    }
}

void randomMatrixInit(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = rand() % 10;  // 随机初始化矩阵中的每个元素
    }
}

int main() {
    int width = 16;  // 矩阵的宽度（假设矩阵是正方形，大小为 width * width）
    int size = width * width;  // 矩阵的总元素个数

    // 分配主机内存
    float* h_A = (float*)malloc(size * sizeof(float));
    float* h_B = (float*)malloc(size * sizeof(float));
    float* h_C = (float*)malloc(size * sizeof(float));

    // 初始化矩阵A和B
    randomMatrixInit(h_A, size);
    randomMatrixInit(h_B, size);

    // 分配设备内存
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, size * sizeof(float));
    cudaMalloc((void**)&d_B, size * sizeof(float));
    cudaMalloc((void**)&d_C, size * sizeof(float));

    // 将主机内存中的数据拷贝到设备内存
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义CUDA网格和块大小
    int blockSize = 16;  // 每个线程块中的线程数（16 x 16 的线程块）
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((width + blockSize - 1) / blockSize, (width + blockSize - 1) / blockSize);

    // 调用CUDA内核进行矩阵乘法运算
    MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, width);

    // 将结果从设备内存拷贝回主机内存
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果（可选）
    printf("Matrix A:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", h_A[i]);
        if ((i + 1) % width == 0) printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", h_B[i]);
        if ((i + 1) % width == 0) printf("\n");
    }

    printf("\nMatrix C (Result):\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", h_C[i]);
        if ((i + 1) % width == 0) printf("\n");
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

```

### 2. **C++ 预处理器**

编译的第一步是C++预处理器，它处理宏定义、头文件包含以及代码中的条件编译部分。在此阶段，`.cu`文件会被预处理，生成中间文件 `.cpp.i`。


### 3. **CUDArtfe 编译步骤（C++ 和 CUDA 编译器）**

#### a.主机代码路径（左侧）


1. **C++ 预处理器**：处理`.cu`文件的主机部分，生成 `.cpp4.ii` 文件。
2. **CUDA 前端编译器 `cudafe++`**：将预处理后的 `.cpp4.ii` 文件转换为 `.cudafe1.cpp` 文件。这个文件包含主机代码和设备代码的占位符。
3. **C++ 编译器**：处理 `.cudafe1.cpp` 文件，生成主机端的 `.o` 或 `.obj` 文件。这是CPU可执行的部分。


#### b.设备代码路径（右侧）

设备代码的编译流程比主机代码复杂，详细流程如下：

1. **C++ 预处理器**：CUDA源代码中的设备代码被提取出来，并进行预处理，生成 `.cpp1.ii` 文件。这个文件是设备代码的初步处理结果。
2. **CUDA 前端编译器 `cudafe`**：cudafe 将 `.cpp1.ii` 转换为 `.cudafe1.gpu`，这是设备端的中间表示文件,它的主要作用是处理CUDA设备代码的结构，提取 `__global__`、`__device__` 和 `__host__` 修饰的函数，并将这些代码转换为适合进一步编译的中间形式。
3. **C 编译器预处理**：对 `.cudafe1.gpu` 进行C预处理，生成 `.cpp2.i` 文件.在这个步骤中，设备代码再次经过C编译器的预处理器。这是为了进一步处理包含的头文件、宏定义等，并对 cudafe1.gpu 文件中的内容进行进一步的C语言级别的预处理。
4. **CUDA 前端编译器 `cudafe`**：再次编译，生成 `.cudafe2.gpu` 文件，准备进一步转换为GPU代码。在这一步，cudafe 再次处理设备代码，分析C语言的结构，并将 .cpp2.i 文件中的设备代码转换为 cudafe2.gpu 文件。这是准备进行PTX代码生成的关键一步，它将设备代码进一步转化为中间表示，准备进入GPU架构相关的编译步骤。
5. **C 编译器预处理**：将 `.cudafe2.gpu` 再次预处理，生成 `.cpp3.i` 文件。
6. **CUDA 编译器 `cicc`**：将 `.cpp3.i` 文件编译为 PTX（并行线程执行）中间代码，输出 `.ptx` 文件。cicc 是CUDA编译器的核心组件之一，它负责将设备代码编译为PTX（Parallel Thread Execution）代码。PTX是NVIDIA的中间代码表示，类似于汇编语言，它是高层次的机器码，独立于具体的GPU硬件架构。
7. **PTX 汇编器 `ptxas`**：将 `.ptx` 文件转化为设备可执行的 `.cubin` 文件。这是最终的设备二进制文件，GPU可以直接执行这个文件。

### 4. **生成胖二进制文件 `fatbinary`**

通过 `fatbinary` 工具，多个 `.cubin` 文件被打包成 `.fatbin` 文件。这是“胖二进制”文件，支持不同架构的GPU设备。

### 5. **生成 `.fatbin.c` 文件**

生成的 `fatbin` 文件会被转换为 `.fatbin.c` 文件，这是C语言代码文件，包含了设备代码的二进制部分，最终将和主机代码一起被编译。



### 6. **最终链接**

最后，主机端的 `.o` 文件和设备端的 `.fatbin.c` 文件通过标准C/C++编译器（如 `gcc` 或 `g++`）进行链接。这个过程会将主机代码和设备代码一起打包成最终的可执行文件。

#### 编译链接全过程
![alt text](/images/GPU编译流程cuda-compilation-from-cu-to-executable.png)

----
### Ref
1. https://blog.csdn.net/dark5669/article/details/53869631
2. https://www.zhangty15226.com/2023/11/25/NVCC%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B/
3. https://blog.csdn.net/fb_help/article/details/80462853
4. https://cloud.baidu.com/article/3224884
5. https://findhao.net/easycoding/2039
6. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
7. https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
8. https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
9. https://docs.nvidia.com/cuda/cuda-driver-api/index.html
10. https://developer.nvidia.com/cuda-examplehttps://developer.nvidia.com/gpu-computing-sdk
