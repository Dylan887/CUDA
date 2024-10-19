# 通过gpgpu模拟器获取内存地址
GPGPU-sim能够在Linux系统下，提供对GPU的功能模拟和性能仿真，让你在没有装NVIDIA显卡的情况下可以编译并运行CUDA程序。当然它更重要的意义是，可以通过修改仿真参数，让开发者修改GPU内部架构，并进行性能仿真，以针对自己的项目需求进行更好的代码设计，获得更好的性能表现


### **步骤**

cuda 版本不宜太高，具体适用的版本在clone下来的git里有 常用的版本基本都可以 
> `This version of GPGPU-Sim has been tested with a subset of CUDA version 4.2, 5.0, 5.5, 6.0, 7.5, 8.0, 9.0, 9.1, 10, and 11`
### 1. 获取项目代码
  ```bash
  git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
  ```
### 2. 进入目录设置环境 ，`source setup_environment`要出现successed才算成功。
  ```bash
  cd gpgpu-sim_distribution
  source setup_environment
  ```
如果不行，执行以下步骤：
```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH
sudo apt-get install xutils-dev
sudo apt-get install bison
sudo apt-get install flex
sudo apt-get install libgl1-mesa-dev

```
### 3. 编译，过程可能比较久
```bash
  cd $GPGPUSIM_ROOT
  make
  make docs
  ```
### 4. 检查配置文件：gpgpusim.config 是架构配置信息
  
  ```
  $GPGPUSIM_ROOT/configs/tested-cfgs/<gpgpusim.config>
  ```

### 5.编写cuda程序： 矩阵乘法 matrixmul.cu 示例
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
### 6. 适用gpgpusim编译cuda程序：
在使用gpgpu-sim时，将要编译的CUDA源代码（.cu文件）复制到
/configs/tested-cfgs/目录中的某个文件里，这个文件夹里是不同型号GPU的模拟config文件，例如想模拟这个程序在GTX480显卡上的运行状况，就将代码复制到GTX480对应的文件夹下，然后在命令行用nvcc进行编译，编译时一定要加参数，如下
```bash
nvcc --cudart shared matrixmul.cu
```



### 7. 查看链路是否运行正确：
```bash
ldd a.out
```

### 8. 运行a.out文件即可

---
### Ref
1. https://blog.csdn.net/NKU_Yang/article/details/114662776
2. https://github.com/gpgpu-sim/gpgpu-sim_distribution