# cudnn的section介绍

`cuDNN`（CUDA Deep Neural Network library）是NVIDIA提供的用于加速深度学习框架中的卷积操作的库，特别适用于运行在CUDA平台上的GPU。
### cudnn主要的二进制文件
`cuDNN` (CUDA Deep Neural Network) 库包含多个二进制文件，每个文件负责加速深度学习任务中的不同操作，特别是卷积、归一化、激活等神经网络操作。这些二进制文件主要是动态链接库文件（`.so` 文件），存储了针对不同任务的具体实现。以下是 `cuDNN` 库的主要二进制文件及其结构和作用的详细介绍：

1. **`libcudnn.so` (主文件)**

   cudnn的核心库文件，包含了几乎所有的 API 实现。这个文件提供了用于深度学习任务的核心函数，比如卷积、激活、归一化、池化、RNN等操作的实现。


2. **`libcudnn_ops_infer.so`**

   cuDNN的推理库文件，专门为推理阶段的操作进行优化。它包含了用于前向推理（inference）的高效实现，适用于推理时的神经网络操作。

3. **`libcudnn_ops_train.so`**


   cuDNN训练库文件，专门为训练阶段的操作进行优化。与推理不同，训练阶段涉及到反向传播和梯度计算，因此这个库包含了与训练相关的函数和API。


4. **`libcudnn_cnn_infer.so`**
用于卷积神经网络（CNN）推理的优化库。该文件针对卷积操作进行了大量优化，特别是在推理阶段的卷积计算。


5. **`libcudnn_cnn_train.so`**
用于卷积神经网络（CNN）训练的优化库。该文件与libcudnn_ops_train.so类似，但专门针对卷积操作的训练进行了优化。


 6. **`libcudnn_adv_infer.so`**
提供了高级推理操作 ，针对更复杂的神经网络模型，提供了额外的优化。它可能包括更复杂的卷积算法或其他推理阶段的高级特性。


7. **`libcudnn_adv_train.so`**
提供高级训练操作的库，特别是针对大规模、复杂的神经网络训练任务。


8. **`libcudnn_static.a`**


   cuDNN的静态链接库版本。与.so 文件不同，静态库会在编译时直接链接到应用程序中，而不是在运行时动态加载。

 9. **调试与辅助库**

* `cuDNN` 库通常还会包含一些调试版本的库文件，它们可能包含更多的调试符号和调试信息，用于在开发和调试深度学习模型时使用。这些库的文件名可能以 `-dbg` 结尾，包含更多的 `.debug` 和 `.symtab` 段，便于调试工具解析符号和源代码信息。


#### 本文以 `libcudnn.so` 文件为例，详细介绍其常见的section.

**使用 `objdump` 工具查看 `cuDNN` 库的所有 section：**

```bash
 objdump -h /usr/lib/x86_64-linux-gnu/libcudnn.so
```
```bash
/usr/lib/x86_64-linux-gnu/libcudnn.so:     file format elf64-x86-64

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  0 .note.gnu.build-id 00000024  00000000000001c8  00000000000001c8  000001c8  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  1 .gnu.hash     00000960  00000000000001f0  00000000000001f0  000001f0  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  2 .dynsym       000020d0  0000000000000b50  0000000000000b50  00000b50  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  3 .dynstr       00002354  0000000000002c20  0000000000002c20  00002c20  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  4 .gnu.version  000002bc  0000000000004f74  0000000000004f74  00004f74  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  5 .gnu.version_d 00000030  0000000000005230  0000000000005230  00005230  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  6 .gnu.version_r 00000100  0000000000005260  0000000000005260  00005260  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  7 .rela.dyn     000000f0  0000000000005360  0000000000005360  00005360  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  8 .rela.plt     000006d8  0000000000005450  0000000000005450  00005450  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  9 .init         0000001a  0000000000005b28  0000000000005b28  00005b28  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 10 .plt          000004a0  0000000000005b50  0000000000005b50  00005b50  2**4
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 11 .plt.got      00000010  0000000000005ff0  0000000000005ff0  00005ff0  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 12 .text         00014566  0000000000006000  0000000000006000  00006000  2**4
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 13 .fini         00000009  000000000001a568  000000000001a568  0001a568  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 14 .rodata       000021c4  000000000001a578  000000000001a578  0001a578  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 15 .eh_frame_hdr 00000ac4  000000000001c73c  000000000001c73c  0001c73c  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 16 .eh_frame     00004ef4  000000000001d200  000000000001d200  0001d200  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 17 .gcc_except_table 000013fc  00000000000220f4  00000000000220f4  000220f4  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 18 .init_array   00000008  0000000000223d30  0000000000223d30  00023d30  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 19 .fini_array   00000008  0000000000223d38  0000000000223d38  00023d38  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 20 .jcr          00000008  0000000000223d40  0000000000223d40  00023d40  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 21 .data.rel.ro  00000008  0000000000223d48  0000000000223d48  00023d48  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 22 .dynamic      00000280  0000000000223d50  0000000000223d50  00023d50  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 23 .got          00000030  0000000000223fd0  0000000000223fd0  00023fd0  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 24 .got.plt      00000260  0000000000224000  0000000000224000  00024000  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 25 .data         0000000c  0000000000224260  0000000000224260  00024260  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 26 .bss          00001158  0000000000224280  0000000000224280  0002426c  2**5
                  ALLOC
```


### 各个 Section 的详细介绍

1. **`.note.gnu.build-id` (Section 0)**
   * **用途**：这个 section 通常包含构建工具生成的唯一构建 ID。用于调试或验证目的，帮助唯一标识不同的编译版本。
   
2. **`.gnu.hash` (Section 1)**
   * **用途**：用于动态链接时的符号查找，存储哈希表。GNU hash 提高了符号查找效率，尤其在动态库中。
  
3. **`.dynsym` (Section 2)**
   * **用途**：动态符号表，用于在动态链接时查找和解析符号。这是库中所有导出符号（如函数、全局变量等）的列表。
  
4. **`.dynstr` (Section 3)**
   * **用途**：字符串表，用于存储符号表中的符号名称。符号名称在动态链接时通过这个段来访问。
   
5. **`.gnu.version` (Section 4)**
   * **用途**：符号版本信息。此段用于跟踪和管理 ELF 文件中的符号版本，确保正确的符号被解析。
   
6. **`.gnu.version_d` (Section 5)**
   * **用途**：符号版本定义，记录符号的版本声明。它与 `.gnu.version` 一起工作，用于跟踪符号的版本变化。
   
7. **`.gnu.version_r` (Section 6)**
   * **用途**：符号版本需求，描述此库所依赖的符号版本。

8. **`.rela.dyn` (Section 7)**
   * **用途**：动态重定位条目。当加载库时，重定位表帮助调整符号和数据的地址。
   
9. **`.rela.plt` (Section 8)**
   * **用途**：为 `PLT`（Procedure Linkage Table）提供重定位信息，处理延迟绑定的函数调用。
   
10. **`.init` (Section 9)**
    * **用途**：初始化代码。通常在程序启动时执行，用于初始化全局状态或资源。
   
11. **`.plt` (Section 10)**
    * **用途**：`PLT` 段用于延迟绑定函数调用。当程序调用某个外部库函数时，函数的实际地址会通过这个段解析。
   
12. **`.plt.got` (Section 11)**
    * **用途**：`GOT`（全局偏移量表）与 `PLT` 共同用于延迟绑定的函数调用。
    
13. **`.text` (Section 12)**
    * **用途**：这是最重要的段，包含了库的可执行代码。`cuDNN` 的所有 CPU 端功能都在这里实现，部分与 GPU kernel 的调用有关。
    
14. **`.fini` (Section 13)**
    * **用途**：终结代码。通常在程序结束时执行，用于清理资源或做最后的处理。
    
15. **`.rodata` (Section 14)**
    * **用途**：只读数据段，通常用于存储常量和字符串等不可修改的数据。
    
16. **`.eh_frame_hdr` 和 `.eh_frame` (Section 15, 16)**
    * **用途**：用于异常处理的堆栈展开信息，特别是在 C++ 程序中的异常处理机制中使用。
    
17. **`.gcc_except_table` (Section 17)**
    * **用途**：异常处理时用于栈展开的辅助数据，通常是由 GCC 编译器生成。
   
18. **`.init_array` (Section 18)**
    * **用途**：存储程序初始化时需要执行的函数指针（构造函数）。
   
19. **`.fini_array` (Section 19)**
    * **用途**：存储程序终结时需要执行的函数指针（析构函数）。
    
20. **`.jcr` (Section 20)**
    * **用途**：Java 类注册表（JCR），用于 Java 异常处理的辅助数据，非 Java 程序不会使用。
   
21. **`.data.rel.ro` (Section 21)**
    * **用途**：已初始化但只读的全局数据（通过重定位后的只读数据）。
  
22. **`.dynamic` (Section 22)**
    * **用途**：存储动态链接信息，包括库的依赖关系和符号表信息。帮助动态链接器解析符号。
   
23. **`.got` (Section 23)**
    * **用途**：全局偏移量表，存储动态链接时需要的符号地址。
   
24. **`.got.plt` (Section 24)**
    * **用途**：与 `PLT` 配合，动态解析函数地址，主要用于延迟绑定。
   
25. **`.data` (Section 25)**
    * **用途**：存储已初始化的全局变量和静态变量。
    
26. **`.bss` (Section 26)**
    * **用途**：存储未初始化的全局变量和静态变量。在程序启动时，这些变量会被初始化为 0。
    

### GPU Kernel 位置和分布

1. GPU kernel 调用流程
   * GPU kernel 的实际代码并不直接存储在 `.text` 段中。CUDA 动态库的 GPU kernel 通常会通过 CUDA Runtime API 调用并从 `.nv_fatbin` 或其他特定段中加载。在这里，我们没有看到 `.nv_fatbin`，因为其在运行时动态加载。
2. GPU kernel 的加载和执行
   * 虽然在 ` objdump` 输出中没有显示 `.nv_fatbin` 段，但 GPU kernel 的调用逻辑会通过 `.text` 段中的 CPU 端代码调用 CUDA API（如 `cudaLaunchKernel`），这些 API 会进一步加载对应的 GPU kernel 并将其传递到 GPU 上执行。
   * 如果要查看具体的 GPU kernel 代码，可以使用 `cuobjdump` 工具提取和查看这些 kernel 的 PTX 或 SASS 代码。
### 注：上述结果的实验环境
**1. 操作系统版本**
```bash
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 18.04.5 LTS
Release:        18.04
Codename:       bionic
```
**2. nvcc 版本**
``` bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:08:53_PST_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
```

**3. cudnn 版本**
``` bash
ii  libcudnn8                              8.8.0.121-1+cuda11.8                amd64        cuDNN runtime libraries
ii  libcudnn8-dev                          8.8.0.121-1+cuda11.8                amd64        cuDNN development libraries and headers
ii  libcudnn8-samples                      8.8.0.121-1+cuda11.8                amd64        cuDNN samples

```
---
### Ref
1. https://docs.nvidia.com/deeplearning/cudnn/index.html
2. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
3. https://developer.nvidia.com/nsight-systems
4. https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump-overview
   