# cublas 的section介绍

cuBLAS 是 NVIDIA 提供的高性能线性代数库，它实现了 BLAS（Basic Linear Algebra Subprograms）的功能，用于在 CUDA 支持的 GPU 上加速矩阵和向量运算。cuBLAS 是专门为在 GPU 上执行大规模线性代数操作设计的，广泛应用于深度学习、科学计算和机器学习等领域。

**使用 `readelf` 工具查看 `libcublas.so.11.3.1.68` 的 section：**

```bash
 readelf -S /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.11.3.1.68
```

```bash
There are 30 section headers, starting at offset 0x6de5af8:

Section Headers:
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  [ 0]                   NULL             0000000000000000  00000000
       0000000000000000  0000000000000000           0     0     0
  [ 1] .note.gnu.build-i NOTE             00000000000001c8  000001c8
       0000000000000024  0000000000000000   A       0     0     4
  [ 2] .gnu.hash         GNU_HASH         00000000000001f0  000001f0
       0000000000000dcc  0000000000000000   A       3     0     8
  [ 3] .dynsym           DYNSYM           0000000000000fc0  00000fc0
       0000000000004560  0000000000000018   A       4     3     8
  [ 4] .dynstr           STRTAB           0000000000005520  00005520
       0000000000002b12  0000000000000000   A       0     0     1
  [ 5] .gnu.version      VERSYM           0000000000008032  00008032
       00000000000005c8  0000000000000002   A       3     0     2
  [ 6] .gnu.version_d    VERDEF           0000000000008600  00008600
       0000000000000030  0000000000000000   A       4     2     8
  [ 7] .gnu.version_r    VERNEED          0000000000008630  00008630
       0000000000000160  0000000000000000   A       4     8     8
  [ 8] .rela.dyn         RELA             0000000000008790  00008790
       000000000009cca8  0000000000000018   A       3     0     8
  [ 9] .rela.plt         RELA             00000000000a5438  000a5438
       0000000000001680  0000000000000018  AI       3    11     8
  [10] .init             PROGBITS         00000000000a6ab8  000a6ab8
       000000000000000e  0000000000000000  AX       0     0     4
  [11] .plt              PROGBITS         00000000000a6ad0  000a6ad0
       0000000000000f10  0000000000000010  AX       0     0     16
  [12] .text             PROGBITS         00000000000a79e0  000a79e0
       00000000009d34dc  0000000000000000  AX       0     0     16
  [13] .fini             PROGBITS         0000000000a7aebc  00a7aebc
       0000000000000009  0000000000000000  AX       0     0     4
  [14] .rodata           PROGBITS         0000000000a7aee0  00a7aee0
       00000000000821a0  0000000000000000   A       0     0     32
  [15] .nv_fatbin        PROGBITS         0000000000afd080  00afd080
       00000000061fdd70  0000000000000000   A       0     0     8
  [16] .eh_frame_hdr     PROGBITS         0000000006cfadf0  06cfadf0
       000000000000fb94  0000000000000000   A       0     0     4
  [17] .eh_frame         PROGBITS         0000000006d0a988  06d0a988
       000000000009c4b4  0000000000000000   A       0     0     8
  [18] .tbss             NOBITS           0000000006fa7000  06da7000
       0000000000001000  0000000000000000 WAT       0     0     4096
  [19] .init_array       INIT_ARRAY       0000000006fa7000  06da7000
       00000000000008b8  0000000000000000  WA       0     0     8
  [20] .fini_array       FINI_ARRAY       0000000006fa78b8  06da78b8
       0000000000000008  0000000000000000  WA       0     0     8
  [21] .jcr              PROGBITS         0000000006fa78c0  06da78c0
       0000000000000008  0000000000000000  WA       0     0     8
  [22] .data.rel.ro      PROGBITS         0000000006fa78e0  06da78e0
       0000000000003a00  0000000000000000  WA       0     0     32
  [23] .dynamic          DYNAMIC          0000000006fab2e0  06dab2e0
       00000000000002a0  0000000000000010  WA       4     0     8
  [24] .got              PROGBITS         0000000006fab580  06dab580
       0000000000000060  0000000000000008  WA       0     0     8
  [25] .got.plt          PROGBITS         0000000006fab5e0  06dab5e0
       0000000000000798  0000000000000008  WA       0     0     8
  [26] .data             PROGBITS         0000000006fabd80  06dabd80
       00000000000382f8  0000000000000000  WA       0     0     32
  [27] .nvFatBinSegment  PROGBITS         0000000006fe4078  06de4078
       0000000000001968  0000000000000000  WA       0     0     8
  [28] .bss              NOBITS           0000000006fe59e0  06de59e0
       00000000000021a0  0000000000000000  WA       0     0     32
  [29] .shstrtab         STRTAB           0000000000000000  06de59e0
       0000000000000117  0000000000000000           0     0     1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
  ```

### 1. **`.note.gnu.build-id` (Section 1)**

* **Type**: `NOTE`
* **用途**：这个段通常包含由构建工具生成的元数据信息，如唯一的构建 ID。它用于调试和验证目的。
* **Flags**: `A` 表示它会被加载到内存中。

### 2. **`.gnu.hash` (Section 2)**

* **Type**: `GNU_HASH`
* **用途**：存储哈希表，用于动态链接时加速符号查找。GNU hash 表通常比传统的 SysV hash 表效率更高。
* **Flags**: `A` 表示该段会加载到内存中。

### 3. **`.dynsym` (Section 3)**

* **Type**: `DYNSYM`
* **用途**：这是动态符号表，存储了动态链接时所需的符号信息（比如库中的函数和全局变量）。
* **Flags**: `A` 表示该段会被加载到内存中。

### 4. **`.dynstr` (Section 4)**

* **Type**: `STRTAB`
* **用途**：存储符号表中的字符串。`dynsym` 段中的符号名称会在这个段中存储。
* **Flags**: `A` 表示它会被加载到内存中。

### 5. **`.gnu.version` (Section 5)**

* **Type**: `VERSYM`
* **用途**：该段用于表示每个符号的版本。它与动态符号表一起使用，以实现更精确的符号版本控制。
* **Flags**: `A` 表示它会被加载到内存中。

### 6. **`.gnu.version_d` (Section 6)**

* **Type**: `VERDEF`
* **用途**：存储定义的符号版本，表示此 ELF 文件中定义了哪些符号版本。
* **Flags**: `A` 表示它会被加载到内存中。

### 7. **`.gnu.version_r` (Section 7)**

* **Type**: `VERNEED`
* **用途**：存储 ELF 文件所需的符号版本，表示此 ELF 文件依赖于其他库中的哪些符号版本。
* **Flags**: `A` 表示它会被加载到内存中。

### 8. **`.rela.dyn` (Section 8)**

* **Type**: `RELA`
* **用途**：存储动态重定位条目。重定位条目用于在加载时调整符号和数据的内存地址。
* **Flags**: `A` 表示它会被加载到内存中。

### 9. **`.rela.plt` (Section 9)**

* **Type**: `RELA`
* **用途**：用于 `PLT` (Procedure Linkage Table) 的重定位条目，处理延迟绑定函数的重定位。
* **Flags**: `A, I`，表示它会被加载到内存中，并且是可重定位的。

### 10. **`.init` (Section 10)**

* **Type**: `PROGBITS`
* **用途**：初始化段，存储程序初始化时需要执行的代码（通常是构造函数）。
* **Flags**: `AX`，表示它是可执行且会被加载到内存中的代码段。

### 11. **`.plt` (Section 11)**

* **Type**: `PROGBITS`
* **用途**：`PLT` 段用于延迟绑定函数调用，动态链接时，函数地址会存储在这里。
* **Flags**: `AX`，表示它是可执行且会被加载到内存中。

### 12. **`.text` (Section 12)**

* **Type**: `PROGBITS`
* **用途**：这是最重要的段之一，存储了程序的所有可执行代码。
* **Flags**: `AX`，表示它是可执行的代码段。

### 13. **`.fini` (Section 13)**

* **Type**: `PROGBITS`
* **用途**：程序结束时执行的代码（通常是析构函数）。
* **Flags**: `AX`，表示它是可执行段。

### 14. **`.rodata` (Section 14)**

* **Type**: `PROGBITS`
* **用途**：存储只读数据，比如常量和字符串字面值。
* **Flags**: `A`，表示该段会被加载到内存中，但只读。

### 15. **`.nv_fatbin` (Section 15)**

* **Type**: `PROGBITS`
* **用途**：这是 CUDA 程序中特有的段，存储了 GPU 代码（如 PTX 或 SASS）。CUDA 程序会将这些代码传递到 GPU 进行执行。
* **Flags**: `A`，表示它会被加载到内存中。

### 16. **`.eh_frame_hdr` (Section 16)**

* **Type**: `PROGBITS`
* **用途**：这是异常处理所需的段，存储异常处理时的栈展开信息。
* **Flags**: `A`，表示它会被加载到内存中。

### 17. **`.eh_frame` (Section 17)**

* **Type**: `PROGBITS`
* **用途**：实际存储栈展开的详细信息，配合 `.eh_frame_hdr` 使用，用于异常处理。
* **Flags**: `A`，表示它会被加载到内存中。

### 18. **`.tbss` (Section 18)**

* **Type**: `NOBITS`
* **用途**：用于存储线程局部存储的未初始化全局变量。
* **Flags**: `WAT`，表示该段会被加载到内存中，可写，并且是线程局部存储。

### 19. **`.init_array` (Section 19)**

* **Type**: `INIT_ARRAY`
* **用途**：存储全局或静态对象的构造函数指针。程序启动时会自动执行这些构造函数。
* **Flags**: `WA`，表示它是可写且会被加载到内存中。

### 20. **`.fini_array` (Section 20)**

* **Type**: `FINI_ARRAY`
* **用途**：存储全局或静态对象的析构函数指针。程序结束时会自动执行这些析构函数。
* **Flags**: `WA`，表示它是可写且会被加载到内存中。

### 21. **`.jcr` (Section 21)**

* **Type**: `PROGBITS`
* **用途**：存储用于 Java 异常处理的段，但在大多数非 Java 程序中不会使用。
* **Flags**: `WA`，表示它会被加载到内存中并且是可写的。

### 22. **`.data.rel.ro` (Section 22)**

* **Type**: `PROGBITS`
* **用途**：存储初始化后只读的数据，通常用于指针或全局变量的重定位。
* **Flags**: `WA`，表示它会被加载到内存中并且是可写的。

### 23. **`.dynamic` (Section 23)**

* **Type**: `DYNAMIC`
* **用途**：存储动态链接时需要的信息，比如动态库的依赖关系、符号解析等。
* **Flags**: `WA`，表示它是可写的并且会被加载到内存中。

### 24. **`.got` (Section 24)**

* **Type**: `PROGBITS`
* **用途**：`GOT`（Global Offset Table）用于存储全局变量和函数的地址，动态链接时会更新此表。
* **Flags**: `WA`，表示它是可写的并且会被加载到内存中。

### 25. **`.got.plt` (Section 25)**

* **Type**: `PROGBITS`
* **用途**：`PLT` 表中的全局偏移量表，动态链接时处理函数调用。
* **Flags**: `WA`，表示它是可写的并且会被加载到内存中。

### 26. **`.data` (Section 26)**

* **Type**: `PROGBITS`
* **用途**：存储全局可写数据（已初始化的全局变量）。
* **Flags**: `WA`，表示它是可写的并且会被加载到内存中。

### 27. **`.nvFatBinSegment` (Section 27)**

* **Type**: `PROGBITS`
* **用途**：这是 CUDA 特有的段，用于存储 fat binary（包含了多个架构的 GPU 代码）。
* **Flags**: `WA`，表示它是可写的并且会被加载到内存中。



### 28. **`.bss` (Section 28)**

* **Type**: `NOBITS`
* **用途**：存储未初始化的全局变量。这些变量在程序启动时会被初始化为 0。
* **Flags**: `WA`，表示它是可写的并且会被加载到内存中。

### 29. **`.shstrtab` (Section 29)**

* **Type**: `STRTAB`
* **用途**：存储段名称字符串。
* **Flags**: 无特别标志。




### GPU Kernel 所在的 Sections

对于 `cuBLAS` 以及其他包含 CUDA GPU kernel 的库，最重要的 sections 通常是：

1. **`.nv_fatbin`**
   * **用途**：这个 section 通常包含了 GPU kernel 代码。`nv_fatbin` 是 CUDA 的“fat binary”段，它用于存储多个架构的 CUDA PTX 代码和 SASS（设备特定的汇编语言）代码。
   * **内容**：在此 section 中，存放的是用于不同 GPU 架构的代码。它包含了一个或多个 CUDA kernel 编译后的二进制表示，可能是 PTX（并行线程执行）代码或直接的 SASS 机器码。每个 `cuBLAS` 函数如果包含调用 GPU kernel，其对应的 GPU 代码将存储在这个段中。
2. **`.nvFatBinSegment`**
   * **用途**：这是另一个 CUDA 专用的 section，它包含了和 `.nv_fatbin` 类似的内容，也是 GPU 加速代码的存储地，特别是针对不同 CUDA 版本或 GPU 架构的优化代码。
   * **内容**：这里也会存放 GPU 设备代码，并可能与 `.nv_fatbin` 类似存放了 fat binary。

### GPU Kernel 在这些 Sections 中的分布

1. **Fat Binary（.nv_fatbin）结构**：
   * CUDA 的 fat binary 允许在一个文件中包含多个目标架构的代码片段（如针对不同 CUDA 架构的代码）。这意味着，如果一个 CUDA 程序或库（如 cuBLAS）需要支持多种不同的 GPU 架构，它可以在 fat binary 中包含多个版本的相同 kernel，每个版本对应一种 GPU 架构（例如 Compute Capability 6.1, 7.0, 7.5 等）。
   * **分布方式**：通常这些 kernel 会以不同的片段存储在 `.nv_fatbin` 段中，每个片段包含针对特定 GPU 架构的优化代码。当程序运行时，CUDA Runtime 会选择最合适的代码片段在目标 GPU 上执行。
2. **Kernel 调用机制**：
   * 在执行时，cuBLAS 库会通过 CUDA 的 Runtime API 或 Driver API 加载这些存储在 `.nv_fatbin` 中的 kernel。每当调用某个 cuBLAS 函数时，它会将相关的 GPU kernel 代码从 fat binary 加载到 GPU 上运行，并根据需要分配 GPU 资源（如 block 和 thread）来执行矩阵或向量操作。


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

**3. cublas 版本**
``` bash
#define CUBLAS_VER_MAJOR 11
#define CUBLAS_VER_MINOR 3
#define CUBLAS_VER_PATCH 1

```
---
### Ref
1. https://docs.nvidia.com/cuda/cublas/
2. https://github.com/NVIDIA/cuda-samples?tab=readme-ov-file#cublas
3. https://blog.csdn.net/zcy0xy/article/details/84555053
4. https://nw.tsuda.ac.jp/lec/cuda/doc_v9_0/pdf/CUBLAS_Library.pdf
5. https://wuli.wiki/online/cublas.html
