# cuda二进制文件的section介绍

在编译CUDA程序（如矩阵乘法示例）并生成二进制可执行文件后，可以使用工具如 `objdump` 和 `readelf` 来查看生成的可执行文件中的各个**section**。本文以matrixMul代码示例说明介绍相关的section。

### matrixMul代码示例
```c++
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
### 1. **将cuda程序编译成二进制文件**

```bash
 nvcc matrixMul.cu -o matrixMul
 ```
 将matrixMul.cu编译为二进制文件matrixMul

### 2. 使用 `objdump` 和 `readelf` 查看二进制文件中的section

#### 使用 `objdump`：

`objdump`是一个多功能的工具，用于反汇编和查看二进制文件的内容。它可以提供更多高级的功能，如反汇编、显示符号表、显示各个section的数据等。objdump 更侧重于显示二进制文件的内容和结构，适合更广泛的二进制分析任务，包括反汇编CPU指令和查看特定数据段。
* 典型用途：反汇编代码、查看section内容、分析符号表、分析跳转表、显示文件头等。

```bash
objdump -h matrixMul
```

这个命令会输出文件中的各个section，显示其名称、大小和位置等信息。
```bash
matrixMul:     file format elf64-x86-64

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  0 .interp       0000001c  0000000000000270  0000000000000270  00000270  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  1 .note.ABI-tag 00000020  000000000000028c  000000000000028c  0000028c  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  2 .note.gnu.build-id 00000024  00000000000002ac  00000000000002ac  000002ac  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  3 .gnu.hash     0000001c  00000000000002d0  00000000000002d0  000002d0  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  4 .dynsym       00000f60  00000000000002f0  00000000000002f0  000002f0  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  5 .dynstr       000007eb  0000000000001250  0000000000001250  00001250  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  6 .gnu.version  00000148  0000000000001a3c  0000000000001a3c  00001a3c  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  7 .gnu.version_r 00000130  0000000000001b88  0000000000001b88  00001b88  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  8 .rela.dyn     00003e58  0000000000001cb8  0000000000001cb8  00001cb8  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  9 .rela.plt     00000eb8  0000000000005b10  0000000000005b10  00005b10  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 10 .init         00000017  00000000000069c8  00000000000069c8  000069c8  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 11 .plt          000009e0  00000000000069e0  00000000000069e0  000069e0  2**4
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 12 .plt.got      00000008  00000000000073c0  00000000000073c0  000073c0  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 13 .text         0005f24e  00000000000073d0  00000000000073d0  000073d0  2**4
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 14 .fini         00000009  0000000000066620  0000000000066620  00066620  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 15 .rodata       0000a6cc  0000000000066640  0000000000066640  00066640  2**5
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 16 .nv_fatbin    00001c78  0000000000070d10  0000000000070d10  00070d10  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 17 __nv_module_id 0000000f  0000000000072988  0000000000072988  00072988  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 18 .eh_frame_hdr 00002efc  0000000000072998  0000000000072998  00072998  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 19 .eh_frame     000137a0  0000000000075898  0000000000075898  00075898  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 20 .gcc_except_table 00000010  0000000000089038  0000000000089038  00089038  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 21 .tbss         00001000  000000000028a000  000000000028a000  0008a000  2**12
                  ALLOC, THREAD_LOCAL
 22 .init_array   00000020  000000000028a000  000000000028a000  0008a000  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 23 .fini_array   00000008  000000000028a020  000000000028a020  0008a020  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 24 .data.rel.ro  00003180  000000000028a040  000000000028a040  0008a040  2**5
                  CONTENTS, ALLOC, LOAD, DATA
 25 .dynamic      00000250  000000000028d1c0  000000000028d1c0  0008d1c0  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 26 .got          00000528  000000000028d410  000000000028d410  0008d410  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 27 .data         00000058  000000000028e000  000000000028e000  0008e000  2**5
                  CONTENTS, ALLOC, LOAD, DATA
 28 .nvFatBinSegment 00000030  000000000028e058  000000000028e058  0008e058  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 29 .bss          00000d60  000000000028e0a0  000000000028e0a0  0008e088  2**5
                  ALLOC
 30 .comment      00000055  0000000000000000  0000000000000000  0008e088  2**0
                  CONTENTS, READONLY

```

#### 使用 `readelf`：

`readelf` 是另一个可以分析二进制文件的工具，特别是ELF（Executable and Linkable Format）格式的文件。可以使用如下命令查看section：

```bash
readelf -S matrixMul
```
这将显示可执行文件中所有的section列表，包括名称、大小、地址和类型。
**输出结果**
```bash
There are 35 section headers, starting at offset 0xaa3d8:

Section Headers:
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  [ 0]                   NULL             0000000000000000  00000000
       0000000000000000  0000000000000000           0     0     0
  [ 1] .interp           PROGBITS         0000000000000270  00000270
       000000000000001c  0000000000000000   A       0     0     1
  [ 2] .note.ABI-tag     NOTE             000000000000028c  0000028c
       0000000000000020  0000000000000000   A       0     0     4
  [ 3] .note.gnu.build-i NOTE             00000000000002ac  000002ac
       0000000000000024  0000000000000000   A       0     0     4
  [ 4] .gnu.hash         GNU_HASH         00000000000002d0  000002d0
       000000000000001c  0000000000000000   A       5     0     8
  [ 5] .dynsym           DYNSYM           00000000000002f0  000002f0
       0000000000000f60  0000000000000018   A       6     1     8
  [ 6] .dynstr           STRTAB           0000000000001250  00001250
       00000000000007eb  0000000000000000   A       0     0     1
  [ 7] .gnu.version      VERSYM           0000000000001a3c  00001a3c
       0000000000000148  0000000000000002   A       5     0     2
  [ 8] .gnu.version_r    VERNEED          0000000000001b88  00001b88
       0000000000000130  0000000000000000   A       6     7     8
  [ 9] .rela.dyn         RELA             0000000000001cb8  00001cb8
       0000000000003e58  0000000000000018   A       5     0     8
  [10] .rela.plt         RELA             0000000000005b10  00005b10
       0000000000000eb8  0000000000000018  AI       5    27     8
  [11] .init             PROGBITS         00000000000069c8  000069c8
       0000000000000017  0000000000000000  AX       0     0     4
  [12] .plt              PROGBITS         00000000000069e0  000069e0
       00000000000009e0  0000000000000010  AX       0     0     16
  [13] .plt.got          PROGBITS         00000000000073c0  000073c0
       0000000000000008  0000000000000008  AX       0     0     8
  [14] .text             PROGBITS         00000000000073d0  000073d0
       000000000005f24e  0000000000000000  AX       0     0     16
  [15] .fini             PROGBITS         0000000000066620  00066620
       0000000000000009  0000000000000000  AX       0     0     4
  [16] .rodata           PROGBITS         0000000000066640  00066640
       000000000000a6cc  0000000000000000   A       0     0     32
  [17] .nv_fatbin        PROGBITS         0000000000070d10  00070d10
       0000000000001c78  0000000000000000   A       0     0     8
  [18] __nv_module_id    PROGBITS         0000000000072988  00072988
       000000000000000f  0000000000000000   A       0     0     8
  [19] .eh_frame_hdr     PROGBITS         0000000000072998  00072998
       0000000000002efc  0000000000000000   A       0     0     4
  [20] .eh_frame         PROGBITS         0000000000075898  00075898
       00000000000137a0  0000000000000000   A       0     0     8
  [21] .gcc_except_table PROGBITS         0000000000089038  00089038
       0000000000000010  0000000000000000   A       0     0     1
  [22] .tbss             NOBITS           000000000028a000  0008a000
       0000000000001000  0000000000000000 WAT       0     0     4096
  [23] .init_array       INIT_ARRAY       000000000028a000  0008a000
       0000000000000020  0000000000000008  WA       0     0     8
  [24] .fini_array       FINI_ARRAY       000000000028a020  0008a020
       0000000000000008  0000000000000008  WA       0     0     8
  [25] .data.rel.ro      PROGBITS         000000000028a040  0008a040
       0000000000003180  0000000000000000  WA       0     0     32
  [26] .dynamic          DYNAMIC          000000000028d1c0  0008d1c0
       0000000000000250  0000000000000010  WA       6     0     8
  [27] .got              PROGBITS         000000000028d410  0008d410
       0000000000000528  0000000000000008  WA       0     0     8
  [28] .data             PROGBITS         000000000028e000  0008e000
       0000000000000058  0000000000000000  WA       0     0     32
  [29] .nvFatBinSegment  PROGBITS         000000000028e058  0008e058
       0000000000000030  0000000000000000  WA       0     0     8
  [30] .bss              NOBITS           000000000028e0a0  0008e088
       0000000000000d60  0000000000000000  WA       0     0     32
  [31] .comment          PROGBITS         0000000000000000  0008e088
       0000000000000055  0000000000000001  MS       0     0     1
  [32] .symtab           SYMTAB           0000000000000000  0008e0e0
       00000000000116e8  0000000000000018          33   1832     8
  [33] .strtab           STRTAB           0000000000000000  0009f7c8
       000000000000aac1  0000000000000000           0     0     1
  [34] .shstrtab         STRTAB           0000000000000000  000aa289
       000000000000014e  0000000000000000           0     0     1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
```
#### 各个Section的解释(以objdump输出结果为例)

#### 1. **.interp** (Section 0)

* **作用**：该section存储的是动态链接器（dynamic linker）的路径，它告诉操作系统在加载可执行文件时应该使用哪个动态链接器。这对于加载共享库（如CUDA的运行时库）至关重要。
* **内容**：文件中包含类似 `/lib64/ld-linux-x86-64.so.2` 的字符串，指向动态链接器的位置。

#### 2. **.note.ABI-tag** (Section 1)

* **作用**：该section通常包含关于系统的ABI（应用程序二进制接口）兼容性的信息，用于描述程序的构建环境。
* **内容**：通常与GNU/Linux系统相关，表示此程序需要Linux内核的特定版本才能运行。

#### 3. **.note.gnu.build-id** (Section 2)

* **作用**：该section存储程序的唯一标识符（Build ID），用于调试和版本控制。
* **内容**：唯一的哈希值，用于标识此可执行文件的构建版本。

#### 4. **.gnu.hash** (Section 3)

* **作用**：`GNU hash` 表存储了符号的哈希值，用于动态链接时的符号解析，以加快符号查找的速度。
* **内容**：该表优化了动态链接过程中对库函数符号的查找。

#### 5. **.dynsym** (Section 4)

* **作用**：动态符号表，存储程序中所有动态符号（如函数、全局变量等）。主机端调用CUDA库函数时，符号表用于查找符号的地址。
* **内容**：包含程序中所有需要动态链接的符号信息，如 `cudaMalloc`、`cudaMemcpy` 等。

#### 6. **.dynstr** (Section 5)

* **作用**：动态字符串表，配合动态符号表使用，存储符号的名称。
* **内容**：与 `dynsym` 一一对应，存储每个符号的字符串名称，如函数名、变量名等。

#### 7. **.gnu.version** 和 **.gnu.version_r** (Sections 6, 7)

* **作用**：这两个sections存储了符号版本控制信息，确保在动态链接时使用正确版本的符号。
* **内容**：这些表用于支持版本化的符号解析，确保动态链接库中使用正确的符号版本。

#### 8. **.rela.dyn** (Section 8)

* **作用**：重定位表，存储需要在加载时修正的地址信息，用于动态链接库和共享对象的重定位。
* **内容**：当程序加载到内存时，操作系统根据此表修正符号地址。

#### 9. **.rela.plt** (Section 9)

* **作用**：存储用于PLT（Procedure Linkage Table）表的重定位信息，支持动态链接的函数调用。
* **内容**：该section用于动态链接函数调用的间接跳转，类似于 `plt` 段的扩展。

#### 10. **.init** (Section 10)

* **作用**：存储初始化函数的指令，程序开始执行时首先调用该段中的代码，通常用于初始化全局变量或设置程序环境。
* **内容**：初始化代码，通常是一些在程序开始时需要执行的初始化逻辑。

#### 11. **.plt** (Section 11)

* **作用**：PLT（Procedure Linkage Table）存储主机代码中需要调用动态链接库的函数的跳转表。CUDA程序中，当调用 `cudaMalloc`、`cudaMemcpy` 等CUDA运行时库函数时，会通过PLT进行间接跳转。
* **内容**：PLT段中的跳转指针指向实际的函数地址，允许主机代码动态调用外部库函数。

#### 12. **.plt.got** (Section 12)

* **作用**：PLT与GOT（Global Offset Table）配合使用，处理动态链接时函数地址的存储和加载。
* **内容**：GOT表中存储了动态库函数的实际地址，而PLT段通过间接跳转调用这些函数。

#### 13. **.text** (Section 13)

* **作用**：`.text` 段存储主机代码的实际执行指令，是主机端程序的核心部分。CUDA程序的主机端代码，包括内存管理、数据传输、内核调用等，都存储在此段中。
* **内容**：主机代码的机器指令，可通过 `objdump -d` 反汇编查看其内容。

#### 14. **.fini** (Section 14)

* **作用**：存储程序结束时需要执行的代码，通常用于清理和释放资源。
* **内容**：终止代码，用于在程序结束时执行一些必要的操作。

#### 15. **.rodata** (Section 15)

* **作用**：只读数据段，存储程序中的静态常量和只读数据。CUDA程序中的常量内存（`__constant__` 修饰的变量）可能被存储在此段中。
* **内容**：存储只读的数据，如常量数组、字符串等。

#### 16. **.nv_fatbin** (Section 16)

* **作用**：该段是CUDA程序中存储设备代码的胖二进制文件（`fatbin`）。`fatbin` 包含了多个架构版本的设备代码，CUDA运行时会根据实际硬件选择合适的设备代码来执行。所有GPU内核的二进制代码都会被打包在此段中。
* **内容**：设备端的GPU内核函数和其他相关设备代码。通过 `cuobjdump` 可以进一步解析此段内容。

#### 17. **__nv_module_id** (Section 17)

* **作用**：这个section存储CUDA模块的唯一标识符，用于区分和管理不同的CUDA模块。
* **内容**：模块的ID，用于管理CUDA设备代码和模块。

#### 18. **.eh_frame_hdr** 和 **.eh_frame** (Sections 18, 19)

* **作用**：这两个段用于存储异常处理和栈展开（stack unwinding）信息。对于主机端C++代码，尤其是在使用异常处理时，这些段是必要的。
* **内容**：异常处理和栈展开信息，帮助调试器或运行时系统在出现异常时进行栈回溯。

#### 19. **.gcc_except_table** (Section 20)

* **作用**：该section存储C++异常处理机制中的异常表，帮助异常处理逻辑正确运行。
* **内容**：异常处理相关的数据。

#### 20. **.tbss** (Section 21)

* **作用**：存储线程局部存储（thread-local storage）的未初始化变量，分配给每个线程独立的存储空间。
* **内容**：线程局部存储变量。

#### 21. **.init_array** 和 **.fini_array** (Sections 22, 23)

* **作用**：存储初始化和终止函数的指针数组，在程序开始时调用初始化函数，在程序结束时调用终止函数。
* **内容**：初始化和终止函数的地址数组。

#### 22. **.data.rel.ro** (Section 24)

* **作用**：存储已初始化的只读全局数据，通常是指针等需要在运行时修正地址的只读数据。
* **内容**：运行时只读数据。

#### 23. **.dynamic** (Section 25)

* **作用**：动态链接信息，描述可执行文件或共享对象中的动态链接信息，包括依赖的库、符号和重定位表等。
* **内容**：用于动态链接库的信息，帮助操作系统加载共享库。

#### 24. **.got** (Section 26)

* **作用**：GOT（Global Offset Table）存储动态链接函数和变量的实际地址。
* **内容**：指向动态链接函数和变量的指针表。

#### 25. **.data** (Section 27)

* **作用**：已初始化的全局变量和静态变量存储在此段。CUDA程序主机端的全局和静态变量会存放在此段中。
* **内容**：已初始化的全局变量和静态变量。

#### 26. **.nvFatBinSegment** (Section 28)

* **作用**：该段与 `.nv_fatbin` 配合使用，存储设备代码的元数据或注册信息，帮助CUDA运行时调度设备代码。
* **内容**：设备代码的辅助信息或元数据。

#### 27. **.bss** (Section 29)

* **作用**：存储未初始化的全局变量和静态变量。这些变量在程序启动时会被初始化为零，但不会占用磁盘空间。
* **内容**：未初始化的全局变量和静态变量。

#### 28. **.comment** (Section 30)

* **作用**：该段通常存储编译器的版本信息或其他注释性信息。
* **内容**：编译器版本或注释。


这些section共同构成了CUDA程序中的主机代码和设备代码的结构。
### 3. CUDA程序中关键的section及其作用

在使用 `objdump` 或 `readelf` 查看完二进制文件的section后，以下是一些CUDA程序中重要的section及其作用：

#### 1. **`.text` section**：代码段

* **作用**：这是二进制文件中存储**主机代码**和**设备代码**的主要部分，包含CPU和GPU代码的实际指令。
* **内核分布**：通常，CUDA内核函数（kernel）的GPU代码会被嵌入在 `.text` 段中，特别是在 `fatbin` 文件中被嵌入后。`readelf` 或 `objdump` 的输出会显示 `.text` section 的地址和大小，你可以进一步查看其中包含的CUDA内核函数的符号。
* **查看内容**：可以使用 `objdump -d matrixMul` 来反汇编 `.text` section 中的代码，查找特定的CUDA内核函数。你会发现这些内核函数名称前通常会带有类似 `__cuda_` 前缀。

#### 2. **`.nv_fatbin` section**：胖二进制段

* **作用**：这是CUDA设备代码的关键段，存储**fatbinary**，即CUDA的胖二进制文件（包含针对不同GPU架构编译的设备代码）。这个段包含了GPU设备代码以及支持不同架构的多版本代码。
* **内核分布**：内核代码作为`fatbin`文件的一部分会存储在这个section中。由于 `fatbin` 文件可以包含多种GPU架构的代码，所有这些内核代码都会被嵌入在 `.nv_fatbin` 段中。通过进一步分析这个段的内容，你可以找到所有与内核相关的设备代码。
* **查看内容**：可以使用 `objdump -s -j .nv_fatbin matrixMul` 来查看 `fatbin` 文件的内容。

#### 3. **`.rodata` section**：只读数据段

* **作用**：存储只读数据。CUDA程序中常见的**常量内存**（`__constant__` 修饰的变量）可能会存储在 `.rodata` section 中。
* **查看内容**：使用 `objdump -s -j .rodata matrixMul` 可以查看这个段的内容，常量内存中的数据可能会存储在这里。

#### 4. **`.bss` section**：未初始化数据段

* **作用**：这个段存储全局的未初始化数据。在CUDA程序中，主机端的全局变量可能会出现在这个段中，设备端的全局数据通常不会在这里。
* **查看内容**：使用 `objdump -s -j .bss matrixMul` 可以查看未初始化的全局数据。

#### 5. **`.data` section**：初始化数据段

* **作用**：存储已经初始化的全局和静态数据。在CUDA程序中，主机代码的全局变量或静态变量可能会存储在这个段中。
* **查看内容**：可以使用 `objdump -s -j .data matrixMul` 来查看这个段的内容。

#### 6. **`.debug` section**（可选）：

* **作用**：如果在调试模式下编译（使用 `-g` 选项），这个段会包含调试符号。它可以帮助调试工具（如 `gdb` 或 `cuda-gdb`）识别主机和设备代码中的符号和变量信息。
* **查看内容**：如果生成了调试符号，可以通过 `readelf -S matrixMul` 查看 `.debug` section，找到内核函数和符号。

### 4. **GPU内核在 `.text` 和 `.nv_fatbin` 中的分布**

* **.text 中的内核**： 内核函数的入口通常会在 `.text` 段中，特别是主机代码需要通过函数指针调用设备端内核时，可能会在 `.text` 段中找到入口点。通过反汇编 `.text` 段，通常可以找到设备内核的符号。
* **.nv_fatbin 中的内核**： 真正的GPU内核二进制代码一般都嵌入在 `.nv_fatbin` 段中。这个段包含了不同架构的PTX或者CUBIN代码，经过 `fatbinary` 工具打包后保存于此。在运行时，CUDA驱动会根据实际运行时的GPU架构，从 `.nv_fatbin` 段中提取对应的内核代码并将其加载到GPU上。


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
---
### Ref
1. https://stackoverflow.com/questions/8979664/readelf-vs-objdump-why-are-both-needed
2. https://www.gnu.org/software/binutils/
3. https://stackoverflow.com/questions/20514587/text0x20-undefined-reference-to-main-and-undefined-reference-to-function
4. https://ftp.gnu.org/old-gnu/Manuals/binutils-2.12/html_node/binutils_16.html
   




