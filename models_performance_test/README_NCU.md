
本项目在原项目的基础上增加了Nsight Compute(ncu)测试的功能，并对相关脚本功能做了一些健硕性的增强，同时，对一些框架的代码进行了更改（主要是数据集的大小和epoch等），增加模型性能测试的效率，同时完善了模型LSTM的有关功能。

### Overview
Nsight Compute (NCU) 是 NVIDIA 提供的 GPU 内核级性能分析工具，专注于 CUDA 程序的优化。它提供详细的计算资源、内存带宽、指令调度等性能数据，适合分析单个或多个 CUDA kernel 的瓶颈。Nsight Systems (NSYS) 则是系统级性能分析工具，关注 GPU、CPU、内存和 I/O 的协同工作，通过时间线展示应用程序的整体性能。两者的分析范围和用途不同：NCU 专注于内核级优化，适合 CUDA 开发者深入分析；NSYS 提供应用级性能概览，适合定位 CPU-GPU 协作的效率问题。实际应用中可以先用 NSYS 确定性能瓶颈，再用 NCU 深入优化 GPU 内核，充分提升程序性能。

### 1.实验环境：

```bash
Ubuntu 22.04.5 LTS
NVIDIA RTX A6000
cuda 12.4
nsys 2024.5.1
ncu 2024.3.2
docker 27.2.1
python 3.10.12
pytorch 2.5.1
```

### 2.docker启动命令

```bash
docker run --rm --gpus all --network host --shm-size=4g -it \
    --cap-add=SYS_ADMIN \
    -v /usr/local/cuda-12.4:/usr/local/cuda-12.4 \
    -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    -v /opt/nvidia/nsight-systems/2024.5.1:/opt/nvidia/nsight-systems/2024.5.1 \
    -v /opt/nvidia/nsight-compute/2024.3.2:/opt/nvidia/nsight-compute/2024.3.2 \
    -e PATH="/opt/nvidia/nsight-systems/2024.5.1/bin:/opt/nvidia/nsight-compute/2024.3.2/bin:/usr/local/cuda-12.4/bin:$PATH" \
    models_test:v2

```
其中，cuda、nsys、ncu仍然使用宿主机的工具包，`--cap-add=SYS_ADMIN`命令是为了提高使用ncu的权限，否则后续无法进行性能测试。

### 3.使用指南
本项目已将相关命令撰写为bash脚本，位于`/workspace/command/`目录下，其中-nsys和-ncu表示使用两种不同的工具进行性能测试的脚本。start表示执行单个模型的性能测试，start-all表示执行全部模型的性能测试。clean脚本用于删除性能测试产生的文件。具体使用方法如下：
**单个模型测试模式**：以MobienetV2为例
进入该模型目录下(包含 train.py 和 inference.py)
1. `start-ncu <arg>`:使用ncu进行性能测试，<arg>参数可选：train|inference|train inference三种，分别表示训练、推理、先训练再推理性能测试,或者`bash /workspace/command/start_ncu.sh <arg>`。例如：`start-ncu train`
2. `start-nsys <arg>`:使用nsys进行性能测试，<arg>参数可选：train|inference|train inference三种，分别表示训练、推理、先训练再推理性能测试,或者`bash /workspace/command/start_nsys.sh <arg>`
3. clean +y(Y):删除当前目录下所有后缀为 `.csv, .pth, .sqlite, and .nsys-rep and .ncu-rep`的文件
**批量测试模式**
确保位于`/workspace/models/`目录下(子目录为各个模型)
1. `start-all-ncu`
2. `start-all-nsys`

由于模型较多,因此不建议使用批量测试模式。并且相对于nsys来说，ncu测试性能时间更长，测试实验中实际执行的命令是`ncu -o train --section LaunchStats --target-processes application-only --replay-mode kernel python train.py`，并且对模型训练的数据集进行了调整。
如果需要扩展更多的功能，请根据需求更改command中相应脚本的内容，以及调整模型的架构或者数据来适应实际项目的需要。

相关代码已上传至：https://github.com/Dylan887/CUDA/tree/main/models_performance_test

### 4. 附：ncu相关命令详解
部分参数根据不同的版本有不同的用法。细节参看`ncu --help`
```bash
ncu --set full --export train.ncu-rep --target-processes all --replay-mode application python train.py
```
1. ```--set full```
用于收集 完整的性能数据集。这个选项会启用 Nsight Compute 提供的所有可用的 sections 和 metrics，对目标程序进行全面的性能分析。

2. ```--export train.ncu-rep```
表示将性能报告导出为train.ncu-rep，也可以写成`-o train`

3. `--target-processes all`
**all** 表示分析所有与目标程序相关的进程（包括主进程及其子进程）

4. `--target-processes`全部的模式如下：

| 模式           | 说明                                                                 |
|----------------|----------------------------------------------------------------------|
| **all**        | 捕获所有与目标程序相关的进程，包括主进程和所有派生的子进程。          |
| **application-only**| 只捕获有关application的进程。    |

5. `--replay-mode application`:重放模式

| 模式         | 说明                                                                                          |
|--------------|-----------------------------------------------------------------------------------------------|
| **application** | 程序完整运行，捕获所有 CUDA kernel 的性能数据，适用于全面分析场景。                             |
| **kernel**      | 每个 CUDA kernel 在程序运行结束后会被单独重放，用于收集更详细的性能指标，可能需要多次运行程序。   |


6. `python train.py`:执行的python程序

**如果直接采取上述命令的话，采集所有的section，时间会变得非常长。例如对模型1.MObienetV2的训练部分进行测试，笔者进行将近1个小时测试之后还没有成功。此时，可以选择下述的命令。执行检测部分section：**

* 在运行以下指令时ncu只会根据section的设置去捕获Kernel在内存利用方面的metrics
```bash
ncu -o train -f --section "regex:MemoryWorkloadAnalysis(_Chart|_Tables)?"  \
--target-processes all --replay-mode application python train.py
# -f表示强制覆盖已有的输出文件，不询问用户
# --section "regex:MemoryWorkloadAnalysis(_Chart|_Tables)?"表示匹配 `MemoryWorkloadAnalysis` 后跟 `_Chart` 或 `_Tables` 的部分, `?` 表示 `_Chart` 和 `_Tables` 是可选的。

```

* 在生成train.ncu-rep报告后，让ncu重新读入并把详细信息打印成csv格式的文件并存储

```ncu -i train.ncu-rep --page details --csv --log-file train.csv```

* 以下的指令中ncu会profiling编号为0的device上的所有因为用户「command」产生的子进程中名称为python的进程，并只会profiling这些进程中不以「nc」开头的Kernel函数

```bash
ncu -o train -f  --replay-mode application --target-processes all \
--section "regex:MemoryWorkloadAnalysis(_Chart|_Tables)?"\
--kernel-name "regex:^[^n][^c].*" --device 0 \
--target-processes-filter python train.py
```
---
**section选项详解：**

| Identifier and Filename           | Description                                                                                          |
|-----------------------------------|------------------------------------------------------------------------------------------------------|
| ComputeWorkloadAnalysis           | Detailed analysis of the compute resources of the streaming multiprocessors (SM), including the achieved instructions per clock (IPC) and the utilization of each available pipeline. Pipelines with very high utilization might limit the overall performance.<br>**详细分析流多处理器 (SM) 的计算资源，包括每周期指令数 (IPC) 和每个可用管道的利用率。利用率过高的管道可能限制整体性能。** |
| InstructionStats                  | Statistics of the executed low-level assembly instructions (SASS). The instruction mix provides insight into the types and frequency of the executed instructions. A narrow mix of instruction types implies a dependency on few instruction pipelines, while others remain unused. Using multiple pipelines allows hiding latencies and enables parallel execution.<br>**统计低级汇编指令 (SASS) 的执行情况。指令类型的分布揭示了指令的种类和频率。狭窄的指令分布可能导致对少数指令管道的依赖，而其他管道未被利用。使用多个管道可以隐藏延迟并实现并行执行。** |
| LaunchStats                      | Summary of the configuration used to launch the kernel. The launch configuration defines the size of the kernel grid, the division of the grid into blocks, and the GPU resources needed to execute the kernel. Choosing an efficient launch configuration maximizes device utilization.<br>**总结内核启动时的配置，包括网格大小、块划分和 GPU 资源分配。选择高效的启动配置可以最大化设备利用率。** |
| MemoryWorkloadAnalysis           | Detailed analysis of the memory resources of the GPU. Memory can become a limiting factor for the overall kernel performance when fully utilizing the involved hardware units (Mem Busy), exhausting the available communication bandwidth between those units (Max Bandwidth), or by reaching the maximum throughput of issuing memory instructions (Mem Pipes Busy). Depending on the limiting factor, the memory chart and tables allow to identify the exact bottleneck in the memory system.<br>**详细分析 GPU 的内存资源。当相关硬件单元利用率高、通信带宽耗尽或内存指令吞吐量达到上限时，内存可能成为性能瓶颈。通过图表和表格可以确定内存系统的具体瓶颈。** |
| NUMA Affinity                    | Non-uniform memory access (NUMA) affinities based on compute and memory distances for all GPUs.<br>**基于计算和内存距离分析所有 GPU 的非均匀内存访问 (NUMA) 亲和性。** |
| Nvlink                           | High-level summary of NVLink utilization. It shows the total received and transmitted (sent) memory, as well as the overall link peak utilization.<br>**高层概述 NVLink 的利用情况，包括接收和发送的总内存以及链接的峰值利用率。** |
| Nvlink_Tables                    | Detailed tables with properties for each NVLink.<br>**显示每个 NVLink 的详细属性表。** |
| Nvlink_Topology                  | NVLink Topology diagram shows logical NVLink connections with transmit/receive throughput.<br>**展示 NVLink 拓扑结构图，显示逻辑连接以及传输/接收吞吐量。** |
| Occupancy                        | Occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps. Another way to view occupancy is the percentage of the hardware’s ability to process warps that is actively in use.<br>**占用率是每个多处理器上活动 warp 数与可能最大 warp 数的比率。它反映了硬件处理 warp 的能力使用率。** |
| PM Sampling                      | Timeline view of metrics sampled periodically over the workload duration. Data is collected across multiple passes.<br>**按时间轴定期采样工作负载持续时间内的指标，数据通过多次采集完成。** |
| SchedulerStats                   | Summary of the activity of the schedulers issuing instructions. Each scheduler maintains a pool of warps that it can issue instructions for. On every cycle each scheduler checks the state of the allocated warps in the pool.<br>**总结指令调度器的活动。每个调度器维护一个 warp 池，并在每个周期检查 warp 的状态。** |
| SourceCounters                   | Source metrics, including branch efficiency and sampled warp stall reasons.<br>**提供源指标，包括分支效率和 warp 停滞原因的采样数据。** |
| SpeedOfLight                     | High-level overview of the throughput for compute and memory resources of the GPU.<br>**概述 GPU 的计算和内存资源的吞吐量，显示理论最大值的利用率百分比。** |
| WarpStateStats                   | Analysis of the states in which all warps spent cycles during the kernel execution.<br>**分析内核执行期间所有 warp 所处的状态及其周期分布。** |

---

上述资料参考：
1. https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead
2. https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules
3. https://www.androsheep.win/post/ncu/
4. https://github.com/pytorch/data?tab=readme-ov-file#installation