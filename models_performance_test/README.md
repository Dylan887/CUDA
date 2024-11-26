
本项目是为使用nsight system 测试多个深度学习模型性能设立。

### 1.基本环境
```bash
Ubuntu 22.04.5 LTS
NVIDIA RTX A6000
cuda 12.4
nsys 2024.5.1
docker 27.2.1
python 3.10.12
pytorch 2.51
```
### 2.使用
**docker 构建**
```bash
FROM ubuntu:22.04
WORKDIR /workspace
# 安装 Python 3.10 和常用工具
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 升级 Python 包管理工具 pip
RUN python3.10 -m pip install --upgrade pip
CMD ["bash"]
```
其中部分依赖依据相关代码自行下载。

**docker 启动**：
```bash
docker run --rm --gpus all --network host --shm-size=4g -it \
    -v /usr/local/cuda-12.4:/usr/local/cuda-12.4 \
    -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    -v /opt/nvidia/nsight-systems/2024.5.1:/opt/nvidia/nsight-systems/2024.5.1 \
    -e PATH="/opt/nvidia/nsight-systems/2024.5.1/bin:/usr/local/cuda-12.4/bin:$PATH" \
    model_test:v3
```

其中：
`--shm-size=4g` ： 如果不设置共享内存，性能测试时会出现共享内存不足的问题。
容器使用的是宿主机的cuda12.4和nsys，其中系统默认cuda12.4 使用的nsys是2023版本的，该版本下，某些模型性能测试会报错，不建议使用旧版本.
版本相关问题解答：
<https://forums.developer.nvidia.com/t/nsys-importation-error/283231/9>。

`--network host` 网络可以采用宿主机的或者默认情况下的bridge.

**镜像文件目录**

`modles`: 包括模型代码(train,inference),以及数据data,预权重pre_weights

`command`: bash文件，包括start.sh,start_all.sh,clean.sh
如果想启动单个模型性能测试，确保进入该模型的目录下，执行
```bash
start <param> 
```
param可以是train ,inference,或者 train inference,表示对训练/推理/训练推理进行性能测试

如果需要追踪更多信息，尝试
```bash
nsys profile --trace=cuda,cudnn,nvxt,osrt --output=<train> python <train.py>
```
与此同时可以更改start.sh该部分内容。

如果需要某个指标的信息，尝试
```bash
nsys stats <inference.nsys-rep> -r <cuda_api_gpu_sum> --format csv --output <inference>
```
如果想删除当前模型下的性能测试文件，尝试
```bash
clean + y(Y)
```
表示删除当前目录下所有 .csv .pth .nsys-rep .splite文件

如果想批量启动所有的模型测试，尝试
```bash
start-all
```
会依次执行所有模型的性能测试(包括训练和推理)。

但不建议批量启动，有时候会导致cuda设备内存不足或出现其他错误等。

准备的代码只是为了进行简单的测试，某些网络结构以及数据集等需自行调整。
