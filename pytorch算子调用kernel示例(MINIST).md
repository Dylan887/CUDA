# pytorch 算子调用kernel示例(MINIST)
当进行 MNIST 分类任务时，PyTorch 中的每一个算子会根据设备类型（CPU 或 CUDA）自动选择合适的内核（kernel）进行计算。本文以GPU为例，介绍算子调用kernel的过程。

### 1. **模型定义与前向传播**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 卷积层1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 卷积层2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, 10)  # 全连接层2
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 激活函数 ReLU
        x = F.max_pool2d(x, 2)  # 最大池化
        x = F.relu(self.conv2(x))  # 激活函数 ReLU
        x = F.max_pool2d(x, 2)  # 最大池化
        x = x.view(-1, 64 * 7 * 7)  # 张量展平
        x = F.relu(self.fc1(x))  # 全连接层激活
        x = self.fc2(x)  # 输出层
        return F.log_softmax(x, dim=1)  # 计算 softmax 概率
```

### 2. **数据加载与模型放置到 GPU**

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 将模型和数据移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)  # 模型加载到 GPU
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练过程
for epoch in range(1, 2):  # 运行1个 epoch
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # 数据加载到 GPU
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
```

### 3. **算子调用与 GPU 内核函数**

当模型和数据都在 GPU 上时，PyTorch 的每一个算子会通过其调度机制（dispatch mechanism）调用相应的 CUDA 内核来加速计算。

#### **(1) Conv2d (卷积层) 的 GPU 调用**

在卷积操作中，PyTorch 会调用 `Conv2d` 算子。此时，设备被指定为 CUDA，调度系统会调用 GPU 专用的内核。

* **算子调用**:

  ```python
  x = F.relu(self.conv1(x))
  ```



* **内核调用**：

  * 对于 CUDA：调用 `conv2d_cuda_kernel`。

* **CUDA 实现的原型代码**： `conv2d_cuda_kernel` 使用高度优化的 cuDNN 库来执行卷积操作，具体原型可能如下：

```c++
Tensor conv2d_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
    
    // 使用 cuDNN 卷积函数来加速计算
    return at::cudnn_convolution(input, weight, bias, padding, stride, dilation, groups);
}
```
**at::cudnn_convolution 是一个封装好的接口，用于调用 cuDNN 库中的卷积操作。cuDNN 提供了一系列高度优化的卷积算法，能够根据输入数据的大小和 GPU 的架构，自动选择最优的计算方式。**

#### **(2) ReLU (激活函数) 的 GPU 调用**

ReLU 激活函数会根据设备类型，调用 CUDA 内核来进行计算。

* **算子调用**：

  ```python

  x = F.relu(self.conv1(x))
  ```

* **内核调用**：

  * 对于 CUDA：调用 `relu_cuda_kernel`。

* **CUDA 实现的原型代码**：

  ```cpp
  Tensor relu_cuda(const Tensor& self) {
    // 使用 CUDA 进行并行计算 ReLU
    return at::clamp_min_cuda(self, 0);
  }
  ```

#### **(3) MaxPool2d (池化层) 的 GPU 调用**

* **算子调用**：

  ``` python
  x = F.max_pool2d(x, 2)
  ```

* **内核调用**：

  * 对于 CUDA：调用 `max_pool2d_cuda_kernel`。

* **CUDA 实现的原型代码**：

  ```cpp
  Tensor max_pool2d_cuda(
      const Tensor& self,
      IntArrayRef kernel_size,
      IntArrayRef stride,
      IntArrayRef padding,
      bool ceil_mode) {
    // 使用 CUDA 并行池化运算
    ...
  }
  ```

#### **(4) Linear (全连接层) 的 GPU 调用**

* **算子调用**：

  ```python

  x = F.relu(self.fc1(x))
  ```

* **内核调用**：

  * 对于 CUDA：调用 `linear_cuda_kernel`。

* **CUDA 实现的原型代码**：

  ```cpp
  Tensor linear_cuda(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias) {
    // 使用 CUDA 进行矩阵乘法和偏置加法
    return at::addmm_cuda(bias, input, weight.t());
  }
  ```

#### **(5) Softmax (输出层) 的 GPU 调用**

* **算子调用**：

  ```python

  return F.log_softmax(x, dim=1)
  ```

* **内核调用**：

  * 对于 CUDA：调用 `log_softmax_cuda_kernel`。

* **CUDA 实现的原型代码**：

  ```cpp
  Tensor log_softmax_cuda(const Tensor& self, int64_t dim, bool half_to_float) {
    // 使用 CUDA 并行计算 softmax
    ...
  }
  ```

---
### **Reference:**
1. https://github.com/pytorch/pytorch
2. https://docs.nvidia.com/deeplearning/cudnn/api/index.html
3. https://github.com/pytorch/pytorch