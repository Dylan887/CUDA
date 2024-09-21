# pytorch 显存分配机制
**pyTorch** 的显存分配机制旨在高效利用 GPU 的显存，并减少不必要的显存分配和释放操作，从而提高模型训练和推理的性能。以下是 PyTorch 在使用 CUDA 进行显存分配和管理时的一些主要机制和特点：

### 1. **显存管理的基础**

PyTorch 使用了 **动态显存分配** 策略。当你在 PyTorch 中使用 CUDA 张量时，显存不会在一开始就分配完所有可用的 GPU 内存，而是根据需要动态分配。例如，当你创建一个张量并将其移动到 GPU 上时，PyTorch 会分配所需的显存。如果张量被删除或不再需要，PyTorch 会释放显存，以便其他任务使用。

```python
import torch

# 张量创建并移动到 GPU
x = torch.randn(1024, 1024, device='cuda')  # 动态分配显存
```

### 2. **显存缓存机制 (Caching Allocator)**

为了优化显存的使用和减少内存碎片，PyTorch 使用了一个 **Caching Allocator**（缓存分配器）。该机制通过以下方式减少显存的频繁分配和释放操作：

* 当一个 CUDA 张量被销毁时，PyTorch 并不会立刻将显存还给操作系统，而是将这部分显存缓存起来，以便在后续的张量操作中复用。这种机制避免了频繁的显存分配和释放带来的开销。
* 下次需要分配相同大小的张量时，PyTorch 会优先复用之前缓存的显存，从而加快内存分配速度并减少碎片。

例如，下面的代码可能只会导致一次显存分配，后续的张量可以复用之前的显存：

```python
x = torch.randn(1024, 1024, device='cuda')
del x  # 不会立即释放显存，而是缓存
y = torch.randn(1024, 1024, device='cuda')  # 复用已缓存的显存
```

### 3. **显存分配与释放的控制**

PyTorch 提供了几个控制和监视显存使用的工具，可以帮助开发者手动管理显存的分配和释放：

* **`torch.cuda.empty_cache()`**：这个函数不会实际释放显存给操作系统，但它会清空 PyTorch 的缓存，使得显存可以被其他 CUDA 程序使用。开发者可以在不希望显存被过度缓存时调用该函数。

  ``` python
  torch.cuda.empty_cache()  # 清空 PyTorch 内部缓存的显存
  ```

* **`torch.cuda.memory_allocated()`** 和 **`torch.cuda.memory_reserved()`**：这些函数可以帮助监视当前显存的使用情况。`memory_allocated()` 返回当前已经分配的显存量，而 `memory_reserved()` 返回当前为缓存保留的显存量。

  ```python
  print(f"Allocated Memory: {torch.cuda.memory_allocated()} bytes")
  print(f"Reserved Memory: {torch.cuda.memory_reserved()} bytes")
  ```

### 4. **显存复用与共享机制**

PyTorch 的 Caching Allocator 不仅支持在同一进程中复用显存，还能够确保多个操作之间共享相同的显存。如果一个操作的输出张量和输入张量具有相同的大小和形状，PyTorch 可以在后台共享显存，以减少显存占用。这种机制在某些场景下可以进一步优化显存使用。

### 5. **多 GPU 显存分配**

当使用多个 GPU 时，PyTorch 为每个 GPU 独立管理显存分配。每个 GPU 都有自己的显存缓存机制，并且 PyTorch 可以自动将张量分配到不同的 GPU 上，前提是你明确指定了设备。例如：

```python
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")

x = torch.randn(1024, 1024, device=device1)  # 分配到 GPU 0
y = torch.randn(1024, 1024, device=device2)  # 分配到 GPU 1
```

PyTorch 的显存分配机制在多 GPU 模式下仍然使用缓存分配器，以减少每个 GPU 的显存分配开销。

### 6. **显存溢出与自动混合精度 (AMP)**

在大模型训练中，显存管理非常重要，尤其是当显存有限时。PyTorch 提供了 **自动混合精度（AMP）** 功能，结合 `torch.cuda.amp` 来减少显存占用。AMP 通过在前向传播中使用 16 位浮点数（FP16）来减少显存使用量，同时在某些关键计算中保持 32 位精度以确保数值稳定性。

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)  # 使用混合精度进行计算
```

使用 AMP 不仅可以减少显存占用，还可以提升计算性能，尤其是在最新的 NVIDIA GPU 上（如 RTX 30 系列），它们对混合精度计算有硬件支持。

### 7. **显存分配的动态调整与调试**

为了帮助调试显存使用问题，PyTorch 提供了一些工具来跟踪和优化显存使用情况：

* **`torch.cuda.set_per_process_memory_fraction()`**：允许设置当前进程最多使用 GPU 显存的比例。例如，你可以限制某个进程最多使用 GPU 总显存的 80%。

  ```python

  torch.cuda.set_per_process_memory_fraction(0.8, device='cuda:0')
  ```

* **`torch.cuda.memory_summary()`**：这是一个详细的显存使用报告工具，可以输出当前 GPU 显存的使用情况，包括缓存的分配器状态。对于调试显存溢出或内存泄漏问题，这个工具非常有用。

  ```python

  print(torch.cuda.memory_summary())
  ```

### 8. **自动释放显存（当不再需要时）**

PyTorch 的 `autograd` 机制会跟踪张量的依赖关系，并自动管理显存的释放。当某些张量不再需要时（例如，在反向传播后），PyTorch 会自动释放这些张量占用的显存。这是通过计算图的生命周期管理来实现的，尤其是在训练结束或前向传播和反向传播完成后，计算图会被销毁，从而释放显存。

```python
# 在反向传播之后，计算图被销毁，相关的张量显存被释放
loss.backward()
```

### 总结

* **动态显存分配**：PyTorch 在需要时动态分配显存，而不是一次性占用所有可用显存。
* **缓存分配器 (Caching Allocator)**：避免频繁分配和释放显存，减少内存碎片，优化性能。
* **显存监控工具**：提供了多种 API 来监控显存使用情况，并在需要时手动清空缓存。
* **自动混合精度 (AMP)**：通过降低部分计算的精度来减少显存占用。
* **多 GPU 管理**：每个 GPU 独立管理显存，支持多 GPU 下的显存分配和调度。

---
### **Reference:**
1. https://github.com/pytorch/pytorch
2. https://stackoverflow.com/questions/tagged/pytorch
3. https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-graph-library.html
4. https://pytorch.org/docs/stable/torch_cuda_memory.html
5. [pytorch显存管理机制优化参考](https://www.cvmart.net/community/detail/6242)





