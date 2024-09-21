# tensorflow的显存分配机制



默认情况下，TensorFlow 会映射进程可见的所有 GPU（取决于 CUDA_VISIBLE_DEVICES）的几乎全部内存。这是为了减少内存碎片，更有效地利用设备上相对宝贵的 GPU 内存资源。为了将 TensorFlow 限制为使用一组特定的 GPU，可以使用 tf.config.set_visible_devices 方法。

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
```

在某些情况下，希望进程最好只分配可用内存的一个子集，或者仅在进程需要时才增加内存使用量。TensorFlow 为此提供了两种控制方法。
### 1.方法一：
第一个选项是通过调用 tf.config.experimental.set_memory_growth 来开启内存增长。此选项会尝试根据运行时分配的需求分配尽可能充足的 GPU 内存：首先分配非常少的内存，随着程序的运行，需要的 GPU 内存逐渐增多，于是扩展 TensorFlow 进程的 GPU 内存区域。内存不会被释放，因为这样会产生内存碎片。要关闭特定 GPU 的内存增长，在分配任何张量或执行任何运算之前使用以下代码。
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
```

第二个启用此选项的方式是将环境变量 TF_FORCE_GPU_ALLOW_GROWTH 设置为 true。这是一个特定于平台的配置。

### 2.方法二：
第二种方法是使用 tf.config.set_logical_device_configuration 配置虚拟 GPU 设备，并且设置可在 GPU 上分配多少总内存的硬性限制。

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
```

这在要真正限制可供 TensorFlow 进程使用的 GPU 内存量时非常有用。在本地开发中，与其他应用（如工作站 GUI）共享 GPU 时，这是常见做法。

---
### **Reference:**
1. https://www.tensorflow.org/guide/gpu?hl=zh-cn
2. https://www.tensorflow.org/guide/gpu_performance_analysis?hl=zh-cn【显存分配优化指南】


