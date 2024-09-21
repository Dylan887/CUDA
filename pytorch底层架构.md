# pytorch 架构结构分析
Pytorch是一个基于Torch框架的，开源的Python深度学习库。Torch框架一开始是不支持Python的，后来成功把Torch移植到python上，所以叫它Pytorch。
Torch是一个科学计算框架，广泛支持将GPU放在首位的机器学习算法。

## pytorch 架构图
![alt text](/images/696df6f6f3a561d6318bbb0e3283c444.jpeg)

**1. 数据存储层：管理数据存储和内存分配，Tensor在整个计算图中流动并执行操作**
* Tensor（张量）: PyTorch中最基础的数据结构，类似于NumPy的数组，但支持GPU加速。Tensor可以存储各种类型的多维数组，并且可以在CPU和GPU之间快速转换。
* Storage: 是Tensor底层的实现，用来管理实际存储的内存。Storage是低层次的，用于分配内存，而Tensor则是更高层的接口。

**2. 网络搭建层：负责网络的搭建，定义了神经网络的结构、层的堆叠方式以及初始化参数等**
* init: 用于初始化模型的参数，比如权重和偏置。可以自定义初始化策略，确保网络的训练效果。
* Parameter: Parameter类是特殊的Tensor，当被赋予nn.Module时，它会自动被注册为可学习的参数（即在反向传播时被更新的权重）。
* nn.Container: nn.Module的容器类，帮助组织网络中的多个层，比如nn.Sequential。
* Module/Sequential: nn.Module是PyTorch中所有神经网络层的基类，可以被继承来创建自定义模型。nn.Sequential是容器类，用于按顺序将多个层组合起来。
* Functional: torch.nn.functional提供了许多函数，比如激活函数、卷积操作等。相比于nn.Module中的层，functional提供了操作符的直接调用，而不需要保存状态。
* 网络模型: PyTorch提供了多种现成的网络模块，例如nn.RNN、nn.LSTM等。这些都是常见的神经网络模型，用户可以直接使用或者进行扩展。

**3. 优化层：负责计算梯度并进行参数更新，自动微分功能显著简化了深度学习的反向传播实现**
* optim: 这一模块包含了PyTorch中各种优化器（如SGD、Adam等），它们根据损失函数的梯度对模型参数进行更新，帮助模型逼近最优解。
* Autograd: PyTorch的自动微分模块，能够自动计算所有操作的梯度。它通过动态计算图记录每次操作，允许模型在每次前向传播中动态构建和反向传播。

**4. 应用层：对数据的加载和处理，将数据转换为模型可以处理的格式，并通过批次传递给模型进行训练或测试**
* DataLoader: 负责批量加载数据，可以并行处理、打乱数据，并支持自定义的数据预处理和数据增强。
* Dataset: 这个类表示数据集，用户可以继承并自定义__getitem__()和__len__()方法来加载特定格式的数据。


## pytorch 结构图
![alt text](/images/b03306f68bc0285b66c7d149a142e8e6.png)

**Pytorch主要分成了Torch和torchvision两大块**
torch是Pytorch深度学习框架的核心，用来定义多维张量（Tensor）结构及基于张量的多种数学操作，是一个科学计算框架，广泛支持将GPU放在首位的机器学习算法。torchvision则是基于torch开发的，专门用来处理计算机视觉或者图像方面的库。

### 1.torch
torch中主要由数据载体模块、数据存储模块、神经网络模块、求导模块、优化器模块、加速模块以及效率工具模块组成。

1）数据载体模块（torch.tensor)：Pytorch深度学习框架是针对tensor类型的数据进行计算的，tensor类型的数据是Pytorch最基础的概念，其参与了整个计算过程，所有tensor类型的数据都具有以下8种基本属性：
    
    ①data:被包装的Tensor数据

    ②dtype: 张量的数据类型

    ③shape: 张量的形状

    ④device: 张量所在的设备， GPU/CPU， 张量放在GPU上才能使用加速

    ⑤grad: data的梯度

    ⑥grad_fn: fn表示function的意思，记录创建张量时用到的方法，比如说加法，这个方法在求导过程需要用到， 是自动求导的关键

    ⑦requires_grad: 表示是否需要计算梯度

    ⑧is_leaf: 表示是否是叶子节点（张量）。为了节省内存，在反向传播完了之后，非叶子节点的梯度是默认被释放掉的，如果想保留中间节点a的梯度，可以使用retain_grad()方法，即a.retain_grad()就行

2）数据存储模块（torch.Storage)：管理Tensor是以byte类型还是char类型，CPU类型还是GPU类型进行存储。

3）神经网络模块（torch.nn)：它是torch的核心，torch.nn模块下包括了参数管理、参数初始化、网络层功能函数、模型创建以及封装好的网络函数这5大工具，具体如图所示。
![alt text](/images/3caa939ca7a49294f76495c9063ba004.png)
①参数管理工具（nn.Parameter）：torch.nn.Parameter继承torch.Tensor,其作用将不可训练的Tensor类型数据转化为可训练的parameter参数。在Pytorch中，模型的参数是需要被优化器训练的，因此，通常要设置参数为 requires_grad = True 的张量，而张量的 requires_grad属性默认是false。同时，在一个模型中，往往有许多的参数，要手动管理这些参数并不是一件容易的事情。Pytorch中的参数用nn.Parameter来管理，因为nn.Parameter管理的参数都 具有 requires_grad = True 属性，不需要再去手动地一个个去管理参数。

②初始化工具（nn.init）：只要采用恰当的权值初始化方法，就可以实现多层神经网络的输出值的尺度维持在一定范围内, 这样在反向传播的时候，就有利于缓解梯度消失或者爆炸现象的发生。Pytorch中提供的权重初始化方法主要分为四大类：

* 针对饱和激活函数（sigmoid， tanh）：Xavier均匀分布， Xavier正态分布

* 针对非饱和激活函数（relu及变种）：Kaiming均匀分布， Kaiming正态分布

* 三个常用的分布初始化方法：均匀分布，正态分布，常数分布

* 三个特殊的矩阵初始化方法：正交矩阵初始化，单位矩阵初始化，稀疏矩阵初始化

③网络层功能函数工具（nn.functional）：都是实现好的函数，调用时需要手动创建好weight、bias变量，该模块中的函数主要用来定义nn中没有而自己又需要的功能。

④模型创建工具（nn.Module）：nn.Module既是存放各种网络层结构的容器（一个module可以包含多个子module），也可以看作是一种结构类型(如卷积层、池化层、全连接层、BN层、非线性层、损失函数、优化器等都是module类型的）。nn.Module是一种结构类型可以和torch.Tensor是一种数据类型对比着理解。nn.Module是所有网络层的基类，管理所有网络层的属性。属于Module类型的结构都有下面8种属性：
![alt text](/images/a155983a51bf9243fcaf39c3a7181e51.png)
* _parameters： 存储管理属于nn.Parameter类的属性，例如权值，偏置这些参数

* _modules: 存储管理Module类型的结构， 比如卷积层，池化层，就会存储在_modules中

* _buffers: 存储管理缓冲属性， 如BN层中的running_mean， std等都会存在这里面

***_hooks: 存储管理钩子函数
* 一个module相当于一个运算， 必须实现forward()函数（从计算图的角度去理解）。在nn.Module的__init__()方法中构建子模块，在nn.Module的forward（）方法中拼接子模块，这就是创建模型的两个要素。
* 还有一种常用的更小型的容器是nn.Sequential，用于按顺序 包装一组网络层，并且继承于nn.Module，也是Module类型的结构。


4）求导模块（torch.autograd)：由于Pytorch采用了动态图机制，在每一次（损失函数）反向传播结束之后，计算图（此时的计算图既有前向传播的数据也有反向传播的梯度数据）都会被释放掉（因为动态图运算和搭建是同时进行的，每计算（前向传播）一次，就会搭建一次计算图，及时释放可以节省内存），只有叶子节点(参数w，b，输入x，以及真实y都是叶子节点，但是x和y这两个张量是不需要求导的，即requires_grad=false，只有参数需要求导，所以最后保留的梯度只有w,b)的梯度会被保留下来（叶子节点的梯度在权重更新时会用到），别的节点n如果想要保留下来，需调用n.retain_grad()函数。由于叶子节点的梯度不会自动清零，每次反向传播叶子节点的梯度都会和上次反向传播的梯度叠加，因此权重更新后需通过optimizer.zero_grad()函数手动将叶子节点的梯度清零。


**Pytorch自动求导机制使用的是torch.autograd.backward（）方法， 功能就是自动求取梯度。**
![alt text](/images/daff5b0b24d41385bb50bac24cf044ab.png)
* tensors：表示用于求导的张量，如loss。

* retain_graph：表示保存计算图， 由于Pytorch采用了动态图机制，在每一次反向传播结束之后，计算图都会被释放掉。如果我们不想被释放，就要设置这个参数为True。

* create_graph：表示创建导数计算图，用于高阶求导。

* grad_tensors：表示多梯度权重。如果有多个loss需要计算梯度的时候，就要设置这些loss的权重比例。

代码中一般使用loss .backward()就可以实现自动求导，那是因为backward()函数还是通过调用了torch.autograd.backward()函数从而实现自动求取梯度的.

5）优化器模块（torch.optim)：通过前向传播的过程，得到了模型输出与真实标签的差异，我们称之为损失， 有了损失，损失就会进行反向传播得到参数的梯度，优化器要根据我们的这个梯度去更新参数，使得损失不断的降低。torch.optim模块下有十款常用的优化器，分别是：

①optim.SGD: 随机梯度下降法
②optim.Adagrad: 自适应学习率梯度下降法
③optim.RMSprop: Adagrad的改进
④optim.Adadelta: Adagrad的改进
⑤optim.Adam: RMSprop结合Momentum
⑥optim.Adamax: Adam增加学习率上限
⑦optim.SparseAdam: 稀疏版的Adam
⑧optim.ASGD: 随机平均梯度下降
⑨optim.Rprop: 弹性反向传播
⑩optim.LBFGS: BFGS的改进

其中最常用的就是optim.SGD和optim.Adam。


6）加速模块（torch.cuda）：用于GPU加速的模块，定义了与CUDA运算相关的一系列函数

7)效率工具模块（torch.utils）：里面包含了Pytorch的数据读取机制torch.utils.data.DataLoader等一些相关的函数（数据读取机制下一篇文章会具体讲解，这里不提了）以及一些可视化工具tensorboard、CAM等涉及的函数


###  2.torchvision
torchvision则是基于torch开发的，专门用来处理计算机视觉或者图像方面的库。主要有torchvision.datasets、torchvision.models、torchvision.transforms以及torchvision.utils四大块，最重要用的最多的就是图像预处理模块torchvision.transforms,[torchvision详情参考](https://pytorch.org/vision/stable/index.html)。


### 源码结构
![alt text](3782e05fcf9569e36bc8dbbbb6f00586.png)
[旧版源码](https://gitee.com/ascend/pytorch/repository/archive/v1.11.0.zip)

---
### **Reference:**
1. https://pytorch.org/vision/stable/index.html
2. https://blog.csdn.net/Mike_honor/article/details/125742111
3. https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/700alpha002/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0049.html
4. https://hurray0.com/menu/151/
5. https://blog.csdn.net/qq_28726979/article/details/120690343