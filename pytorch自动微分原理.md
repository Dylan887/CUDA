# pytorch自动微分原理


## autograd简介

`Tensor`是PyTorch实现多维数组计算和自动微分的关键数据结构。一方面，它类似于numpy的ndarray，用户可以对`Tensor`进行各种数学运算；另一方面，当设置`.requires_grad = True`之后，在其上进行的各种操作就会被记录下来，用于后续的梯度计算，其内部实现机制被称为**动态计算图(dynamic computation graph)**。

> `Variable`变量：在PyTorch早期版本中，`Tensor`只负责多维数组的运算，自动微分的职责是`Variable`完成的，因此经常可以看到因而产生的包装代码。而在0.4.0版本之后，二者的功能进行了合并，使得自动微分的使用更加简单了。

autograd机制能够记录作用于`Tensor`上的所有操作，生成一个动态计算图。图的叶子节点是输入的数据，根节点是输出的结果。当在根节点调用`.backward()`的时候就会从根到叶应用链式法则计算梯度。默认情况下，只有`.requires_grad`和`is_leaf`两个属性都为`True`的节点才会被计算导数，并存储到`grad`中。

> 动态计算图本质上是一个有向无环图，因此“叶”和“根”的称呼是不太准确的，但是这种简称可以帮助理解，PyTorch的文档中仍然采用这种说法。

## requires_grad属性

`requires_grad`属性默认为False，也就是`Tensor`变量默认是不需要求导的。如果一个节点的`requires_grad`是True，那么所有依赖它的节点`requires_grad`也会是True。换言之，如果一个节点依赖的所有节点都不需要求导，那么它的`requires_grad`也会是False。在反向传播的过程中，该节点所在的子图会被排除在外。



```python
>>> x = torch.randn(5, 5)  # requires_grad=False by default
>>> y = torch.randn(5, 5)  # requires_grad=False by default
>>> z = torch.randn((5, 5), requires_grad=True)
>>> a = x + y
>>> a.requires_grad
False
>>> b = a + z
>>> b.requires_grad
True
```

## Function类

我们已经知道PyTorch使用动态计算图(DAG)记录计算的全过程，那么DAG是怎样建立的呢？一些博客认为DAG的节点是Tensor(或说Variable)，这其实是不准确的。DAG的节点是Function对象，边表示数据依赖，从输出指向输入。因此Function类在PyTorch自动微分中位居核心地位，但是用户通常不会直接去使用，导致人们对Function类了解并不多。

每当对Tensor施加一个运算的时候，就会产生一个Function对象，它产生运算的结果，记录运算的发生，并且记录运算的输入。`Tensor`使用`.grad_fn`属性记录这个计算图的入口。反向传播过程中，autograd引擎会按照逆序，通过Function的backward依次计算梯度。

![alt text](torch动态图-1.gif)

## backward函数

backward函数是反向传播的入口点，在需要被求导的节点上调用backward函数会计算梯度值到相应的节点上。backward需要一个重要的参数grad_tensor，但如果节点只含有一个标量值，这个参数就可以省略（例如最普遍的`loss.backward()`与`loss.backward(torch.tensor(1))`等价），否则就会报如下的错误：

> Backward should be called only on a scalar (i.e. 1-element tensor) or with gradient w.r.t. the variable

要理解这个参数的内涵首先要从数学角度认识梯度运算。如果有一个向量函数 $\vec{y}=f(\vec{x})$ ，那么 $\vec{y}$ 相对于 $\vec{x}$ 的梯度是一个雅克比矩阵(Jacobian matrix)：

$$
J=\left(\begin{array}{ccc}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{array}\right)
$$



本文讨论的主角torch.autograd本质上是一个向量-雅克比乘积(*vector-Jacobian product*)的计算引擎，即计算vT⋅JvT⋅J，而所谓的参数grad_tensor就是这里的vv。由定义易知，参数grad_tensor需要与Tensor本身有相同的size。通过恰当地设置grad_tensor，容易计算任意的∂ym∂xn∂ym∂xn求导组合。

反向传播过程中一般用来传递上游传来的梯度，从而实现链式法则，简单的推导如下所示：

$$
J^T \cdot v=\left(\begin{array}{ccc}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_1} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial x_n} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{array}\right)\left(\begin{array}{c}
\frac{\partial l}{\partial y_1} \\
\vdots \\
\frac{\partial l}{\partial y_m}
\end{array}\right)=\left(\begin{array}{c}
\frac{\partial l}{\partial x_1} \\
\vdots \\
\frac{\partial l}{\partial x_n}
\end{array}\right)
$$



(注：这里的计算结果被转置为列向量以方便查看)

**注意：梯度是累加的**

backward函数本身没有返回值，它计算出来的梯度存放在叶子节点的grad属性中。PyTorch文档中提到，如果grad属性不为空，新计算出来的梯度值会直接加到旧值上面。

为什么不直接覆盖旧的结果呢？这是因为有些Tensor可能有多个输出，那么就需要调用多个backward。叠加的处理方式使得backward不需要考虑之前有没有被计算过导数，只需要加上去就行了，这使得设计变得更简单。因此我们用户在反向传播之前，常常需要用zero_grad函数对导数手动清零，确保计算出来的是正确的结果。

---
### **Reference:**
https://www.cnblogs.com/cocode/p/10746347.html