# torch

包 torch 包含了多维张量的数据结构以及基于其上的多种数学操作。另外，它也提供了多种工具，其中一些可以更有效地对张量和任意类型进行序列化。

它有`CUDA` 的对应实现，可以在 `NVIDIA GPU` 上进行张量运算(计算能力>=2.0)。

## 张量 Tensors 

##### `torch.is_tensor`

```python
torch.is_tensor(obj)
```

如果 *obj* 是一个 `pytorch` 张量，则返回 True

· 参数： obj (Object) – 判断对象

`torch.is_storage`

如何 *obj* 是一个 `pytorch storage` 对象，则返回 True

· 参数： input (Object) – 判断对象

```python
import torch

# #创建一个 Tensor，并获取它的 Storage

x = torch.tensor([1, 2, 3])
storage = x.storage()

# #检查是否是 Storage 对象

print(torch.is_storage(storage))  # True
print(torch.is_storage(x))       # False（x 是 Tensor，不是 Storage）
print(torch.is_storage([1, 2, 3]))  # False（Python 列表不是 Storage）
```

- `torch.FloatStorage`（存储 `float32` 数据）
- `torch.LongStorage`（存储 `int64` 数据）
- `torch.ByteStorage`（存储 `uint8` 数据）

##### `torch.  set_default_tensor_type`  

```
torch.set_default_tensor_type(t)
```

- **参数 `t`**：
  - 必须是一个 **Tensor 类型**（如 `torch.FloatTensor`、`torch.DoubleTensor`、`torch.cuda.FloatTensor` 等）。
  - 不能是字符串（如 `"float32"`），必须直接传入 Tensor 类。
- **作用**：
  - 设置后，所有**新创建的 Tensor** 都会默认使用这个类型（除非显式指定 `dtype` 或 `device`）。

```python
torch.set_default_tensor_type(torch.DoubleTensor)
#例如设置默认Double
x = torch.tensor([1.0, 2.0, 3.0])
print(x.dtype)  # torch.float64
```

##### `**torch.numel**`

```python
y = torch.tensor([[1, 2], [3, 4]])
print(torch.numel(y))  # 4（2×2=4 个元素）
```

返回 input 张量中的元素个数，必须是int，不能是浮点型。

##### `torch.set_printoptions`

设置打印选项。 完全参考自 `Numpy`。参数:

· `precision` – 浮点数输出的精度位数 (默认为 8 )

· `threshold` – 阈值，触发汇总显示而不是完全显示(repr)的数组元素的总数 （默认为 1000）

· `edgeitems` – 汇总显示中，每维（轴）两端显示的项数（默认值为 3）

· `linewidth` – 用于插入行间隔的每行字符数（默认为 80）。`Thresholded matricies will ignorethis parameter.`

· `profile – pretty` 打印的完全默认值。 可以覆盖上述所有选项 (默认为 short, full)

```python
import torch

x = torch.rand(3) * 100
print("默认精度:", x)  # 例如: tensor([95.12345678, 12.34567890, 78.90123456])

torch.set_printoptions(precision=2)
print("精度=2:", x)    # 输出: tensor([95.12, 12.35, 78.90])

y = torch.arange(1000)  # 元素总数=1000（默认 threshold=1000，触发缩略）
print("默认缩略显示:", y)  # 输出: tensor([0, 1, 2, ..., 997, 998, 999])

torch.set_printoptions(threshold=10)  # 元素数>10即缩略
z = torch.arange(12)
print("强制缩略:", z)  # 输出: tensor([0, 1, 2, ..., 9, 10, 11])

torch.set_printoptions(edgeitems=1)  # 每维只显示1个边缘项
print("edgeitems=1:", z)  # 输出: tensor([0, ..., 11])

w = torch.rand(5, 5)
torch.set_printoptions(linewidth=40)  # 每行最多40字符
print("窄行宽:\n", w)  # 输出会频繁换行

torch.set_printoptions(profile='full')  # 完全显示（禁用缩略）
print("完全显示模式:\n", torch.arange(1000))  # 打印所有1000个元素

torch.set_printoptions()  # 重置所有参数
```

## 创建操作 Creation Ops 

##### `torch.eye`

```python
torch.eye(n, m=None, out=None)
```



返回一个 2 维张量，对角线位置全 1，其它位置全 0

参数:

· n (int ) – 行数

· m (int, optional) – 列数.如果为None,则默认为 n

· out (Tensor**, optinal) - Output tensor

返回值: 对角线位置全 1，其它位置全 0 的 2 维张量返回值类型: Tensor

例子:

`torch.eye(3)`

1 0 0

0 1 0

0 0 1

`[torch.FloatTensor of size 3x3]`

##### `from_numpy`

```python
torch.from_numpy(ndarray) #→ Tensor
```

`Numpy` 桥，将 `numpy.ndarray` 转换为 `pytorch` 的 `Tensor`。返回的张量 `tensor` 和 的 `ndarray` 共享同一内存空间。修改一个会导致另外一个也被修改。返回的张量不能改变大小。

例子:

```python
a = numpy.array([1, 2, 3])

t = torch.from_numpy(a)

t

torch.LongTensor([1, 2, 3])

t[0] = -1

a

array([-1, 2, 3])
```

```python
#反向转换（Tensor → NumPy）

t = torch.tensor([1, 2, 3])
a = t.numpy()  # 共享内存，类似 from_numpy
```

### **与 `torch.tensor()` 的区别**

| 方法                  | 内存共享 | 支持调整大小 | 适用场景                   |
| --------------------- | -------- | ------------ | -------------------------- |
| `torch.from_numpy(a)` | 是       | 否           | `NumPy → Tensor`（零拷贝） |
| `torch.tensor(a)`     | 否       | 是           | 深拷贝数据，独立内存       |

##### torch.linspace

`torch.linspace(start, end, steps=100, out=None) → Tensor` 是 `PyTorch` 中的一个函数，用于**在指定的区间 `[start, end]` 内生成等间距的 `steps` 个点**，并返回一个 **1 维张量**（向量）。这些点包括区间的端点 `start` 和 `end`。

- **参数**：
  - `start` (**float**)：序列的起始值。
  - `end` (**float**)：序列的结束值。
  - `steps` (**int**, 可选)：生成的样本数量（默认为 `100`）。
  - `out` (**Tensor**, 可选)：可选的输出张量（用于存储结果）。
- **返回值**：
  - 一个 **1 维张量**，包含 `steps` 个在 `[start, end]` 之间均匀分布的值。

```python
import torch

# 生成 5 个从 3 到 10 的等间距点

x = torch.linspace(3, 10, steps=5)
print(x)
#默认为float32
# 输出: tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])
```

##### torch.logspace

`torch.logspace(start, end, steps=100, out=None) → Tensor` 是 PyTorch 中的一个函数，用于**在对数尺度上生成均匀间隔的数值序列**。它返回一个 1 维张量，包含从10^start^到 10^end^的 `steps` 个点，这些点在对数空间中是等距分布的。

```python
torch.logspace(start, end, steps=100, out=None)# → Tensor
```

- **参数**：
  - `start` (**float**)：指数部分的起始值（实际起点为 10^start^）。
  - `end` (**float**)：指数部分的结束值（实际终点为 10^end^）。
  - `steps` (**int**, 可选)：生成的样本数量（默认为 `100`）。
  - `out` (**Tensor**, 可选)：可选的输出张量（用于存储结果）。
- **返回值**：
  - 一个 **1 维张量**，包含 `steps` 个在对数尺度上均匀分布的值。

```python
import torch

# 生成 5 个从 10^-10 到 10^10 的对数间隔点

x = torch.logspace(start=-10, end=10, steps=5)
print(x)

# 输出: tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10])
```

##### torch.ones

```python
torch.ones(*sizes, out=None) #→ Tensor
```

返回一个全为 1 的张量，形状由可变参数 sizes 定义。参数:

· sizes (int...) – 整数序列，定义了输出形状

· out (Tensor, optional) – 结果张量

 例子:

```python
torch.ones(2, 3)
```

##### `torch.arange`

**均匀分布**
生成的随机数服从 U(0,1)均匀分布，即每个数在 `[0, 1)` 区间内出现的概率均等。

**形状灵活**
通过 `*sizes` 参数可以生成任意维度的张量：

- 标量：`torch.rand(1)` → 形状为 `[1]`
- 向量：`torch.rand(4)` → 形状为 `[4]`
- 矩阵：`torch.rand(2, 3)` → 形状为 `[2, 3]`
- 高维张量：`torch.rand(2, 3, 4)` → 形状为 `[2, 3, 4]`

**默认数据类型**
返回的张量是 `torch.float32` 类型（除非通过 `dtype` 参数指定其他类型）。

### **与其他随机函数的区别**

| 函数              | 分布         | 区间           | 常见用途             |
| ----------------- | ------------ | -------------- | -------------------- |
| `torch.rand()`    | 均匀分布     | `[0, 1)`       | 初始化权重、生成概率 |
| `torch.randn()`   | 标准正态分布 | (−∞,+∞)(−∞,+∞) | 深度学习模型初始化   |
| `torch.randint()` | 离散均匀分布 | `[low, high)`  | 生成随机整数标签     |

##### `torch.arange`

```python
torch.arange(start, end, step=1, out=None) #→ Tensor
```

- **参数**：
  - `start` (float/int)：序列起始值（**包含**）。
  - `end` (float/int)：序列结束值（**不包含**）。
  - `step` (float/int)：步长（默认 1）。
  - `out` (Tensor, optional)：输出张量。
- **返回值**：
  - **1 维张量**

### **关键特性**

- **区间半开**：`[start, end)`，不包含 `end`。
- **支持浮点步长**：如 `step=0.5`。
- **推荐使用**：官方建议优先用此函数。

```python
import torch

# 整数序列（默认 step=1）

a = torch.arange(1, 4)
print(a)  # tensor([1, 2, 3])

# 浮点序列（step=0.5）

b = torch.arange(1, 2.5, 0.5)
print(b)  # tensor([1.0, 1.5, 2.0])
```

### **核心区别**

| 特性               | `torch.arange` | `torch.range` (已弃用) |
| ------------------ | -------------- | ---------------------- |
| **区间**           | `[start, end)` | `[start, end]`         |
| **包含 `end`**     | ❌              | ✅                      |
| **步长类型**       | 支持 float/int | 支持 float/int         |
| **官方推荐**       | ✅              | ❌（建议改用 `arange`） |
| **浮点数精度问题** | 较少           | 可能出现意外结果       |

## 索引,切片,连接,换位Indexing, Slicing, Joining, Mutating Ops 

##### `torch.cat()`

`torch.cat()` 是 `PyTorch` 中用于**沿指定维度拼接张量**的核心函数，可将多个张量（Tensor）连接成一个更大的张量。

### **核心特性**

1. **形状规则**：

   - 除 `dim` 维度外，其他维度大小必须相同。
   - 输出张量的 `dim` 维度大小是输入张量该维度大小的总和。

2. **与 `torch.stack` 的区别**：

   | 函数            | 作用           | 新增维度 | 输入要求                             |
   | --------------- | -------------- | -------- | ------------------------------------ |
   | `torch.cat()`   | 沿现有维度拼接 | ❌        | 所有张量形状需兼容                   |
   | `torch.stack()` | 在新维度上堆叠 | ✅        | 所有张量形状必须完全相同**核心特性** |

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])  # 形状 (2, 2)

# 沿第0维（行）拼接

y = torch.cat((x, x, x), dim=0)
print(y)
"""
tensor([[1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4]])  # 形状 (6, 2)
"""

# 沿第1维（列）拼接

z = torch.cat((x, x, x), dim=1)
print(z)
"""
tensor([[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]])  # 形状 (2, 6)
"""

a = torch.randn(2, 3, 4)
b = torch.randn(2, 5, 4)

# 沿第1维拼接（dim=1）

c = torch.cat((a, b), dim=1)
print(c.shape)  # torch.Size([2, 8, 4])
```

##### torch.chunk

`torch.chunk` 是 `PyTorch` 中的一个函数，用于将输入张量沿着指定的维度分割成多个块。以下是关于 `torch.chunk` 的详细说明：

###### 参数

- **tensor (Tensor)**: 待分块的输入张量。
- **chunks (int)**: 分块的个数。表示要将张量分割成多少个块。
- **dim (int, optional)**: 沿着此维度进行分块，默认值为 `0`。

###### 返回值

返回一个包含多个张量的元组，每个张量是原张量的一个块。这些块的大小可能不完全相同，具体取决于输入张量的大小和分块的个数。

```python
import torch

# 创建一个形状为 (6, 3) 的张量

tensor = torch.arange(18).view(6, 3)
print("原始张量:")
print(tensor)

# 沿着第 0 维将张量分成 3 块

chunks = torch.chunk(tensor, chunks=3, dim=0)
print("\n分割后的块:")
for i, chunk in enumerate(chunks):
    print(f"块 {i + 1}:")
    print(chunk)

#原始张量:
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]])

#块 1:
tensor([[0, 1, 2],
        [3, 4, 5]])

#块 2:
tensor([[6, 7, 8],
        [9, 10, 11]])

#块 3:
tensor([[12, 13, 14],
        [15, 16, 17]])
```

##### torch.squeeze

`torch.squeeze` 是 `PyTorch` 中的一个函数，用于去除输入张量形状中的单维度条目（即大小为 1 的维度）。它可以根据是否指定 `dim` 参数来决定如何去除这些单维度。以下是关于 `torch.squeeze` 的详细说明：

### 参数

- **input (Tensor)**: 输入张量。
- **dim (int, optional)**: 如果指定，则只在该维度上进行挤压操作。如果输入张量在该维度的大小不为 1，则不会有任何变化。
- **out (Tensor, optional)**: 输出张量。如果指定，结果将存储在该张量中。注意，`out` 张量的形状必须与挤压后的张量形状一致。

### 返回值

返回一个形状中去除了单维度的张量。如果输入张量中没有单维度，则返回的张量与输入张量相同。

```python
import torch

# 创建一个形状为 (1, 3, 1, 4) 的张量

tensor = torch.randn(1, 3, 1, 4)
print("原始张量形状:", tensor.shape)

# 去除所有大小为 1 的维度

squeezed = torch.squeeze(tensor)
print("去除所有大小为 1 的维度后的形状:", squeezed.shape)

# 在第 0 维进行挤压

squeezed_dim0 = torch.squeeze(tensor, dim=0)
print("在第 0 维挤压后的形状:", squeezed_dim0.shape)

# 在第 2 维进行挤压

squeezed_dim2 = torch.squeeze(tensor, dim=2)
print("在第 2 维挤压后的形状:", squeezed_dim2.shape)

原始张量形状: torch.Size([1, 3, 1, 4])
去除所有大小为 1 的维度后的形状: torch.Size([3, 4])
在第 0 维挤压后的形状: torch.Size([3, 1, 4])
在第 2 维挤压后的形状: torch.Size([1, 3, 4])
```

##### torch.transpose

`torch.transpose` 是 PyTorch 中用于交换张量中两个维度的函数。它不会改变张量的数据内容，只是重新排列维度的顺序。以下是关于 `torch.transpose` 的详细说明：

###### 参数

- **input (Tensor)**: 输入张量。
- **dim0 (int)**: 要交换的第一个维度。
- **dim1 (int)**: 要交换的第二个维度。

###### 返回值

返回一个转置后的张量，其形状与输入张量相同，但 `dim0` 和 `dim1` 的位置被交换了。输出张量与输入张量共享内存，因此修改其中一个张量的内容会同时影响另一个张量。

```python
# 创建一个形状为 (2, 3, 4) 的三维张量

tensor = torch.arange(24).view(2, 3, 4)
print("原始张量形状:", tensor.shape)

# 交换第 1 维和第 2 维

transposed = torch.transpose(tensor, dim0=1, dim1=2)
print("转置后的张量形状:", transposed.shape)

原始张量形状: torch.Size([2, 3, 4])
转置后的张量形状: torch.Size([2, 4, 3])
```

##### torch.mean

`torch.mean` 是 PyTorch 中的一个函数，用于计算输入张量在给定维度上的均值。以下是关于 `torch.mean` 的详细说明：

###### 参数

- **input (Tensor)**: 输入张量。
- **dim (int)**: 要计算均值的维度。
- **out (Tensor, optional)**: 输出张量。如果指定，结果将存储在该张量中。

###### 返回值

返回一个张量，其中包含输入张量在给定维度上的均值。输出张量的形状与输入张量相同，除了在给定维度上大小为 1。

```python
import torch

# 创建一个形状为 (4, 4) 的张量

a = torch.randn(4, 4)
print("原始张量:")
print(a)

# 计算所有元素的均值

mean_all = torch.mean(a)
print("所有元素的均值:", mean_all)

# 计算每行的均值

mean_dim1 = torch.mean(a, dim=1)
print("每行的均值:")
print(mean_dim1)

原始张量:
tensor([[-1.2738, -0.3058,  0.1230, -1.9615],
        [ 0.8771, -0.5430, -0.9233,  0.9879],
        [ 1.4107,  0.0317, -0.6823,  0.2255],
        [-1.3854,  0.4953, -0.2160,  0.2435]])

所有元素的均值: tensor(-0.2157)
每行的均值:
tensor([-0.8545,
         0.0997,
         0.2464,
        -0.2157])
```

##### torch.inverse

`torch.inverse` 是 `PyTorch` 中的一个函数，用于计算输入方阵（即行数和列数相等的二维张量）的逆矩阵。如果输入矩阵不可逆（即行列式为零），则会抛出错误。

###### 参数

- **input (Tensor)**: 输入的二维方阵张量。
- **out (Tensor, optional)**: 输出张量。如果指定，结果将存储在该张量中。

###### 返回值

返回一个张量，表示输入方阵的逆矩阵。返回的逆矩阵与输入矩阵的形状相同。

###### 注意事项

1. **输入必须是方阵**：输入张量必须是一个二维方阵，即行数和列数相等。
2. **矩阵可逆性**：如果输入矩阵不可逆（即行列式为零），`torch.inverse` 会抛出错误。
3. **输出矩阵的存储方式**：无论输入矩阵的存储方式如何，返回的逆矩阵总是以连续的存储方式返回。

```python
import torch

# 创建一个形状为 (3, 3) 的可逆矩阵

A = torch.tensor([[4, 7, 2],
                  [3, 5, 2],
                  [2, 1, 4]], dtype=torch.float32)
print("原始矩阵 A:")
print(A)

# 计算矩阵 A 的逆矩阵

A_inv = torch.inverse(A)
print("\n矩阵 A 的逆矩阵 A_inv:")
print(A_inv)

# 验证逆矩阵的正确性

identity_matrix = torch.mm(A, A_inv)
print("\nA 乘以 A_inv 的结果（应接近单位矩阵）:")
print(identity_matrix)

原始矩阵 A:
tensor([[4., 7., 2.],
        [3., 5., 2.],
        [2., 1., 4.]])

矩阵 A 的逆矩阵 A_inv:
tensor([[ 0.2500, -0.5000,  0.2500],
        [-0.2500,  0.5000, -0.2500],
        [ 0.2500, -0.2500,  0.2500]])

A 乘以 A_inv 的结果（应接近单位矩阵）:
tensor([[ 1.0000,  0.0000,  0.0000],
        [ 0.0000,  1.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0000]])
```

##### torch.unsqueeze

`torch.unsqueeze` 是 PyTorch 中的一个函数，用于在指定位置插入一个大小为 1 的维度，从而增加张量的维度。这个操作通常用于调整张量的形状，以满足某些函数或模型的输入要求。

```python
torch.unsqueeze(input, dim, out=None) #→ Tensor
```

###### 参数

- **input (Tensor)**: 输入张量。
- **dim (int)**: 要插入大小为 1 的维度的位置。
- **out (Tensor, optional)**: 输出张量。如果指定，结果将存储在该张量中。

###### 返回值

返回一个形状中在指定位置插入了大小为 1 的维度的新张量。输出张量与输入张量共享内存，因此修改其中一个张量的内容会同时影响另一个张量。

```python
import torch

# 创建一个形状为 (3, 4) 的二维张量

tensor = torch.randn(3, 4)
print("原始张量形状:", tensor.shape)

# 在第 0 维插入大小为 1 的维度

tensor_unsqueezed_dim0 = torch.unsqueeze(tensor, dim=0)
print("在第 0 维插入后的形状:", tensor_unsqueezed_dim0.shape)

# 在第 1 维插入大小为 1 的维度

tensor_unsqueezed_dim1 = torch.unsqueeze(tensor, dim=1)
print("在第 1 维插入后的形状:", tensor_unsqueezed_dim1.shape)

原始张量形状: torch.Size([3, 4])
在第 0 维插入后的形状: torch.Size([1, 3, 4])
在第 1 维插入后的形状: torch.Size([3, 1, 4])
```

## torch.nn

##### torch.nn.Parameter

`torch.nn.Parameter` 是 `PyTorch` 中的一个非常重要的类，它继承自 `torch.Tensor`，但主要用于表示神经网络中的参数。这些参数会自动被优化器（如 `torch.optim` 中的优化器）识别并更新，而普通的张量（`torch.Tensor`）则不会。

###### 参数

- **data (Tensor, optional)**: 参数的初始值。如果未指定，则默认为 `None`。
- **requires_grad (bool, optional)**: 是否需要计算梯度。默认为 `True`，表示该参数在反向传播时会计算梯度。

###### 主要特性

1. **继承自 `torch.Tensor`**：`torch.nn.Parameter` 是 `torch.Tensor` 的子类，因此它继承了 `torch.Tensor` 的所有特性，例如支持各种张量操作。
2. **自动注册为模型参数**：当 `torch.nn.Parameter` 被赋值给模块的属性时，它会被自动注册为模块的参数。这意味着优化器可以自动识别并更新这些参数。
3. **默认需要梯度**：与普通张量不同，`torch.nn.Parameter` 默认需要梯度（`requires_grad=True`），这使得它在训练过程中可以被自动更新。

###### 使用场景

`torch.nn.Parameter` 主要用于定义神经网络中的可训练参数，例如权重和偏置。通过将参数定义为 `torch.nn.Parameter`，可以确保这些参数在训练过程中被优化器更新。

```python
import torch
import torch.nn as nn

class SimpleLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearLayer, self).__init__()
        # 定义权重和偏置为可训练参数
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        return torch.mm(x, self.weight.t()) + self.bias

# 创建一个输入维度为 3，输出维度为 2 的线性层

layer = SimpleLinearLayer(input_dim=3, output_dim=2)

# 打印模型的参数

for name, param in layer.named_parameters():
    print(f"参数名称: {name}, 形状: {param.shape}")

参数名称: weight, 形状: torch.Size([2, 3])
参数名称: bias, 形状: torch.Size([2])
```

##### `class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`

一维卷积是一种在序列数据上进行的卷积操作，它通过将卷积核（filter）滑动到序列的每个位置，计算卷积核与序列局部区域的点积，从而生成新的序列。下面通过一个简单的例子来说明一维卷积的计算过程。

###### 例子

假设我们有一个一维序列 x=[1,2,3,4,5]，卷积核 w=[1,0,−1]，步长（stride）为 1，零填充（padding）为 1。

###### 步骤

1. **零填充**：由于零填充为 1，我们在序列 x 的两端各添加一个 0，得到 x′=[0,1,2,3,4,5,0]。
2. **滑动卷积核**：将卷积核 w 从左到右滑动到序列 x′ 的每个位置，计算卷积核与序列局部区域的点积。

##### `class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`

`torch.nn.Conv2d` 是 `PyTorch` 中的二维卷积层，用于对二维数据（如图像）进行卷积操作。它是构建卷积神经网络（CNN）的核心组件之一，广泛应用于图像识别、分类、分割等领域。

##### `class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`

`torch.nn.Conv3d` 是 `PyTorch` 中的三维卷积层，用于对三维数据（例如视频、3D 图像或体积数据）进行卷积操作。它在计算机视觉、医学图像处理和视频分析等领域中非常有用。

##### `torch.nn.ConvTranspose2d`

`torch.nn.ConvTranspose2d` 是 `PyTorch` 中的二维转置卷积层（也称为反卷积层或分数步长卷积层）。它主要用于上采样（`upsampling`）操作，即将输入张量的空间尺寸（高度和宽度）放大。这种操作在生成模型（如 GANs）、语义分割和超分辨率等任务中非常常见。

## `torch.optim` 

**`optim.SGD`**

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### 主要组成部分

1. **`optim.SGD`**:
   - 这是 PyTorch 中实现随机梯度下降（SGD）优化算法的类。
   - SGD 是一种常用的优化算法，通过迭代更新模型参数以最小化损失函数。
2. **`model.parameters()`**:
   - 这是 PyTorch 模型（通常是 `torch.nn.Module` 的实例）的参数。
   - `model.parameters()` 返回一个生成器，包含模型中所有需要更新的参数（通常是权重和偏置）。
3. **`lr=0.01`**:
   - `lr` 是学习率（learning rate），表示每次参数更新的步长。
   - 学习率决定了参数更新的幅度。学习率过高可能导致训练不稳定，过低则可能导致训练速度过慢。
   - 在这个例子中，学习率设置为 0.01。
4. **`momentum=0.9`**:
   - `momentum` 是动量（momentum）参数，用于加速梯度下降过程，并减少震荡。
   - 动量通过在当前梯度方向上添加一部分之前的速度来平滑更新，有助于加速收敛并避免局部最小值。
   - 在这个例子中，动量设置为 0.9，这是一个常用的值。

**`optim.Adam`**

```python
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

#### 主要组成部分

1. **`optim.Adam`**:
   - 这是 PyTorch 中实现 Adam 优化算法的类。
   - Adam 是一种自适应矩估计（Adaptive Moment Estimation）优化算法，结合了动量（Momentum）和 RMSprop（Root Mean Square Propagation）的优点。
2. **`[var1, var2]`**:
   - 这是一个参数列表，包含需要优化的变量。
   - `var1` 和 `var2` 是需要优化的张量，通常是模型的参数（如权重和偏置）。
   - 这些变量必须是 `torch.Tensor` 类型，并且需要设置 `requires_grad=True`，以便计算梯度。
3. **`lr=0.0001`**:
   - `lr` 是学习率（learning rate），表示每次参数更新的步长。
   - 在这个例子中，学习率设置为 0.0001。

#### Adam 优化算法的特点

1. **自适应学习率**：Adam 为每个参数维护一个独立的学习率，根据参数的历史梯度动态调整学习率。这使得优化过程更加稳定，尤其是在处理稀疏梯度时。
2. **动量**：Adam 引入了动量机制，通过在当前梯度方向上添加一部分之前的速度来平滑更新，有助于加速收敛并避免局部最小值。
3. **偏差修正**：Adam 在计算梯度的指数加权移动平均时，对初始阶段的偏差进行了修正，使得优化器在训练初期也能表现良好。

```python
# 训练循环
for epoch in range(10):  # 训练 10 个 epoch
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
```

## 保存模型参数

### 保存和加载模型参数（`state_dict`）

`state_dict` 是一个包含模型所有参数的字典对象。保存和加载 `state_dict` 是最常用的方法之一。

```python
# 假设 model 是你的模型实例

torch.save(model.state_dict(), 'model.pth')
```

```python
# 假设 model 是你的模型实例，且已经定义了相同的模型结构

model.load_state_dict(torch.load('model.pth'))
```

### 保存和加载整个模型

除了保存模型参数外，还可以保存整个模型，包括模型的结构和参数。

```python
torch.save(model, 'model.pth')
```

```python
# 加载整个模型时，不需要重新定义模型结构

model = torch.load('model.pth')
```


