# RetNet

Transformer是大型语言模型的主流架构。然而，transformer的训练并行性是以低效的推理为代价，这使得transformer对部署不友好。不断增长的序列长度会增加GPU内存消耗和延迟，并降低推理速度。许多算法都在继续开发下一代架构，旨在保持训练并行性和transformer的竞争性能，同时具有高效的O(1)推理复杂度。但同时实现上述目标是具有挑战性的，即所谓的不可能三角形（如下图）。

![1747960765983](RetNet.assets\1747960765983.png)

参考论文：Retentive Network: A Successor to Transformer for Large Language Models

链接：https://arxiv.org/abs/2307.08621

RetNet代码：https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py#L33

本文提出的算法同时具备低成本的推理、高效的长序列建模、与transformer相当的性能，并同时进行并行模型训练。引入了一种[Multi-Scale Retention](https://zhida.zhihu.com/search?content_id=239657237&content_type=Article&match_order=1&q=Multi-Scale+Retention&zhida_source=entity)机制来替代多头注意力，该机制有三种计算范式，即并行、循环和chunkwise循环表示。

* 并行表示使训练并行性能够充分利用GPU设备
* 递归表示能够在内存和计算方面有效地进行  推理。这样可以显著降低部署成本和延迟

* 块递归表示可以进行高效的长序列建模。并行编码每个局部块以提高计算速度，同时反复编码全局块以节省GPU内存

## 对比Transformer

![1747961199723](RetNet.assets\1747961199723.png)

图 1：与 Transformer 相比，保留网络 (RetNet) 实现了低成本推理（即 GPU 内存、吞吐量和延迟）、训练并行性和良好的扩展曲线。推理成本结果以 8k 作为输入长度。图 6 显示了不同序列长度下的更多结果。

![1747961443761](RetNet.assets\1747961443761.png)

当输入序列长度增加的时候，RetNet 的 GPU 显存占用一直是稳定的和权值差不多，而 Transformer 则是和输入长度成正比。

首先看红色线和紫色线，都是输入长度在 8192 下，RetNet 和 Transformer 推理延时的对比。

可以看到当 batch size 增加的时候，RetNet的推理延时也还是很稳定，而 Transformer 的推理延时则是和 batch size 成正比。

而 Transformer 即使是输入长度缩小到 1024 ，推理延时也还是比 RetNet要高。

## **RetNet 架构解读**

在这项工作中，我们提出了保留网络（RetNet），同时实现了低成本推理、高效的长序列建模、可与 Transformer 比较的性能以及并行模型训练。具体来说，我们引入了一种多尺度保持机制来替代多头注意，它有三种计算范式，即并行、递归和分块递归表示。首先，并行表示增强了训练的并行性，以充分利用 GPU 设备。其次，递归表示法可以在内存和计算方面实现高效的 O(1) 推断。部署成本和延迟可以显著降低。此外，无需键值缓存技巧，大大简化了实现过程。第三，chunkwise 循环表示法可以执行高效的长序列建模。我们对每个局部块进行并行编码以提高计算速度，同时对全局块进行并发编码以节省 GPU 内存。

RetNet 架构和 Transformer 类似，也是堆叠  层同样的模块，每个模块内部包含两个子模块：一个 [multi-scale retention](https://zhida.zhihu.com/search?content_id=231680498&content_type=Article&match_order=1&q=multi-scale+retention&zhida_source=entity)（MSR）和一个 [feed-forward network](https://zhida.zhihu.com/search?content_id=231680498&content_type=Article&match_order=1&q=feed-forward+network&zhida_source=entity)

 (FFN)。

下面详细解读一下这个 retention 子模块。

首先给定一个输入序列
$$
\{x_i\}_{i=1}^{|x|}
$$

$$
x=x_1\ldots x_{|x|}
$$

其中|X|表示序列的长度。然后输入序列首先经过 embedding 层得到词嵌入向量：
$$
X^0=[x_1,\ldots,x_{|x|}]\in\mathbb{R}^{|x|\times d}
$$
其中d表示隐含层的维度。

## **Retention 机制**

![1747962091270](RetNet.assets\1747962091270.png)
$$
v_{N}=X_{n}\cdot w_{V}
$$
然后同样有类似 Transformer 架构的 Q 和 K 的投影：
$$
Q=XW_Q,K=XW_K
$$
![1747962359619](RetNet.assets\1747962359619.png)
$$
s_n=As_{n-1}+K_n^Tv_n
$$

$$
o_n=Q_ns_n=\sum_{m=1}^nQ_nA^{n-m}K_m^Tv_m
$$

![1747962508715](RetNet.assets\1747962508715.png)
$$
\begin{aligned}Q_{n}s_{n}&=Q_n(As_{n-1}+K_n^Tv_n)\\&=Q_n(A(As_{n-2}+K_{n-1}^Tv_{n-1})+K_n^Tv_n)\\&=Q_n(A^2s_{n-2}+A^1K_{n-1}^Tv_{n-1}+A^0K_n^Tv_n)\\&=Q_n(A^2(As_{n-3}+K_{n-2}^Tv_{n-2})+A^1K_{n-1}^Tv_{n-1}+A^0K_n^Tv_n)\\&=Q_n(A^3s_{n-3}+A^2K_{n-2}^Tv_{n-2}+A^1K_{n-1}^Tv_{n-1}+A^0K_n^Tv_n)\end{aligned}
$$
![1747962610000](RetNet.assets\1747962610000.png)
$$
s_1=As_0+K_1^Tv_1=K_1^Tv_1
$$
再继续上述推导过程：
$$
\begin{aligned}Q_{n}s_{n}&=Q_n(A^3s_{n-3}+A^2K_{n-2}^Tv_{n-2}+A^1K_{n-1}^Tv_{n-1}+A^0K_n^Tv_n)\\&=Q_n(A^{n-(n-3)}s_{n-3}+A^{n-(n-2)}K_{n-2}^Tv_{n-2}+A^{n-(n-1)}K_{n-1}^Tv_{n-1}\\&+A^{n-n}K_{n}^{T}v_{n})\end{aligned}
$$
所以根据上述推导过程和条件归纳可得：
$$
\begin{aligned}Q_{n}s_{n}&=Q_n(A^{n-1}s_1+A^{n-2}K_2^Tv_2+\ldots+A^{n-(n-2)}K_{n-2}^Tv_{n-2}+A^{n-(n-1)}K_{n-1}^Tv_{n-1}\\&+A^{n-n}K_n^Tv_n)\\&=Q_n(A^{n-1}K_1^Tv_1+A^{n-2}K_2^Tv_2+\ldots+A^{n-(n-2)}K_{n-2}^Tv_{n-2}\\&+A^{n-(n-1)}K_{n-1}^Tv_{n-1}+A^{n-n}K_n^Tv_n)\\&=\sum_{m=1}^nQ_nA^{n-m}K_m^Tv_m\end{aligned}
$$
然后我们来看一下A矩阵是什么，论文中定义了A是一个可对角化的矩阵，具体定义为：
$$
A=\Lambda(\gamma e^{i\theta})\Lambda^{-1}
$$
![1747962877437](RetNet.assets\1747962877437.png)
$$
e^{ix}=\cos x+i\sin x
$$
![1747962949397](RetNet.assets\1747962949397.png)
$$
\theta = [\theta_1, \theta_2, \ldots, \theta_{d-1}, \theta_d]
$$
![1747963054932](RetNet.assets\1747963054932.png)
$$
e^{i\theta}=[\cos\theta_1,\sin\theta_2,\ldots,\cos\theta_{d-1},\sin\theta_d]
$$
![1747963115557](RetNet.assets\1747963115557.png)

现在我们知道了矩阵A的构成就能得到：
$$
A^{n-m}=(\Lambda(\gamma e^{i\theta})\Lambda^{-1})^{n-m}
$$
![1747963240422](RetNet.assets\1747963240422.png)
$$
A^{n-m}=\Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}
$$
然后我们回到计算On的公式：
$$
\begin{aligned}o_{n}&=\sum_{m=1}^nQ_nA^{n-m}K_m^Tv_m\\&=\sum_{m=1}^nQ_n(\Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1})K_m^Tv_m\\&=\sum_{m=1}^nX_nW_Q\Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}(X_mW_K)^Tv_m\\&=\sum_{m=1}^nX_nW_Q\Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}W_K^TX_m^Tv_m\end{aligned}
$$
![1747963397741](RetNet.assets\1747963397741.png)
$$
\begin{aligned}o_{n}&=\sum_{m=1}^nQ_n(\gamma e^{i\theta})^{n-m}K_m^Tv_m\\&=\sum_{m=1}^nQ_n(\gamma e^{i\theta})^n(\gamma e^{i\theta})^{-m}K_m^Tv_m\\&=\sum_{m=1}^nQ_n(\gamma e^{i\theta})^n(K_m(\gamma e^{i\theta})^{-m})^Tv_m\\&=\sum_{m=1}^nQ_n(\gamma^ne^{in\theta})(K_m(\gamma^{-m}e^{i(-m)\theta}))^Tv\end{aligned}
$$
![1747963524744](RetNet.assets\1747963524744.png)
$$
o_{n}=\sum_{m=1}^nQ_n(\gamma^ne^{in\theta})(K_m(\gamma^{-m}e^{i(-m)\theta}))^Tv_m\\=\sum_{m=1}^n\gamma^{n-m}(Q_ne^{in\theta})(K_me^{i(-m)\theta})^Tv_m
$$
![1747963682367](RetNet.assets\1747963682367.png)
$$
e^{i(-m)\theta} = [\cos -m\theta_1, \sin -m\theta_2, \ldots, \cos -m\theta_{d-1}, \sin -m\theta_d]
$$

$$
e^{i(-m)\theta}=[\cos-m\theta_{1},\sin-m\theta_{2},\ldots,\cos-m\theta_{d-1},\sin-m\theta_{d}]\\=[\cos m\theta_{1},-\sin m\theta_{2},\ldots,\cos m\theta_{d-1},-\sin m\theta_{d}]
$$

### 复数向量相乘——参考

论文：[RoFormer: Enhanced Transformer with Rotary Position Embedding     ](http://arxiv.org/abs/2104.09864)

链接：https://arxiv.org/pdf/2104.09864

![1747978666882](RetNet.assets\1747978666882.png)

转为复数形式表示就是：
$$
e^{i(-m)\theta} = [\cos m\theta_1 - i \sin m\theta_2, \ldots, \cos m\theta_{d-1} - i \sin m\theta_d]
$$
![1747963955260](RetNet.assets\1747963955260.png)

![1747963991744](RetNet.assets\1747963991744.png)

所以可得：
$$
o_{n}=\sum_{m=1}^n\gamma^{n-m}(Q_ne^{in\theta})(K_me^{i(-m)\theta})^Tv_m\\=\sum_{m=1}^n\gamma^{n-m}(Q_ne^{in\theta})(K_me^{im\theta})^\dagger v_m
$$
![1747964082453](RetNet.assets\1747964082453.png)

## **Retention 的训练并行表示**

首先回顾单个时间步n的输出On的计算公式如下：
$$
o_{n}=\sum_{m=1}^{n}\gamma^{n-m}(Q_{n}e^{in\theta})(K_{m}e^{im\theta})^{\dagger}v_{m}
$$
而所有时间步的输出是可以并行计算的，用矩阵形式表达如下：
$$
((Q\odot\Theta)(K\odot\bar{\Theta})^T\odot D)V
$$
![1747964579409](RetNet.assets\1747964579409.png)
$$
Retention(X)=(QK^T\odot D)V
$$

## **Retention 的推理循环表示**

推理阶段的循环表示论文中定义如下:
$$
S_n=\gamma S_{n-1}+K_n^TV_n
$$
怎么理解呢，还是先回顾单个时间步n的输出On的计算公式：
$$
\begin{aligned}o_{n}&=\sum_{m=1}^n\gamma^{n-m}(Q_ne^{in\theta})(K_me^{in\theta})^\dagger v_m\\&=Q_ne^{in\theta}(\sum_{m=1}^n\gamma^{n-m}(K_me^{in\theta})^\dagger v_m)\\&=Q_ne^{in\theta}(\gamma^{n-n}(K_ne^{in\theta})^\dagger v_n+\sum_{m=1}^{n-1}\gamma^{n-m}(K_me^{im\theta})^\dagger v_m)\\&=Q_ne^{in\theta}((K_ne^{in\theta})^\dagger v_n+\sum_{m=1}^{n-1}\gamma^{n-m}(K_me^{in\theta})^\dagger v_m)\\&=Q_ne^{in\theta}((K_ne^{in\theta})^\dagger v_n+\gamma(K_{n-1}e^{i(n-1)\theta})^\dagger v_{n-1}+\sum_{m=1}^{n-2}\gamma^{n-m}(K_me^{im\theta})^\dagger v_m)\\&=Q_ne^{in\theta}((K_ne^{in\theta})^\dagger v_n+\gamma((K_{n-1}e^{i(n-1)\theta})^\dagger v_{n-1}+\sum_{m=1}^{n-2}\gamma^{n-m-1}(K_me^{in\theta})^\dagger v_m)\end{aligned}
$$

$$
S_n=\gamma S_{n-1}+K_n^TV_n
$$

$$
Retention( X_n) = Q_nS_n, n= 1, \ldots , | x|
$$

上述公式最后一步和推理阶段循环表示公式中各个元素的对应关系是：
$$
Q_{n}=Q_{n}e^{in\theta}
$$

$$
S_{n-1}=(K_{n-1}e^{i(n-1)\theta})^\dagger v_{n-1}+\sum_{m=1}^{n-2}\gamma^{n-m-1}(K_m e^{im\theta})^\dagger v_m
$$

$$
K_n^TV_n=(K_ne^{in\theta})^\dagger v_n
$$

![1747967967584](RetNet.assets\1747967967584.png)



![1747968039924](RetNet.assets\1747968039924.png)

## **Chunkwise递归表示**

![1747972742012](RetNet.assets\1747972742012.png)

训练我们在训练过程中使用平行(等式(5))和分块递归(等式(7))表示。序列或块内的并行化有效地利用GPU来加速计算。更有利的是，分块递归对于长序列训练特别有用，这在FLOPs和内存消耗方面都是有效的。

## **Gated Multi-Scale Retention**

![1747972849270](RetNet.assets\1747972849270.png)
$$
\gamma=1-2^{-5-arange(0,h)}\in R^h
$$

$$
head_i=Retention(X,\gamma_i)
$$

$$
Y = GroupNorm_h(Concat(head_1, \ldots, head_h))
$$

$$
MSR(X)=(swish(XW_G)\odot Y)W_O
$$

![1747973105403](RetNet.assets\1747973105403.png)

![1747973136617](RetNet.assets\1747973136617.png)

## **Retention Score 标准化**

![1747973223922](RetNet.assets\1747973223922.png)

## **Retention 网络总体结构**

![1747973316077](RetNet.assets\1747973316077.png)
$$
Y^{l}=MSR(LN(X^{l}))+X^{l}
$$

$$
X^{l+1}=FFN(LN(Y^l))+Y^l
$$

![1747973434379](RetNet.assets\1747973434379.png)

## 计算

链接：https://zhuanlan.zhihu.com/p/654411874