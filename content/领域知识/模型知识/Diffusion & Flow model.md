> [!NOTE] Reference
> 参考了[Step-by-Step Diffusion: An Elementary Tutorial | PDF](https://arxiv.org/pdf/2406.08929)，[Flow Matching Guide and Code | PDF](https://arxiv.org/pdf/2412.06264)，[An Introduction to Flow Matching and Diffusion Models | PDF](https://arxiv.org/pdf/2506.02070)

# Diffusion model
生成式模型的**目标**是：

> 从多个在未知分布$p^*(x)$中独立同分布的数据中，提取出一个新的样本，是从分布$p^*$中提取的

Diffusion model也一样，它想从目标分布（如狗的图像）$p^*$中采样出一个点。但是从初始的简单分布（如高斯分布）中推导到复杂的目标分布根本不可能。所以Diffusion先反过来，研究怎么将一个多峰的复杂分布简化为一个简单的正态分布，再反过来研究怎么反向降噪，实现对复杂分布的采样。

## Notation

在讲解数学原理前，先详细解释一下数学标号：
$$
\text{将}t\in[0,1]\text{视为一个中间值，具体来说为}(0,\Delta t,2\Delta t,\cdots , T\Delta t), \qquad \text{其中}\Delta t = \frac{1}{T}
$$
因为统一不同分步数量而导致的公式差异，所以将$t$约束为在0到1之间的连续的数。

## Gaussian Diffusion

Gaussian Diffusion作为一个正向Diffusion过程，假定$x_0$作为一个在$\mathbb{R}^d$中随机变量，然后构建一系列随机变量$x_0,x_{\Delta t}, x_{2\Delta t}, \cdots, x_{t-\Delta t}$，用$x_t$表示在离散时间$t$的$x$，有：
$$
x_{t+\Delta t} := x_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma^2)
$$
对此，由于$\text{总方差} = (\text{总步数}) \times (\text{单步方差}) = \left(\frac{1}{\Delta t}\right) \times \sigma^2$，有$x_t$的分布：
$$
x_t \sim \mathcal{N}(x_0, \frac{1}{\Delta t} \sigma^2)
$$
为了让最终方差与最终设定的噪音强度相等，所以要求每一步的$\sigma = \sigma_q \sqrt{\Delta t}$，因此，Gaussian Diffusion的最终表述为：
$$
x_{t+\Delta t} := x_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma_q^2 \Delta t)
$$
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20251025160740.png)
这张图展示了Gaussian Diffusion由一个多峰分布最终转化为一个正态分布的过程。

## 反向采样

有了怎么从一个目标分布到正态分布的正向过程，现在追求通过构建一个反向采样器来达到这个目的：给定一系列边际分布 $p_t$，步骤 t 的反向采样器是一个潜在的随机函数$F_t$，这样如果 $x_t \sim p_t$，则 $F_t(x_t)$ 的边际分布恰好是 $p_{t-1}$：
$$
\{F_t(z):z\sim p_t\} \equiv p_{t-1} 
$$
为了证明这个反向采样是存在且可行的，先用一个不严谨的例子给一点感觉（intuition）：
对于较小的 σ 和正向过程中定义的高斯扩散过程，条件分布$p(x_{t-1}|x_t)$本身接近高斯分布。也就是说，对于所有时间 t 和条件 $z \in \mathbb{R}^d$，存在一些平均参数 $\eta \in \mathbb{R}^d$ 使得:
$$
p(x_{t-1}|x_t=z)\approx \mathcal{N}(x_{t-1}; \eta, \sigma^2)
$$

![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20251025161523.png)
### 随机采样DDPM成立性证明
