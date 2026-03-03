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

$$
\begin{array}{|l|} \hline \textbf{Algorithm 1: Stochastic Reverse Sampler (DDPM-like)} \\ \hline \text{For input sample } x_t, \text{ and timestep } t, \text{ output:} \\ \\ \quad \hat{x}_{t-\Delta t} \leftarrow \mu_{t-\Delta t}(x_t) + \mathcal{N}(0, \sigma_q^2 \Delta t) \hfill (15) \\ \hline \end{array}
$$
这个算法成立的基础是：
**Claim 1 (Informal).** 令 $p_{t-\Delta t}(x)$ 为 $\mathbb{R}^d$ 上任意足够平滑的密度函数。考虑 $(x_{t-\Delta t}, x_t)$ 的联合分布，其中 $x_{t-\Delta t} \sim p_{t-\Delta t}$ 且 $x_t \sim x_{t-\Delta t} + \mathcal{N}(0, \sigma_q^2 \Delta t)$。那么，对于足够小的 $\Delta t$，下式成立。对于所有条件变量 $z \in \mathbb{R}^d$，存在 $\mu_z$ 使得：

$$p(x_{t-\Delta t} \mid x_t = z) \approx \mathcal{N}(x_{t-\Delta t}; \mu_z, \sigma_q^2 \Delta t) \tag{16}$$

其中常数 $\mu_z$ 仅取决于 $z$。此外，取以下定义即可：

$$
\begin{aligned}
\mu_z &:= \mathbb{E}_{(x_{t-\Delta t}, x_t)} [x_{t-\Delta t} \mid x_t = z] & (17) \\
&= z + (\sigma_q^2 \Delta t) \nabla \log p_t(z), & (18)
\end{aligned}
$$

其中 $p_t$ 是 $x_t$ 的边缘分布。

**Proof of Claim 1 (Informal).** 这里有一个启发式论证，说明为什么“分数（score）”会出现在反向过程中。我们基本上只需应用贝叶斯定理，然后进行适当的泰勒展开。我们从贝叶斯定理开始：
$$p(x_{t-\Delta t} | x_t) = \frac{p(x_t | x_{t-\Delta t}) p_{t-\Delta t}(x_{t-\Delta t})}{p_t(x_t)} \tag{19}$$

然后对两边取对数。在整个过程中，我们将舍弃对数中的任何加性常数（它们会转化为归一化因子），并舍弃所有 $O(\Delta t)$ 阶的项。注意，在此推导中，我们应将 $x_t$ 视为常数，因为我们想要了解作为 $x_{t-\Delta t}$ 函数的条件概率。现在：

$$
\begin{aligned}
\log p(x_{t-\Delta t} | x_t) &= \log p(x_t | x_{t-\Delta t}) + \log p_{t-\Delta t}(x_{t-\Delta t}) - \cancel{\log p_t(x_t)}  \qquad \text{由于可以去掉常量}  \\
&= \log p(x_t | x_{t-\Delta t}) + \log p_t(x_{t-\Delta t}) + \mathcal{O}(\Delta t) \\ &\text{用泰勒展开舍去极小项：}p_{t-\Delta t}(\cdot) = p_t(\cdot) + \Delta t 
\frac{\partial}{\partial t} p_t(\cdot) \\
&= -\frac{1}{2\sigma_q^2 \Delta t} \|x_{t-\Delta t} - x_t\|_2^2 + \log p_t(x_{t-\Delta t}) \\
&\text{将一阶线性展开拓展到多维：} f(x) \approx f(a) + f'(a) \cdot (x - a) \\

&= -\frac{1}{2\sigma_q^2 \Delta t} \|x_{t-\Delta t} - x_t\|_2^2 \\
&\quad + \cancel{\log p_t(x_t)} + \langle \nabla_x \log p_t(x_t), (x_{t-\Delta t} - x_t) \rangle + \mathcal{O}(\Delta t) \\
&= -\frac{1}{2\sigma_q^2 \Delta t} \left( \|x_{t-\Delta t} - x_t\|_2^2 - 2\sigma_q^2 \Delta t \langle \nabla_x \log p_t(x_t), (x_{t-\Delta t} - x_t) \rangle \right) \qquad \text{由完全平方公式}\\
&= -\frac{1}{2\sigma_q^2 \Delta t} \|x_{t-\Delta t} - x_t - \sigma_q^2 \Delta t \nabla_x \log p_t(x_t)\|_2^2 + C \\
&= -\frac{1}{2\sigma_q^2 \Delta t} \|x_{t-\Delta t} - \mu\|_2^2
\end{aligned}
$$

除了加性因子外，这与均值为 $\mu$、方差为 $\sigma_q^2 \Delta t$ 的正态分布的对数密度相同。因此：
$$p(x_{t-\Delta t} \mid x_t) \approx \mathcal{N}(x_{t-\Delta t}; \mu, \sigma_q^2 \Delta t). \tag{20}$$

回顾这一推导过程，其核心思想是：对于足够小的 $\Delta t$，反向过程 $p(x_{t-\Delta t} \mid x_t)$ 的贝叶斯展开主要由前向过程中的 $p(x_t \mid x_{t-\Delta t})$ 项主导。这在直觉上解释了为什么反向过程和前向过程具有相同的函数形式（此处均为高斯分布）。

### 伪代码

$$
\begin{array}{|l|}
\hline
\textbf{Pseudocode 1: DDPM train loss} \\
\hline
\textbf{Input:} \text{Neural network } f_\theta; \text{ Sample-access to target distribution } p. \\
\textbf{Data:} \text{Terminal variance } \sigma_q; \text{ step-size } \Delta t. \\
\textbf{Output:} \text{Stochastic loss } L_\theta \\
1 \quad x_0 \leftarrow \text{Sample}(p) \\
2 \quad t \leftarrow \text{Unif}[0, 1] \\
3 \quad x_t \leftarrow x_0 + \mathcal{N}(0, \sigma_q^2 t) \\
4 \quad x_{t+\Delta t} \leftarrow x_t + \mathcal{N}(0, \sigma_q^2 \Delta t) \\
5 \quad L \leftarrow \|f_\theta(x_{t+\Delta t}, t + \Delta t) - x_t\|_2^2 \\
6 \quad \textbf{return } L_\theta \\
\hline
\end{array}
$$

$$
\begin{array}{|l|}
\hline
\textbf{Pseudocode 2: DDPM sampling (Code for Algorithm 1)} \\
\hline
\textbf{Input:} \text{Trained model } f_\theta. \\
\textbf{Data:} \text{Terminal variance } \sigma_q; \text{ step-size } \Delta t. \\
\textbf{Output:} x_0 \\
1 \quad x_1 \leftarrow \mathcal{N}(0, \sigma_q^2) \\
2 \quad \textbf{for } t = 1, (1 - \Delta t), (1 - 2\Delta t), \dots, \Delta t \textbf{ do} \\
3 \quad \quad \eta \leftarrow \mathcal{N}(0, \sigma_q^2 \Delta t) \\
4 \quad \quad x_{t-\Delta t} \leftarrow f_\theta(x_t, t) + \eta \\
5 \quad \textbf{end} \\
6 \quad \textbf{return } x_0 \\
\hline
\end{array}
$$

$$
\begin{array}{|l|}
\hline
\textbf{Pseudocode 3: DDIM sampling (Code for Algorithm 2)} \\
\hline
\textbf{Input:} \text{Trained model } f_\theta \\
\textbf{Data:} \text{Terminal variance } \sigma_q; \text{ step-size } \Delta t. \\
\textbf{Output:} x_0 \\
1 \quad x_1 \leftarrow \mathcal{N}(0, \sigma_q^2) \\
2 \quad \textbf{for } t = 1, (1 - \Delta t), (1 - 2\Delta t), \dots, \Delta t, 0 \textbf{ do} \\
3 \quad \quad \lambda \leftarrow \frac{\sqrt{t}}{\sqrt{t-\Delta t} + \sqrt{t}} \\
4 \quad \quad x_{t-\Delta t} \leftarrow x_t + \lambda (f_\theta(x_t, t) - x_t) \\
5 \quad \textbf{end} \\
6 \quad \textbf{return } x_0 \\
\hline
\end{array}
$$

