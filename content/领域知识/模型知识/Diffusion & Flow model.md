---
date: 2026-02-14
lastmod: 2026-03-16
---
> [!NOTE] TL;DR
> 本文讲解了 Diffusion 和 Flow model 的数学原理，包括 Diffusion 中的 DDPM 和 DDIM 模型。整体来说，DDPM 至 DDIM 再到 Flow model 呈现清晰的渐进发展的趋势，体现为逐渐舍弃带有随机性的高斯分布的影响，而是在反向过程和正向过程中引入确定性的步骤。推荐将三者作为一个脉络研究。
> 本文主要从概率角度理解，缺少了 ODE 和 SDE 的视角，未来如果有机会可以考虑补上。

> [!NOTE] Reference
> 主要参考了[Step-by-Step Diffusion: An Elementary Tutorial | PDF](https://arxiv.org/pdf/2406.08929)。
> 同时可以参阅[Flow Matching Guide and Code | PDF](https://arxiv.org/pdf/2412.06264)，[An Introduction to Flow Matching and Diffusion Models | PDF](https://arxiv.org/pdf/2506.02070)。
> 在构建文档的时候部分参考了[一文贯通Diffusion原理：DDPM、DDIM和Flow Matching](https://zhuanlan.zhihu.com/p/12591930520)这篇知乎文章的汉化。

# Diffusion model
生成式模型的**目标**是：

> 从多个在未知分布$p^*(x)$中独立同分布的数据中，提取出一个新的样本，是从分布$p^*$中提取的

Diffusion model也一样，它想从目标分布（如狗的图像）$p^*$中采样出一个点。但是从初始的简单分布（如高斯分布）中推导到复杂的目标分布难以实现。所以Diffusion先反过来，研究怎么将一个多峰的复杂分布简化为一个简单的正态分布，再反过来研究怎么反向降噪，实现对复杂分布的采样。

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
同理：
$$
x_t \sim \mathcal{N}(x_0, \sigma_t^2), \qquad \text{where} \, \sigma_t := \sigma_q \sqrt{t}
$$
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20251025160740.png)
这张图展示了Gaussian Diffusion由一个多峰分布最终转化为一个正态分布的过程。

## 反向采样

有了怎么从一个目标分布到正态分布的正向过程，现在追求通过构建一个反向采样器来达到这个目的：给定一系列边际分布 $p_t$，步骤 t 的反向采样器是一个潜在的随机函数$F_t$，这样如果 $x_t \sim p_t$，则 $F_t(x_t)$ 的边际分布恰好是 $p_{t-1}$：
$$
\{F_t(z):z\sim p_t\} \equiv p_{t-1} 
$$
为了证明这个反向采样是存在且可行的，先用一个不严谨的例子给一点感觉（intuition）：
对于较小的 σ 和正向过程中定义的高斯扩散过程，条件分布$p(x_{t-1}|x_t)$本身接近高斯分布。也就是说，对于所有时间 t 和条件 $z \in \mathbb{R}^d$，存在一些平均参数 $\mu \in \mathbb{R}^d$ 使得:
$$
p(x_{t-1}|x_t=z)\approx \mathcal{N}(x_{t-1}; \mu, \sigma^2) \tag{12}
$$

> [!NOTE] 注
> $p(x_t)$和$x_t$的分布是一个东西

![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20251025161523.png)
### 随机采样DDPM成立性证明

$$
\begin{array}{|l|} \hline \textbf{Algorithm 1: Stochastic Reverse Sampler (DDPM-like)} \\ \hline \text{For input sample } x_t, \text{ and timestep } t, \text{ output:} \\ \\ \quad \hat{x}_{t-\Delta t} \leftarrow \mu_{t-\Delta t}(x_t) + \mathcal{N}(0, \sigma_q^2 \Delta t) \qquad \qquad \qquad \quad (15)
\\ \hline \end{array}
$$
这个算法成立的基础是：
**Claim 1 (Informal).** 令 $p_{t-\Delta t}(x)$ 为 $\mathbb{R}^d$ 上任意足够平滑的密度函数。考虑 $(x_{t-\Delta t}, x_t)$ 的联合分布，其中 $x_{t-\Delta t} \sim p_{t-\Delta t}$ 且 $x_t \sim x_{t-\Delta t} + \mathcal{N}(0, \sigma_q^2 \Delta t)$。那么，对于足够小的 $\Delta t$，下式成立。对于所有条件变量 $z \in \mathbb{R}^d$，存在 $\mu_z$ 使得：

$$
p(x_{t-\Delta t} \mid x_t = z) \approx \mathcal{N}(x_{t-\Delta t}; \mu_z, \sigma_q^2 \Delta t) \tag{16}
$$

其中常数 $\mu_z$ 仅取决于 $z$。此外，取以下定义即可：

$$
\begin{aligned}
\mu_z &:= \mathbb{E}_{(x_{t-\Delta t}, x_t)} [x_{t-\Delta t} \mid x_t = z] & (17) \\
&= z + (\sigma_q^2 \Delta t) \nabla \log p_t(z) & (18)
\end{aligned}
$$
其中 $p_t$ 是 $x_t$ 的边缘分布。

> [!NOTE] 注意
> $\nabla\log{p_t(z)}$  是Score函数，由Tweedie 公式，可以转化为 
> $$
>\nabla_{x_t} \log p_t(x_t) = \frac{\mathbb{E}[x_0 \mid x_t] - x_t}{\sigma_t^2}
>$$
> 此处由神经网络等模型拟合。

**Proof of Claim 1 (Informal).** 这里有一个启发式论证，说明为什么“分数（score）”会出现在反向过程中。我们基本上只需应用贝叶斯定理，然后进行适当的泰勒展开。我们从贝叶斯定理开始：
$$
p(x_{t-\Delta t} | x_t) = \frac{p(x_t | x_{t-\Delta t}) p_{t-\Delta t}(x_{t-\Delta t})}{p_t(x_t)} \tag{19}
$$

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
$$
p(x_{t-\Delta t} \mid x_t) \approx \mathcal{N}(x_{t-\Delta t}; \mu, \sigma_q^2 \Delta t). \tag{20}
$$

回顾这一推导过程，其核心思想是：对于足够小的 $\Delta t$，反向过程 $p(x_{t-\Delta t} \mid x_t)$ 的贝叶斯展开主要由前向过程中的 $p(x_t \mid x_{t-\Delta t})$ 项主导。这在直觉上解释了为什么反向过程和前向过程具有相同的函数形式（此处均为高斯分布）。

### 训练目标
如此，可以把复杂的生成问题转为一个回归问题，只需要学习$p(x_{t-\Delta t}|x_t)$即可:
$$
\begin{aligned}
	\mu_{t-\Delta t}(z) &:= \mathbb{E}[x_{t-\Delta t} \mid x_t = z] \\
\implies \mu_{t-\Delta t} &= \mathop{\mathrm{argmin}}_{f: \mathbb{R}^d \to \mathbb{R}^d} \mathbb{E}_{x_t, x_{t-\Delta t}} \|f(x_t) - x_{t-\Delta t}\|_2^2\\
&= \mathop{\mathrm{argmin}}_{f: \mathbb{R}^d \to \mathbb{R}^d} \mathbb{E}_{x_{t-\Delta t}, \eta} \|f(x_{t-\Delta t} + \eta_t) - x_{t-\Delta t})\|_2^2,
\end{aligned}
$$
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
这些算法理论上成立，但是用神经网络拟合的时候，$p(x_{t - \Delta t}|x_t)$主要是高斯噪音，模型分不清哪些是要生成的特征，哪些是噪音。将训练目标改为预测$\mathbb{E}[x_0|x_t]$可以有效减小方差（等效地估计所有先前噪声步骤的平均值，而不是估计单个噪声步骤，方差小得多）。
由于前向过程中每一步的噪音都是相互独立的，单步噪音是总噪音的$\frac{\Delta t}{t}$，由此有：
$$
	\mathbb{E}[(x_{t-\Delta t}-x_t)|x_t] =  \frac{\Delta t}{t} \mathbb{E}[(x_0-x_t)|x_t]
$$
或等价的：
$$
\mathbb{E}[x_{t-\Delta t} \mid x_t] = \frac{\Delta t}{t}\,\mathbb{E}[x_0 \mid x_t] + \left(1 - \frac{\Delta t}{t}\right) x_t
$$
但是Diffusion model的工作原理没变，只是预测每一小步的噪音。

### 确定性DDIM正确性证明

但是DDPM因为要细分成很多步，而每步都需要通过神经网络预测方向，这导致速度奇慢。这也提出了新的问题：
$$
\text{能否找到不同的反向采样函数}\, F_t(z)\,\, \text{减少神经网络的调用？}
$$
因此提出了确定性的DDIM算法：
$$
\begin{array}{|l|}
\hline
\textbf{Algorithm 2: Deterministic Reverse Sampler (DDIM-like)} \\
\hline
\text{For input sample } x_t \text{, and step index } t \text{, output:} \\
\\
\quad \quad \widehat{x}_{t-\Delta t} \leftarrow x_t + \lambda(\mu_{t-\Delta t}(x_t) - x_t) \quad \quad \quad \quad \quad \quad \quad (33) \\
\\
\text{where } \lambda := \left( \frac{\sigma_t}{\sigma_{t-\Delta t} + \sigma_t} \right) \text{ and } \sigma_t \equiv \sigma_q \sqrt{t} \text{ from Equation (12).} \\
\hline
\end{array}
$$
注意：$\lambda$ 是一个缩放系数，这个算法的逻辑是**新位置 = 老位置 + 步长 $\times$ 移动方向**

想要证明这个反向采样器是正确的，因为它是一个确定性的采样器，用类似DDPM那种从 $p(x_{x-\Delta t}| x_t)$ 随机采样是行不通的。只能通过证明**这个方程表示了一个有效的映射，在边际分布 $p_t$ 和$p_{t-\Delta t}$ 间**。

这个证明等价于对于：
$$
\begin{aligned}
F_t(z) :&= z + \lambda(\mu_{t-\Delta t}(z) - z) \\ \phantom{F_t(z)} &= z + \lambda(\mathbb{E}[x_{t-\Delta t} \mid x_t = z] - z)
\end{aligned}
$$
我们想证明：
$$
\{F(x)\}_{x\sim p_t} =F_t♯p_t \approx p_{t-\Delta t}
$$

> [!NOTE] 注
> - $\eta(z)$和DDPM中的神经网络一致；
> - $G_t(z), \quad z \in \mathbb{R}^d \to \mathbb{R}^d$ 是一个从向量映射到向量的函数，分布中每一个向量通过映射获得新的向量集构成新的分布；
> - 如果 $z \sim p_t$ 那么 $G_t(z)$ 的分布就是 pushforward 测度 $G_t \sharp p_t$；
> - 此部分设计测度论，如果想深入了解可以参阅测度论的教材。

#### Case1: Single point
让我们首先尝试目标分布 $p_0$ 是 $R^d$ 中的单点质量的简单情况。不失一般性，我们可以假定那个点是 $x_0=0$。为了验证DDIM算法是准确的，我们希望考虑任意步长 $t$ 下 $x_t$ 和 $x_{t-\Delta t}$ 的分布。根据扩散前向过程，在时刻 $t$ 相关的随机变量为：
$$
\begin{aligned}
x_0 &= 0 \quad \text{（确定性）} \\
x_{t-\Delta t} &\sim \mathcal{N}(x_0, \sigma_{t-\Delta t}^2) \\
x_t &\sim \mathcal{N}(x_{t-\Delta t}, \sigma_t^2 - \sigma_{t-\Delta t}^2)
\end{aligned}
$$
$x_{t-\Delta t}$ 的边缘分布是 $p_{t-\Delta t} = \mathcal{N}(0, \sigma_{t-\Delta t}^2)$，而 $x_t$ 的边缘分布是 $p_t = \mathcal{N}(0, \sigma_t^2)$。

让我们首先寻找某个确定性函数 $G_t : \mathbb{R}^d \to \mathbb{R}^d$，使得 $G_t \sharp p_t = p_{t-\Delta t}$。虽然有许多可能的函数可行，但最明显的一个是：
$$
G_t(z) := \left( \frac{\sigma_{t-\Delta t}}{\sigma_t} \right) z. \tag{37}
$$
上述函数 $G_t$  简单地重新缩放 $p_t$ 的高斯分布，以匹配 $p_{t-\Delta t}$ 高斯分布的方差。事实证明，这个 $G_t$ 正好等价于算法 2 所采取的步骤 $F_t$，我们接下来将展示这一点。

断言 3. 当目标分布是一个点质量 $p_0 = \delta_0$ 时，更新 $F_t$（如公式 35 所定义）等价于缩放 $G_t$（如公式 37 所定义）：
$$
F_t \equiv G_t. \tag{38}
$$
因此，算法 2 定义了针对目标分布 $p_0 = \delta_0$ 的反向采样器。
证明. 要应用 F_t，我们需要为我们的简单分布计算 $\mathbb{E}[x_{t-\Delta t} \mid x_t]$。由于 $(x_{t-\Delta t}, x_t)$ 是联合高斯分布，因此有：
$$
\mathbb{E}[x_{t-\Delta t} \mid x_t] = \left( \frac{\sigma_{t-\Delta t}^2}{\sigma_t^2} \right) x_t. \tag{39}
$$
其余部分即是代数运算：
$$
\begin{aligned}
F_t(x_t) &:= x_t + \lambda (\mathbb{E}[x_{t-\Delta t} \mid x_t] - x_t) \\
&= x_t + \left( \frac{\sigma_t}{\sigma_{t-\Delta t} + \sigma_t} \right) (\mathbb{E}[x_{t-\Delta t} \mid x_t] - x_t) \\
&= x_t + \left( \frac{\sigma_t}{\sigma_{t-\Delta t} + \sigma_t} \right) \left( \frac{\sigma_{t-\Delta t}^2}{\sigma_t^2} - 1 \right) x_t \\
&= \left( \frac{\sigma_{t-\Delta t}}{\sigma_t} \right) x_t \\
&= G_t(x_t).
\end{aligned}
$$
因此我们得出结论：算法 2 是一个正确的反向采样器，因为它等价于 $G_t$，且 $G_t $是有效的。
$\square$

即使 $x_0$ 是一个任意点而不是 $x_0 = 0$，算法 2 的正确性依然成立，因为所有事物都具有转移对称性。

可以把DDIM的更新视作速度场，使在 $t$ 时刻的点向他们 $t-\Delta t$ 时刻的位置移动，具体来说，可以把向量场定义为：
$$
v_t(x_t):=\frac{\lambda}{\Delta t}(\mathbb{E}[x_{t-\Delta t}|x_t]-x_t) \tag{40}
$$
于是DDIM更新可以写作：
$$
\begin{aligned}
\hat{x}_{t-\Delta t} :&= x_t + \lambda(\eta_{t-\Delta t}(x_t)-x_t) \\
&= x_t + v_t(x_t)\Delta t
\end{aligned}
$$
#### Case2: Two Points
现在让我们证明当目标分布是两点混合时算法 2 是正确的：
$$
p_0:=\frac 12 \delta_a+ \frac 12 \delta_b \tag{43}
$$
根据扩散的前向过程，在时间 $t$ 的分布是一个混合高斯分布：
$$
p_t:=\frac{1}{2}N(a,\sigma_t^2)+\frac{1}{2}N(b,\sigma_t^2)\tag{44}
$$
我们想要证明的是式(40)中的速度场 $v_t$ 可以完成分布转换：$p_t\overset{v_t}{→}p_{t-\Delta t}$。

首先我们尝试构建一个满足反向采样 $p_t\overset{v^*_t}{→}p_{t-\Delta t}$ 的正确的速度场 $v_t^*$。由于 Case1 的结果对单个点是成立的，那么速度场可以将 $\{a,b\}$ 的每个混合分量进行转换。即存在一个速度场 $v_t^{[a]}$：
$$
v^a_t(x_t):=\lambda\underset{x_0\sim \sigma_a}{E}[x_{t-\Delta t}-x_t|x_t] \tag{45}
$$
可以对 $a$ 进行转换：
$$
N(a,\sigma_t^2)\overset{v_t^a}{→}N(a,\sigma_{t-\Delta t}^2) \tag{46}
$$
同理 $b$ 的速度场 $v_t^{[b]}$ 也成立。

为了将两个速度场合并成 $v_t^*$ 来表示：
$$
\underbrace{(\frac{1}{2}N(a,\sigma_t^2)+\frac{1}{2}N(b,\sigma_t^2))}_{p_t}\overset{v_t^*}{→}\underbrace{(\frac{1}{2}N(a,\sigma_{t-\Delta t}^2)+\frac{1}{2}N(b,\sigma_{t-\Delta t}^2))}_{p_{t-\Delta t }} \tag{47}
$$
直观地取速度场的平均是错的，应该是独立速度场的加权：
$$
\begin{aligned}
v_t^*(x_t) &= \frac{v_t^a(x_t)\cdot p(x_t|x_0=a)+v_t^b(x_t)\cdot p(x_t|x_0=b)}{p(x_t|x_0=a)+p(x_t|x_0=b)} \\
&=v_t^a(x_t)\cdotp(x_0=a|x_t)+v_t^b(x^t)\cdot(x_0=b|x_t)
\end{aligned}
\tag{48-49}
$$
其中，$v_t^a$ 的权重是点从 $a$ 中生成的概率，而不是从 $x_0 = b$ 的概率。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260304174840967.png)
以气体为例子，左图中的左部点虽然针对 $b$ 的速度大于 $a$ 的速度（因为离 $b$ 远，需要加速收敛到 $b$ 的分布内），但是对 $b$ 的权重低（离 $b$ 远，在 $b$ 生成的概率就低）。所以最后在右图生成了一个向 $a$ 的速度。

接下来我们需要证明式子 $(40)$ 和 $v_t^*$ 相同，首先考虑单个向量场 $v_t^a$ 可以写作一个条件期望，根据式(45)中的定义 :
$$
\begin{aligned}
v_t^a(x_t)&=\lambda \underset{x_0\sim \delta _a}{E}[x_{t-\Delta t}-x_t|x_t]\\ &=\lambda \underset{x_0\sim 1/2\delta_a+1/2\delta_b}{E}[x_{t-\Delta t}-x_t|x_0=a,x_t]
\end{aligned}
\tag{50-51}
$$
那么 $v_t^*$ 可以写为一个条件期望：
$$
\begin{aligned}
v_t^*(x_t)&=v_t^a(x_t)\cdot p(x_0=a|x_t)+v_t^b(x_t)\cdot p(x_0=b|x_t)\\ 
&=\lambda E[x_{t-\Delta t}-x_t|x_0=a,x_t]\cdot p(x_0=a|x_t)\\
&+\lambda E[x_{t-\Delta t}-x_t|x_0=b,x_t]\cdot p(x_0=b|x_t) \\ 
&=\lambda E[x_{t-\Delta t}-x_t|x_t]
\\&=v_t(x_t)
\end{aligned}
\tag{52-55}
$$
####  Case3: 任意分布
现在我们知道如何处理两个点，我们可以将这个想法推广到 $x_0$ 的任意分布。我们不会在这里详细讨论，因为一般证明将包含在后续部分中。事实证明，我们算法 2 的整体证明策略可以显着推广到其他类型的扩散，而无需太多的工作。这就产生了流匹配的想法，我们将在下一节中看到。一旦我们开发了流机制，实际上就可以直接从方程（37）的简单单点缩放算法直接导出 DDIM：请参见原论文附录 B.5

#### SDE视角下的扩散&概率流常微分方程 
论文中说可选，暂时没读，或许以后还需要深入理解的时候会读。

### DDPM和DDIM
回顾之前提出的公式和算法：
$$
\begin{array}{|l|} \hline \textbf{Algorithm 1: Stochastic Reverse Sampler (DDPM-like)} \\ \hline \text{For input sample } x_t, \text{ and timestep } t, \text{ output:} \\ \\ \quad \hat{x}_{t-\Delta t} \leftarrow \mu_{t-\Delta t}(x_t) + \mathcal{N}(0, \sigma_q^2 \Delta t) \qquad \qquad \qquad \quad (15)
\\ \hline \end{array}
$$
$$
\begin{array}{|l|}
\hline
\textbf{Algorithm 2: Deterministic Reverse Sampler (DDIM-like)} \\
\hline
\text{For input sample } x_t \text{, and step index } t \text{, output:} \\
\\
\quad \quad \widehat{x}_{t-\Delta t} \leftarrow x_t + \lambda(\mu_{t-\Delta t}(x_t) - x_t) \quad \quad \quad \quad \quad \quad \quad (33) \\
\\
\text{where } \lambda := \left( \frac{\sigma_t}{\sigma_{t-\Delta t} + \sigma_t} \right) \text{ and } \sigma_t \equiv \sigma_q \sqrt{t} \text{ from Equation (12).} \\
\hline
\end{array}
$$
$$
\begin{aligned}
\mu_z &:= \mathbb{E}_{(x_{t-\Delta t}, x_t)} [x_{t-\Delta t} \mid x_t = z] & (17) \\
&= z + (\sigma_q^2 \Delta t) \nabla \log p_t(z) & (18)
\end{aligned}
$$

前面说到，从DDPM转到DDIM的主要动机是因为DDPM需要过太多次模型 $\mu_{t - \Delta t}$ 了，但是又不能通过减小步数的方法，因为随机的性质导致减小步数会导致崩坏。而DDIM用确定性的方法规避了这个问题，可以通过减小步骤的方法来减小生成的时间。

# Flow model

Flow model 其实是 DDIM 算法的一种泛化，Flow model的核心思想和 DDIM 中介绍的相差不大：
1. 首先，我们定义了如何生成单点。具体来说，我们构建了向量场 $\{v^{[a]}_t\}_t$，当应用于所有时间步时，将标准高斯分布传输到任意 delta 分布 $δ_a$。  
2. 其次，我们确定如何将两个矢量场组合成单个有效矢量场。这让我们可以构建从标准高斯到两个点的传输（或者更一般地说，到点上的分布 - 我们的目标分布）。
其中任意一点都不需要高斯采样，所以可以完全舍弃高斯分布的正向分布，提出Flow model。

## Flow
首先先定义 Flow 是啥：flow是一个随时间演进的向量场的集合 $v=\{v_t\}_{t\in[0,1]}$ ，换成物理概念可以理解为是一个气体在不同时间 $t$ 构成的集合。任何一个flow都定义了一条从初始点 $x_1$ 到最终的点 $x_0$ 的轨迹。
对于 flow $v$ 和初始点 $x_1$，考虑常微分方程(ODE)对应的离散时间迭代 $x_{t-\Delta t} \leftarrow x_t + v_t(x_t)\Delta t$： 
$$
\frac{dx_t}{dt} = -v_t(x_t) \tag{59} 
$$ 
在 $t=1$ 初始点为 $x_1$，将： 
$$
x_t := \text{RunFlow}(v, x_1, t) \tag{60} 
$$
看作 flow ODE 在时间 $t$ 的解，终点是 $x_0$。也就是说 RunFlow 是将 $x_1$ 验着 flow $v$ 移动到时间 $t$ 的结果。 flow 不仅定义了初始点和终点的映射，其实也定义了整个分布的转换。如果 $p_1$ 是初始点的分布，那么经过 flow $v$ 终点的分布是： 
$$
p_0 = \{\text{RunFlow}(v, x_1, t=0)\}_{x_1 \sim p_1} \tag{61} 
$$ 
在这个过程中，还是用气体来比拟，起点即初始状态是分子服从分布 $p_1$ 的气体，然后气体中的每个分子的轨迹由 flow $v$ 来决定，那么这个气体（这些分子）最终的分布是 $p_0$。 Flow Matching 的最终目标是：学习一个 flow $v^*$ 使得 $q \xrightarrow{v^*} p$，其中 $p$ 是目标数据分布，$q$ 是某个简单的基础分布（比如高斯分布）。如果拥有 $v^*$，我们可以这样从 $p$ 生成样本：先从基础分布 $q$ 中采样得到 $x_1$，然后通过得到的 flow 输入是 $x_1$ 输出是最终的 $x_0$。DDIM 算法这类方法的一个 case，现在我们如何构建更通用的 flow 呢？

### Pointwise Flows & Marginal Flows

Flow的最基础单元室单点的flow，将一个点 $x_1$ 移动到 $x_0$。直观上，给定一个联系 $x_1$ 和 $x_0$ 的任意路径 $\{x_t\}_{t \in [0,1]}$，一个 pointwise flow 描述了这个轨迹：由给定速度 $v_t(x_t)$ 下的每个点 $x_t$ 构成，如下图所示：
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260306111131912.png)
正式的表述：一个 pointwise flow 是起点为 $x_1$、终点为 $x_0$ 且满足式(59)的任意 flow $\{v_t\}_t$，记作 $v^{[x_1,x_0]}$。这样的 pointwise flow 有很多，不是唯一的。

有了逐点的流后，我们可以像 DDIM 的双点的证明一样，我们需要一个 flow $v^*$ 来实现分布间的转换，可以采用加权平均的方式，权重是每个分子通过各自的pointwise flow $v_t^{[x_1,x_0]}$ 在 $x_t$ 出现的概率，即 

$$
v_t^*(x_t):=\underset{x_0,x_1|x_t}{E}[v_t^{[x_1,x_0]}(x_t)|x_t] \tag{64}
$$

上面的期望关于联合分布 $(x_1,x_0,x_t)$ 的期望：通过采样 $(x_1,x_0)\sim \prod_{q,p}$ 然后得到 $x_t←\text{RunFLow}(v^{[x_1,x_0]},x_1,t)$ 。

这提出了两个问题：
1. 我们应该选择什么样的pointwise $v^{[x_1,x_0]}$ 和coupling $\prod_{q,p}$
2. 如何计算marginal flow $v^*$ ? 我们无法直接根据式(64)计算，因为需要在给定的 $x_t$ 从 $p(x_0|x_t)$ 中采样，非常复杂。

这个问题的提出是很自然的：在完全舍弃高斯分布带来的随机性后，提出了两个问题：
- 要怎么选取原本正向过程会提供的训练集 $\Pi_{q,p}$，简单的是独立采样，各采各的随机配对。但也可以用更聪明的配对（比如 optimal transport coupling），让轨迹更短、训练更高效。
- 以及 $x_0$ 和 $x_1$ 之间怎么走。最常用的是线性插值 $x_t = (1-t)x_0 + tx_1$，但理论上可以是任意路径。

### Pointwise flow的一种简单选择
考虑简单的情况，我们需要选择简单的 pointwise flow，基础分布 $q$ 和 coupling $\Pi_{q,p}$。虽然高斯分布是一个简单的基础分布但不是唯一的选择，也有其他的，比如的环形分布也是基础分布。

至于 coupling，最简单的选择是独立 coupling，即从 $p$ 和 $q$ 中各自采样。其中 $p$ 和 $q$ 是终点和起始点的分布。

对于 pointwise，最简单的是 linear pointwise flow：

$$
\begin{aligned}
v_t^{[x_1,x_0]}(x_t) &= x_0 - x_1 \\
\implies \text{RunFlow}(v^{[x_1,x_0]}, x_1, t) &= t x_1 + (1 - t) x_0 
\end{aligned}
\tag{65-66}
$$

如上式中，只是在 $x_1$ 和 $x_0$ 之间做了线性插值。linear pointwise flow 的一个 marginal flow 例子如下图所示：基础分布 $q$ 是一个环状的均匀分布，目标分布 $p$ 是位于 $x_0$ 的狄拉克函数。灰色箭头描绘了在不同时间 $t$ 的 flow 场。最左边是基础分布，最右边所有点坍缩到一个点即目标点上。这里 $x_0$ 恰好是前面螺旋分布上的一个点。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260306113414358.png)

剩下的问题是：在这种情况下，如果所有训练点都是随机抽取的，要怎么实现在训练中的加权呢？
我们可以借鉴DDIM的解决方法，用我们可以从联合分布 $(x_0,x_t)$ 中采样足够多的样本，然后作为回归问题来解决。类似DDPM中的处理，式(64)中的条件期望函数可以写作：

$$
\begin{aligned}
v_t^*(x_t)&:=\underset{x_0,x_1|x_t}{E}[v_t^{[x_1,x_0]}(x_t)|x_t]\\ \Longrightarrow v_t^*&=\underset{f:R^d→R^d}{\text{argmin}}\underset{(x_0,x_1,x_t)}{E}\|f(x_t)-v_t^{[x_1,x_0]}(x_t)\|_2^2\\
\end{aligned}
\tag{67-68}
$$

这样，我们就可以把训练集从 $x_t$ 空间中选取点来学习每对 $(x_0, x_1)$ 被采到的概率确实相等。但不同的配对在同一个时间 $t$ 会产生不同位置的 $x_t$。在某些位置，很多条轨迹汇聚经过，网络在那里收到大量不同方向的监督信号。在另一些位置，只有少数轨迹经过，监督信号方向比较一致。

最后模型通过神经网络实现了对加权的拟合，而不是通过建模的方式来实现加权。

$$
\begin{array}{|l|}
\hline
\textbf{Pseudocode 4: } \text{Flow-matching train loss, generic pointwise flow } \color{red}{\text{[or linear flow]}} \\
\hline
\textbf{Input: } \text{Neural network } f_\theta \\
\textbf{Data: } \text{Sample-access to coupling } \Pi_{q,p}; \\
\qquad \quad \text{Pointwise flows } \{v_t^{[x_1,x_0]}\} \text{ for all } x_1, x_0. \\
\textbf{Output: } \text{Stochastic loss } L \\
1 \quad (x_1, x_0) \leftarrow \text{Sample}(\Pi_{q,p}) \\
2 \quad t \leftarrow \text{Unif}[0, 1] \\
3 \quad x_t \leftarrow \underbrace{\text{RunFlow}(v_t^{[x_1,x_0]}, x_1, t)}_{\color{red}{tx_1+(1-t)x_0}} \\
4 \quad L \leftarrow \left\| f_\theta(x_t, t) - \underbrace{v_t^{[x_1,x_0]}(x_t)}_{\color{red}{(x_0-x_1)}} \right\|_2^2 \\
5 \quad \textbf{return } L \\
\hline
\end{array}
$$
$$
\begin{array}{|l|}
\hline
\textbf{Pseudocode 5: } \text{Flow-matching sampling} \\
\hline
\textbf{Input: } \text{Trained network } f_\theta \\
\textbf{Data: } \text{Sample-access to base distribution } q; \text{ step-size } \Delta t. \\
\textbf{Output: } \text{Sample from target distribution } p. \\
1 \quad x_1 \leftarrow \text{Sample}(q) \\
2 \quad \textbf{for } t = 1, (1 - \Delta t), (1 - 2\Delta t), \dots, \Delta t \textbf{ do} \\
3 \quad \vert \quad x_{t-\Delta t} \leftarrow x_t + f_\theta(x_t, t)\Delta t \\
4 \quad \textbf{end} \\
5 \quad \textbf{return } x_0 \\
\hline
\end{array}
$$
