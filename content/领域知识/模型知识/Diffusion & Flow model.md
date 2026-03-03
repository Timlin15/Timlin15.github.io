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
对于较小的 σ 和正向过程中定义的高斯扩散过程，条件分布$p(x_{t-1}|x_t)$本身接近高斯分布。也就是说，对于所有时间 t 和条件 $z \in \mathbb{R}^d$，存在一些平均参数 $\eta \in \mathbb{R}^d$ 使得:
$$
p(x_{t-1}|x_t=z)\approx \mathcal{N}(x_{t-1}; \eta, \sigma^2) \tag{12}
$$

> [!NOTE] 注
> $p(x_t)$和$x_t$的分布是一个东西

![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20251025161523.png)
### 随机采样DDPM成立性证明

$$
\begin{array}{|l|} \hline \textbf{Algorithm 1: Stochastic Reverse Sampler (DDPM-like)} \\ \hline \text{For input sample } x_t, \text{ and timestep } t, \text{ output:} \\ \\ \quad \hat{x}_{t-\Delta t} \leftarrow \mu_{t-\Delta t}(x_t) + \mathcal{N}(0, \sigma_q^2 \Delta t) \tag{15} \\ \hline \end{array}
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
&= z + (\sigma_q^2 \Delta t) \nabla \log p_t(z), & (18)
\end{aligned}
$$

其中 $p_t$ 是 $x_t$ 的边缘分布。

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
\implies \mu_{t-1} &= \mathop{\mathrm{argmin}}_{f: \mathbb{R}^d \to \mathbb{R}^d} \mathbb{E}_{x_t, x_{t-\Delta t}} \|f(x_t) - x_{t-\Delta t}\|_2^2\\
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
> - $\eta(z)$和DDPM中的神经网络一致。
> - $G_t(z), \quad z \in \mathbb{R}^d \to \mathbb{R}^d$ 是一个从向量映射到向量的函数，分布中每一个向量通过映射获得新的向量集构成新的分布。
> - 如果 $z \sim p_t$ 那么 $G_t(z)$ 的分布就是 pushforward 测度 $G_t \sharp p_t$

#### Case1: Single point
让我们首先尝试目标分布 $p_0$ 是 $R^d$ 中的单点质量的简单情况。不失一般性，我们可以假定那个点是 $x_0=0$。为了验证DDIM算法是准确的，我们希望考虑任意步长 $t$ 下 $x_t$ 和 $x_{t-\Delta t}$ 的分布。根据扩散前向过程，在时刻 $t$ 相关的随机变量为：
$$
\begin{aligned}
x_0 &= 0 \quad \text{（确定性）} \\
x_{t-\Delta t} &\sim \mathcal{N}(x_0, \sigma_{t-\Delta t}^2) \\
x_t &\sim \mathcal{N}(x_{t-\Delta t}, \sigma_t^2 - \sigma_{t-\Delta t}^2)
\end{aligned}
$$
$x_{t-\Delta t}$ 的边缘分布是 $p_{t-\Delta t} = \mathcal{N}(0, \sigma_{t-1}^2)$，而 $x_t$ 的边缘分布是 $p_t = \mathcal{N}(0, \sigma_t^2)$。

让我们首先寻找某个确定性函数 $G_t : \mathbb{R}^d \to \mathbb{R}^d$，使得 $G_t \sharp p_t = p_{t-\Delta t}$。虽然有许多可能的函数可行，但最明显的一个是：
$$
G_t(z) := \left( \frac{\sigma_{t-\Delta t}}{\sigma_t} \right) z. \tag{37}
$$
上述函数 $G_t$  simply 重新缩放 $p_t$ 的高斯分布，以匹配 $p_{t-\Delta t}$ 高斯分布的方差。事实证明，这个 $G_t$ 正好等价于算法 2 所采取的步骤 $F_t$，我们接下来将展示这一点。

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
v_t(x_t):=\frac{\lambda}{\Delta t}(\mathbb{E}[x_{t-\Delta t}|x_t]-x_t)
$$
于是DDIM更新可以写作：
$$
\begin{aligned}
\hat{x}_{t-\Delta t} :&= x_t + \lambda(\eta_{t-\Delta t}(x_t)-x_t) \\
&= x_t + v_t(x_t)\Delta t
\end{aligned}
$$
