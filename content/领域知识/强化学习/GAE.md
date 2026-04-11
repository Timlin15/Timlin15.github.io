---
date: 2026-04-08
lastmod: 2026-04-09
---
> [!NOTE] Reference
> [GAE原始论文](http://arxiv.org/abs/1506.02438)
## TD(0) 到 n-step return
对于我们自己选择的 Metric：
$$
J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]
$$
[[Model free 基础方法]]中介绍了两种估计方式：
TD方法，基础的TD方法即 TD(0)，只选取下一步进行单步 Booststrap：
$$
\hat{G}_t^{(1)} = r_t + \gamma V(s_{t+1})
$$
或者进行蒙特卡罗方法估计：
$$
\hat{G}_t^{(\infty)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$
前者高偏差但是低方差，后者低偏差高方差。仅后者是无偏的。

[[Model free 基础方法]]中提到了可以采用多步 TD 的方法来降低方差。TD(λ) 也是为了实现这个目的：结合TD 和 MC 的特点，它以把所有 n-step return 都结合起来：
$$
G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{G}_t^{(n)}
$$
这里的 $(1-\lambda)$ 是一个归一化系数，保证权重之和为 1。直觉上：

- $\lambda = 0$：退化为 TD(0)，只用 $\hat{G}_t^{(1)}$​
- $\lambda = 1$：退化为 Monte Carlo，用完整回报
- $0 < \lambda < 1$：短步的 return 权重更大（因为 $\lambda^{n-1}$ 指数衰减），长步的权重越来越小

## GAE
$TD(\lambda)$ 是用于估计价值函数的方法，GAE则把这个方法与 Actor-Critic方法做了结合。

先给定优化目标，GAE 最初的优化目标是优化下面的 Metric，使之最大化：
$$
J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}  r_t\right]
$$

> [!NOTE] 注
> 这与上文的优化 metric 不同的原因是：首先本文会通过证明证明上述那个 metric 的偏差不会太大；同时实践中很难获得远期的回报值，导致远期方差极大，一般都会加上衰减系数减小其影响。

作为一个轨迹的回报，由于没有储存转移函数等，这个优化目标不能直接求出，需要一个估计器来近似。近似函数一般近似目标函数的梯度，可以直接求出目标函数的增长方向：
$$
g = \mathbb{E}\left[\sum_{t=0}^\infty \Psi_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]
$$
这里 $\Psi_t$ 可以是这几个函数近似：

| #   | $\Psi_t$                                | 说明                                            |
| --- | --------------------------------------- | --------------------------------------------- |
| 1   | $\sum_{t=0}^{\infty} r_t$               | 整条轨迹的总回报（最原始的 REINFORCE，方差最大）                 |
| 2   | $\sum_{t'=t}^{\infty} r_{t'}$           | reward-to-go（利用因果性：$t$ 之前的 reward 与 $a_t$ 无关） |
| 3   | $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$  | 加 baseline（比如 $V(s_t)$），进一步降方差                |
| 4   | $Q^{\pi}(s_t, a_t)$                     | Actor-Critic 的基础形式                            |
| 5   | $A^{\pi}(s_t, a_t)$                     | Advantage，几乎是方差最小的选择                          |
| 6   | $r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$ | TD residual（one-step advantage 估计）            |
其中:
$$
\begin{aligned}
V^{\pi}(s_t) &:= \mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}} \left[ \sum_{l=0}^{\infty} r_{t+l} \right] & Q^{\pi}(s_t, a_t) &:= \mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}} \left[ \sum_{l=0}^{\infty} r_{t+l} \right] \\
A^{\pi}(s_t, a_t) &:= Q^{\pi}(s_t, a_t) - V^{\pi}(s_t), & \text{(Advantage function).}
\end{aligned}
$$
其中 Advantage 基本是方差最小的选择，在[[深度 RL#优势 Actor-Critic]]中可见。

为了进一步降低方差，文章接着引入了[[GAE#TD(0) 到 n-step return]]中的技巧，使用了衰减系数，将上述三个函数重写为：
$$
\begin{aligned}
V^{\pi,\gamma}(s_t) &:= \mathbb{E}_{s_{t+1:\infty}, a_{t:\infty}} \left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \right] & Q^{\pi,\gamma}(s_t, a_t) &:= \mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}} \left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \right] \\
A^{\pi,\gamma}(s_t, a_t) &:= Q^{\pi,\gamma}(s_t, a_t) - V^{\pi,\gamma}(s_t). &
\end{aligned}
$$
$$
g^\gamma := \mathbb{E}_{\substack{s_{0:\infty} \\ a_{0:\infty}}} \left[ \sum_{t=0}^{\infty} A^{\pi, \gamma}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right].
$$
接下来找出 $\hat{A}_t$ 估计是一个有偏（但是不太过有偏）的 $A^{\pi,\gamma}$ 估计。其中找 $\hat{A}_t$ 是因为 $A^{\pi,\gamma}$ 是一个空间极大的，只能用估计的方式，所以实践中采用函数 $\hat{A}_t$ 用于估计。当这个估计函数取代 $A^{\pi,\gamma}$ 时不引入偏差，则称其为 $\gamma$-just 的。

如果 $\hat{A}_t$ 满足 $\gamma$-just 条件，则：
$$
\mathbb{E}_{\substack{s_{0:\infty} \\ a_{0:\infty}}} \left[ \hat{A}_t(s_{0:\infty}, a_{0:\infty})
\nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \mathbb{E}_{\substack{s_{0:\infty} \\ a_{0:\infty}}} \left[ A^{\pi,\gamma}(s_t, a_t)
\nabla_\theta \log \pi_\theta(a_t \mid s_t) \right].
$$
即如果 $\hat{A}_t$ 是 对于所有 $t$ 满足 $\gamma$-just 的话，则
$$
\mathbb{E}_{\substack{s_{0:\infty} \\ a_{0:\infty}}} \left[ \sum_{t=0}^\infty \hat{A}_t(s_{0:\infty}, a_{0:\infty})
abla_\theta \log \pi_\theta(a_t \mid s_t) \right] = g^\gamma
$$
可以证明如果 $\hat{A}_t$ 可以写成如下形式，且满足 $(s_t, a_t)$ 的条件，那么 $\hat{A}$ 是 $\gamma$-just。此处 $Q_t$ 是个
$$
\begin{aligned}
\hat{A}_t(s_{0:\infty}, a_{0:\infty}) &= Q_t(s_{t:\infty}, a_{t:\infty}) - b_t(s_{0:t}, a_{0:t-1})\\
\mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty} \mid s_t, a_t} [Q_t(s_{t:\infty}, a_{t:\infty})] &= Q^{\pi, \gamma}(s_t, a_t)
\end{aligned}
$$
则这几个表达都是 $\gamma$-just 的：
* $\sum_{l=0}^{\infty} \gamma^l r_{t+l}$
* $Q^{\pi,\gamma}(s_t, a_t)$
* $A^{\pi,\gamma}(s_t, a_t)$
* $r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t).$

作者在这里引入了两层 bias，即两层估计。第一层是使用 $g^\gamma$ 来代替 $g$，这个是明确引入了 bias 了。而第二层是使用 $\hat{A}_t$ 代替了 $g^\gamma$ 中的 $A^{\pi,\gamma}$，这里在找到一个没有偏差的估计。

## 优势函数估计

接下来找到一个准确的估计 $\hat{A}_t$，用于估计优势函数 $A^{\pi,\gamma}$。由[[深度 RL|优势函数是由 *TD error* 近似]]的结论，可以尝试使用 TD error 来估计。特别的，对于价值函数 $V$，定义该价值函数的 *TD error* 为：$\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t)$。*TD error* 可以看作一个动作的优势函数的估计，如果使用正确的价值函数 $V=V^{\pi,\gamma}$，则 *TD error* 是一个 $\gamma$-just 优势函数估计并且是 $A^{\pi,\gamma}$ 的无偏估计。
$$
\begin{aligned}
\mathbb{E}_{s_{t+1}} \left[ \delta_t^{V^{\pi, \gamma}} \right] &= \mathbb{E}_{s_{t+1}} [r_t + \gamma V^{\pi, \gamma}(s_{t+1}) - V^{\pi, \gamma}(s_t)] \\
&= \mathbb{E}_{s_{t+1}} [Q^{\pi, \gamma}(s_t, a_t) - V^{\pi, \gamma}(s_t)] = A^{\pi, \gamma}(s_t, a_t).
\end{aligned}
$$
然而这只对 $V = V^{\pi,\gamma}$ 是 $\gamma$-just 的，其他都不能保证。

再引入 telescoping sum，标注为 $\hat{A}_t^{(k)}$，这是一种降低 bias 的方法。
$$
\hat{A}_t^{(k)} := \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
$$
当 $k \to \infty$ 时， 有：
$$
\hat{A}_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V = -V(s_t) + \sum_{l=0}^{\infty} \gamma^l r_{t+l},
$$
再引入 n-steps return 的思路，可以得到*泛化优势估计*（generalized advantage estimator）：$GAE(\gamma,\lambda)$ 
$$
\begin{aligned}
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} &:= (1-\lambda)\left(\hat{A}_t^{(1)} + \lambda\hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + \dots\right) \\
&= (1-\lambda)\left(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \dots\right) \\
&= (1-\lambda)\left(\delta_t^V(1 + \lambda + \lambda^2 + \dots) + \gamma\delta_{t+1}^V(\lambda + \lambda^2 + \lambda^3 + \dots) \right. \\
&\quad \left. + \gamma^2\delta_{t+2}^V(\lambda^2 + \lambda^3 + \lambda^4 + \dots) + \dots\right) \\
&= (1-\lambda)\left(\delta_t^V\left(\frac{1}{1-\lambda}\right) + \gamma\delta_{t+1}^V\left(\frac{\lambda}{1-\lambda}\right) + \gamma^2\delta_{t+2}^V\left(\frac{\lambda^2}{1-\lambda}\right) + \dots\right) \\
&= \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}^V
\end{aligned}
$$
同样的：
- $\lambda = 0$：退化为 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$​
- $\lambda = 1$：$\hat{A}_t := \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$

后者无论什么价值函数 $V$ 都是 $\gamma$-just 的，但是引入了高方差，前者只对 $V=V^{\pi,\gamma}$ 是 $\gamma$-just 的。
$$
V^{\pi,\gamma}(s) = \mathbb{E}\left[\sum_{l=0}^{\infty} \gamma^l r_{t+l} \,\Big|\, s_t = s, \pi\right]
$$
这在实践中基本不可能实现一个逐点相等的价值函数。

> [!NOTE] 这段的总结
> 我们描述了具有两个独立参数 γ 和 λ 的优势估计器，这两个参数在使用近似值函数时都有助于偏差-方差权衡。然而，它们有不同的目的，并且在不同的值范围下效果最好。 γ 最重要的是决定了价值函数 $V^{π,γ}$ 的尺度，它不依赖于 λ。无论价值函数的准确性如何，采用 ​​γ < 1 都会在策略梯度估计中引入偏差。另一方面，仅当价值函数不准确时，λ < 1 才会引入偏差。根据经验，我们发现 λ 的最佳值远低于 γ 的最佳值，这可能是因为对于相当准确的值函数，λ 引入的偏差远小于 γ。

## INTERPRETATION AS REWARD SHAPING

本段提供了另一种视角观察参数 $\lambda$，暂时先跳过。

## Value Function Estimation

在估计价值函数的时候会遇到以下几个问题：
- **目标在漂移**：策略 $\pi$ 在不断更新，$V^{\pi,\gamma}$ 这个回归目标也在漂移，critic 永远在追一个移动靶。
- **数据分布在漂移**：每次迭代用的是新策略收集的数据，旧数据和新数据分布不同。
- **过拟合到最近批次**：如果对最近一批数据优化得过于激进，critic 会遗忘之前学到的全局结构，在下一次迭代时给出质量很差的估计。
- **critic 崩坏的连锁反应**：critic 变差 → advantage 估计变差 → 策略梯度变差 → 策略更新方向错误 → 收集到的数据质量更差 → critic 更难学。这是一个正反馈失控循环。

一个朴素的解决方法是直接求解：
$$
\text{minimize}_\phi \sum_{n=1}^N \|V_\phi(s_n) - \hat{V}_n\|^2
$$
其中 $\hat{V}_n = \sum_{l=0}^\infty \gamma^l r_{t+l}$ 表示蒙特卡洛采样，虽然是无偏采样，但是因为整条轨迹的随机性累积导致方差大，且存在前面说的过拟合问题。并没有解决上述的问题。

于是论文引入了信赖域算法：
$$
\begin{aligned}
\underset{\phi}{\text{minimize}} \quad & \sum_{n=1}^N |V_\phi(s_n) - \hat{V}n|^2 \\
\text{subject to} \quad & \frac{1}{N}\sum_{n=1}^N \frac{|V_\phi(s_n) - V_{\phi_\text{old}}(s_n)|^2}{2\sigma^2} \leq \epsilon
\end{aligned}
$$
其中 $\sigma^2 = \frac{1}{N}\sum_n \|V_{\phi_\text{old}}(s_n) - \hat{V}_n\|^2$，$\phi_{\text{old}}$ 表示的是旧的参数，此约束等价于约束新旧 value function 之间的平均 KL 散度小于 $\epsilon$，如果将 value function 解释为一个条件高斯分布 $\mathcal{N}(V_\phi(s),\sigma^2)$ 的均值，具体来说，两个同方差高斯分布 $\mathcal{N}(\mu_1, \sigma^2)$ 和 $\mathcal{N}(\mu_2, \sigma^2)$ 之间的 KL 散度为：
$$
\text{KL} = \frac{(\mu_1 - \mu_2)^2}{2\sigma^2}
$$
因此有：
$$
\frac{1}{N}\sum_n \frac{\|V_\phi(s_n) - V_{\phi_\text{old}}(s_n)\|^2}{2\sigma^2} = \mathbb{E}_n[\text{KL}(p_\text{old}(s_n) \| p_\text{new}(s_n))]
$$
这是一个自适应归一化：如果旧 critic 的预测误差本来就很大（$\sigma^2$ 大），说明问题本身噪声大，允许较大的参数更新；如果旧 critic 已经很准（$\sigma^2$ 小），则更新应该更保守。

这个信赖域的求解非常复杂，因此论文将这个优化问题转为线性化和二次近似的方法：
$$
\begin{aligned}
\underset{\phi}{\text{minimize}} \quad & g^T(\phi - \phi_\text{old}) \\
\text{subject to} \quad & \frac{1}{N}\sum_{n=1}^N (\phi - \phi_\text{old})^T H (\phi - \phi_\text{old}) \leq \epsilon
\end{aligned}
$$
其中：
- $g = \nabla_\phi \sum_n \|V_\phi(s_n) - \hat{V}_n\|^2 \big|_{\phi = \phi_\text{old}}$ 是目标函数的梯度。
- $H = \frac{1}{N}\sum_n j_n j_n^T$，而 $j_n = \nabla_\phi V_\phi(s_n)$ 是每个样本预测值对参数的梯度。
Q