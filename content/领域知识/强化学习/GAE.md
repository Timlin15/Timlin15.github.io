---
date: 2026-04-08
lastmod: 2026-04-08
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
可以证明如果 $\hat{A}_t$ 可以写成如下形式，且满足 $(s_t, a_t)$ 的条件，那么 $\hat{A}$ 是 $\gamma$-just。
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

接下来找到一个准确的估计 $\hat{A}_t$，用于估计优势函数 $A^{\pi,\gamma}$。由[[深度 RL|优势函数是由 *TD error* 近似]]的结论，可以尝试使用 TD error 来估计。特别的，对于价值函数 $V$，