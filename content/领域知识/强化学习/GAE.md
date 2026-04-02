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

