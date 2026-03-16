---
date: 2026-03-09
lastmod: 2026-03-16
---
> [!NOTE] TL;DR
> 暂无

[[Bellman 公式]]中总结了对于 With model RL 的基本原则和更新方式，但是现实中的问题通常不能保证有严谨的公式。本笔记想对 model free RL的基本方法和思想进行一些概括和总结，好对后面的[[Actor Critic]]等算法打好基础。此处所说的 model 指的是**环境模型**，即对环境的**状态转移概率**和**奖励机制**进行建模。在拥有 Model 的情况下（类似围棋），智能体可以预知执行某个动作后环境会发生怎样的变化（确定性或随机性）；而 Model-free RL 则是要在不掌握这些环境动态规律的情况下，直接通过与环境的交互来学习策略。

## 蒙特卡洛方法

蒙特卡罗方法是所有 model free RL 的基础，因为没有了 Model 预测执行某个动作后会发生，就只能通过通过大量采样取平均来近似期望回报，从而估计 $q(s,a)$。

比如说[[Bellman 公式]]中的计算出状态价值函数 $v_{\pi_k}$ 后可以得到状态-动作价值函数的计算方法：
$$
q_{\pi_k}(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}(s').
$$
然而这个方法需要知道系统模型 $\{\,\,p(r|s,a),p(s'|s,a)\,\,\}$ 已知，对于 model free 方法来说，回归定义：
$$
\begin{aligned} q_{\pi_k}(s, a) &= \mathbb{E}[G_t | S_t = s, A_t = a] \\ &= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots | S_t = s, A_t = a], \end{aligned}
$$
也就是累计折扣回报的期望，假定总共 $n$ 轮采样，而第 $i$ 轮回报是 $g_{\pi_k}^{(i)}(s,a)$，则蒙特卡罗方法将 $q_{\pi_k}(s,a)$ 可以由下列公式近似：
$$
q_{\pi_k}(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a] \approx \frac{1}{n} \sum_{i=1}^{n} g_{\pi_k}^{(i)}(s, a).
$$

### MC Basic 算法
$$
\begin{array}{l} \textbf{Algorithm 5.1: MC Basic (a model-free variant of policy iteration)} \\ \\ \textbf{Initialization:} \text{ Initial guess } \pi_0. \\ \textbf{Goal:} \text{ Search for an optimal policy.} \\ \\ \text{For the } k\text{th iteration } (k = 0, 1, 2, \ldots), \text{ do} \\ \quad \text{For every state } s \in \mathcal{S}, \text{ do} \\ \quad\quad \text{For every action } a \in \mathcal{A}(s), \text{ do} \\ \quad\quad\quad \text{Collect sufficiently many episodes starting from } (s, a) \text{ by following } \pi_k \\ \\ \quad\quad\quad \textbf{Policy evaluation:} \\ \quad\quad\quad q_{\pi_k}(s, a) \approx q_k(s, a) = \text{the average return of all the episodes starting from } (s, a) \\ \\ \quad\quad\quad \textbf{Policy improvement:} \\ \quad\quad\quad a_k^*(s) = \arg\max_a q_k(s, a) \\ \quad\quad\quad \pi_{k+1}(a|s) = \begin{cases} 1 & \text{if } a = a_k^* \\ 0 & \text{otherwise} \end{cases} \end{array} 
$$

可以看到这个方法基本就是 Policy iteration 方程的 model free 方法，显示 Policy evaluation 然后接着 Policy improvement。但是注意原来的 PE 是更新状态 $V$ 状态价值函数，此处改为更新 $Q$ 状态-动作价值函数，这是因为在 PI 这步，model-based 方法是通过该方程求解：
$$
a^* = \arg\max_a \sum_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma v_{\pi}(s') \right]
$$
但是 model free 中，$p(s'|s,a)$ 是未知的，所以采用对状态-动作奖励函数进行优化。

$$
\text{Collect sufficiently many episodes starting from } (s, a) \text{ by following } \pi_k 
$$
这步的采样深度 $\text{episode length}$ 也会极大影响质量，即从 $(s,a)$ 出发，走几步才停止采样？

整体来说，MC basic 方法过于原始，工程上会采取许多优化方法。比如说把一个策略的 episode 多次利用：
$$
s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \cdots
$$
对于这一条轨迹，不仅可以估计 $q(s_1,a_2)$ 的值，并且可以同时估计沿线的所有状态的状态-动作价值函数，比如可以同时利用 $(s_2,a_4),(s_5,a_1)$ 的状态来估值。其中，如果每一个 状态-动作对 只可以将第一次用于估计，则成为 $\text{first-visit}$，否则则成为 $\text{every-visit}$

$$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{if } a = a^* \\ \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{otherwise} \end{cases}$$
这个改进首先是增大了每一个 episode 的利用率，不再是一个轨迹只修改一个 状态动作对的价值函数，而是可以修改整个轨迹中的状态动作对的价值函数，可以利用下一步的 累计折扣回报$g$ 来更新上一步的回报，即 $g \leftarrow \gamma g + r_{t+1}$ 这步。

同时，这个算法是每采样完一个 episode 就评价和更新策略，而不是像经典MC那样采够了足够的轨迹才开始更新。

但是这个方法没法保证每种状态动作对都探索到了，如果没有探索到，因为没有样本，回报会严重不足，可能会导致策略陷入局部最优解。因此算法引入了一些随机性，采用了 $\epsilon$-greedy policies 的方法，采用部分随机的方式来改进策略：
$$
\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{for the greedy action } \\ \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{for the other}\,\, |\mathcal{A}(s)|-1 \text{actions} \end{cases}
$$
最后改进为：
$$
\begin{array}{l}
\textbf{Algorithm 5.3: MC } \epsilon\text{-Greedy (a variant of MC Exploring Starts)} \\
\\
\textbf{Initialization:} \text{ Initial policy } \pi_0(a|s) \text{ and initial value } q(s,a) \text{ for all } (s,a). \\
\quad \text{Returns}(s,a) = 0 \text{ and } \text{Num}(s,a) = 0 \text{ for all } (s,a). \\
\quad \epsilon \in (0,1] \\
\\
\textbf{Goal:} \text{ Search for an optimal policy.} \\
\\
\text{For each episode, do} \\
\quad \textit{Episode generation:} \text{ Select a starting state-action pair } (s_0, a_0) \text{ (the exploring starts} \\
\quad \text{condition is not required). Following the current policy, generate an episode of length } T: \\
\quad s_0, a_0, r_1, \ldots, s_{T-1}, a_{T-1}, r_T. \\
\quad \text{Initialization for each episode: } g \leftarrow 0 \\
\quad \text{For each step of the episode, } t = T-1, T-2, \ldots, 0, \text{ do} \\
\quad\quad g \leftarrow \gamma g + r_{t+1} \\
\quad\quad \text{Returns}(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) + g \\
\quad\quad \text{Num}(s_t, a_t) \leftarrow \text{Num}(s_t, a_t) + 1 \\
\quad\quad \textit{Policy evaluation:} \\
\quad\quad\quad q(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) / \text{Num}(s_t, a_t) \\
\quad\quad \textit{Policy improvement:} \\
\quad\quad\quad \text{Let } a^* = \arg\max_a q(s_t, a) \text{ and} \\
\quad\quad\quad \pi(a|s_t) = 
\begin{cases}
1 - \dfrac{|\mathcal{A}(s_t)| - 1}{|\mathcal{A}(s_t)|}\epsilon, & a = a^* \\
\dfrac{1}{|\mathcal{A}(s_t)|}\epsilon, & a \ne a^*
\end{cases}
\end{array}
$$
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260312215103064.png)
可以看到，当 $\epsilon$ 越大的时候，策略越趋向随机来遍历所有状态-动作状态。