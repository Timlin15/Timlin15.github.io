---
date: 2026-03-09
lastmod: 2026-03-18
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
这步的采样深度 $\text{episode length}$ 也会极大影响质量，即从 $(s,a)$ 出发，走几步才停止采样。

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


## 随机近似

对于上述的算法中，利用增量的方式求均值的方式，是如何知道其是有效的：
$$
w_{k+1}=w_k -\frac 1k (w_k-x_k).
$$
对于系数为 $\frac 1k$ 来说，可以通过数学推导来得到结果，但是对于其他系数呢，什么系数能满足这个公式？ 为什么这些系数可以逼近平均数？

Robbins-Monro 公式是来解答这个问题的，即逼近方程解的有效性，由于重心不在这，因此我只是粗略介绍。增量近似和 SGD 算法都是 RM 方程的一种特例。

总体来说，RM 算法想回答一个方程解的问题，我们想求一个方程的根，由此可以等价成求方程的任意解：
$$
g(w)=0
$$
有了这个方程解，我们可以求出梯度解，得到最优质，或者得到 $g(w)=c$ 这类方程的答案。但是其中的方程 $g$ 是带有随机性的方程，或者说我们只能观测到带有噪音的 $g(w)$，比如说 SGD 或者增量求平均数的时候，我们不知道下一个样本对客观量的偏移量:
$$
\tilde{g}(w, \eta)=g(w)+\eta
$$
这个 $\eta \in \mathbb{R}$ 是一个观测误差，不一定是高斯分布。

RM 算法保证，可以通过下列方程求解 $g(w)=0$ ：
$$
w_{k+1}=w_k-a_k \tilde{g}\left(w_k, \eta_k\right), \quad k=1,2,3, \ldots
$$
其中 
- $w_k$是第 $k$ 个根的估计；
- $\tilde{g}\left(w_k, \eta_k\right)$ 是第 $k$ 个带噪采样；
- $a_k$ 是个正参数。

只要满足以下条件，则可以保证能求出正确参数：
1. 对所有 $w$ 有 $0 < c_1 \leq \nabla_w g(w) \leq c_2$；  
2. $\sum_{k=1}^{\infty} a_k = \infty$ 且 $\sum_{k=1}^{\infty} a_k^2 < \infty$；  
3. $\mathbb{E}[\eta_k|\mathcal{H}_k] = 0$ 且 $\mathbb{E}[\eta_k^2|\mathcal{H}_k] < \infty$；  

其中 $\mathcal{H}_k = \{w_k, w_{k-1}, \ldots\}$，则 $w_k$ 几乎必然收敛到满足 $g(w^*) = 0$ 的根 $w^*$。

证明略，见书第六章。

可以发现求增量平均和 SGD 方法中的参数 $\frac 1n$ 满足其本身是级数发散，平方是级数收敛，正好满足 RM 算法对参数的要求。


## TD 算法

把随机近似应用到 model free 估计中就可以得到 TD(Temporal-Difference) 方法。不同于 MC 方法使用统计学的方法，TD 方法采用基于动态规划的随机近似，即基于上文中阐述的随机近似方法。
$$
\begin{aligned}
v_{t+1}\left(s_t\right) & =v_t\left(s_t\right)-\alpha_t\left(s_t\right)\left[v_t\left(s_t\right)-\left(r_{t+1}+\gamma v_t\left(s_{t+1}\right)\right)\right] \\
v_{t+1}(s) & =v_t(s), \quad \text { for all } s \neq s_t
\end{aligned}
$$
这是由于 $v_\pi(s)=\mathbb{E}\left[R_{t+1}+\gamma v_\pi\left(S_{t+1}\right) \mid S_t=s\right], \quad s \in \mathcal{S} .$ 因此这就是和求增量平均那个随机近似方法一样，用新的采样和上一步参数来估计下一步参数：$w_{k+1}=w_k -\frac 1k (w_k-x_k).$

$$
\underbrace{v_{t+1}\left(s_t\right)}_{\text {new estimate }}=\underbrace{v_t\left(s_t\right)}_{\text {current estimate }}-\alpha_t\left(s_t\right)[\overbrace{v_t\left(s_t\right)-(\underbrace{r_{t+1}+\gamma v_t\left(s_{t+1}\right)}_{\text {TD target } \bar{v}_t})}^{\text {TD error } \delta_t}],
$$
这个方程中：
$$
\bar{v}_t \doteq r_{t+1}+\gamma v_t\left(s_{t+1}\right)
$$
被称为 *TD target* ，同时：
$$
\delta_t \doteq v\left(s_t\right)-\bar{v}_t=v_t\left(s_t\right)-\left(r_{t+1}+\gamma v_t\left(s_{t+1}\right)\right)
$$
被称为 *TD error*，因为 *TD error* 度量的是 Bellman 方程的残差，真值 $v_\pi(s_t)$  满足：
$$
v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]
$$
所以当 $v_t = v_\pi$ 时，$\mathbb{E}[\delta_t | S_t = s_t] = v_\pi(s_t) - \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s_t] = 0$。同时
$$
\left|v_{t+1}\left(s_t\right)-\bar{v}_t\right|=\left|1-\alpha_t\left(s_t\right)\right|\left|v_t\left(s_t\right)-\bar{v}_t\right| .
$$
所以有：
$$
\left|v_{t+1}\left(s_t\right)-\bar{v}_t\right|<\left|v_t\left(s_t\right)-\bar{v}_t\right|
$$
可以保证新的估计值比旧的估计值更接近真值，因此，这个方程可以肯定最后是可以收敛到结果中的。

| TD学习                                                                                                                                   | MC学习                                                                                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **增量式（Incremental）**：TD学习是增量式的。它可以在接收到一个经验样本后立即更新状态/动作值。                                                                               | **非增量式（Non-incremental）**：MC学习是非增量式的。它必须等待一个完整的回合（episode）被收集完毕。这是因为必须计算该回合的折扣回报。                                                                                                                                                                           |
| **持续任务（Continuing tasks）**：由于TD学习是增量式的，它可以处理回合制和持续性任务。持续性任务可能没有终止状态。                                                                   | **回合制任务（Episodic tasks）**：由于MC学习是非增量式的，它只能处理在有限步数后终止的回合制任务。                                                                                                                                                                                                 |
| **自举（Bootstrapping）**：TD学习使用自举，因为状态/动作值的更新依赖于该值的先前估计。因此，TD学习需要对值进行初始猜测。                                                                | **非自举（Non-bootstrapping）**：MC学习不使用自举，因为它可以直接估计状态/动作值，而无需初始猜测。                                                                                                                                                                                               |
| **低估计方差（Low estimation variance）**：TD的估计方差低于MC，因为它涉及的随机变量更少。例如，为了估计动作值 $q_\pi(s_t, a_t)$，Sarsa仅需三个随机变量的样本：$R_{t+1}, S_{t+1}, A_{t+1}$。 | **高估计方差（High estimation variance）**：MC的估计方差更高，因为涉及许多随机变量。例如，为了估计 $q_\pi(s_t, a_t)$，我们需要样本 $R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$。假设每个回合的长度为 $L$，且每个状态的动作数量为 $\|\mathcal{A}\|$，那么在软策略下，有 $\|\mathcal{A}\|^L$ 种可能的回合。如果仅用少数几个回合来估计，估计方差自然会很高。 |
TD 算法的成立性证明见课本。

但是只有价值函数没法在 model free 的环境中指导动作选择，需要获得状态动作价值函数 $q(s,a)$ 才可以，把 TD 算法自然外推，就可以很自然地得到求 $q(s,a)$：
$$
\begin{array}{l}{{q_{t+1}(s_{t},a_{t})=q_{t}(s_{t},a_{t})-\alpha_{t}(s_{t},a_{t})\biggl[q_{t}(s_{t},a_{t})-(r_{t+1}+\gamma q_{t}(s_{t+1},a_{t+1})\biggr],}}\\ {{q_{t+1}(s,a)=q_{t}(s,a),\quad\mathrm{for~all~}(s,a)\not=(s_{t},a_{t}),}}\end{array}
$$
同 TD algo 一样，Sarsa 是 Bellman 方程的随机近似：
$$
q_{\pi}(s,a)=\mathbb{E}\left[R+\gamma q_{\pi}(S^{\prime},A^{\prime})|s,a\right],\quad{\mathrm{for~all~}}\left(s,a\right).
$$

$$
\begin{array}{l}
\textbf{Algorithm 7.1: Optimal policy learning by Sarsa} \\
\\
\textbf{Initialization: } \alpha_t(s,a) = \alpha > 0 \text{ for all } (s,a) \text{ and all } t. \; \epsilon \in (0,1). \text{ Initial } q_0(s,a) \text{ for all } (s,a).\\ \text{ Initial } \epsilon\text{-greedy policy } \pi_0 \text{ derived from } q_0. \\
\\
\textbf{Goal: } \text{Learn an optimal policy that can lead the agent to the target state from an initial state } s_0. \\
\\
\text{For each episode, do} \\
\quad \text{Generate } a_0 \text{ at } s_0 \text{ following } \pi_0(s_0) \\
\quad \text{If } s_t \ (t = 0,1,2,\ldots) \text{ is not the target state, do} \\
\qquad \text{Collect an experience sample } (r_{t+1}, s_{t+1}, a_{t+1}) \text{ given } (s_t, a_t): \text{ generate } r_{t+1}, s_{t+1} \\
\qquad \text{by interacting with the environment; generate } a_{t+1} \text{ following } \pi_t(s_{t+1}). \\
\qquad \text{Update } q\text{-value for } (s_t, a_t): \\
\qquad\qquad q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[ q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})) \right] \\
\qquad \text{Update policy for } s_t: \\
\qquad\qquad \pi_{t+1}(a|s_t) = 1 - \dfrac{\epsilon}{|\mathcal{A}(s_t)|}(|\mathcal{A}(s_t)| - 1) \quad \text{if } a = \arg\max_a q_{t+1}(s_t, a) \\
\qquad\qquad \pi_{t+1}(a|s_t) = \dfrac{\epsilon}{|\mathcal{A}(s_t)|} \quad \text{otherwise} \\
\qquad s_t \leftarrow s_{t+1}, \; a_t \leftarrow a_{t+1}
\end{array}
$$
可以在书上 P.136 查看例子。

TD learning 可以增大采样深度，即不止依赖下一步的回报值，而是依赖多步的回报：
$$
\begin{aligned}
q_{t+n}(s_{t},a_{t})&=q_{t+n-1}(s_{t},a_{t})\\ &{{-\,\alpha_{t+n-1}(s_{t},a_{t})\Big[q_{t+n-1}(s_{t},a_{t})-\big(r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n}q_{t+n-1}(s_{t+n},a_{t+n})\Big)\Big].}}
\end{aligned}
$$
这个方法兼采了 MC 方法和 TD 方法的特点，n 愈大，算法愈趋向于 MC 算法，方差大但是 bias 小。