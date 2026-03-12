> [!NOTE] TL;DR
> 暂无

[[Bellman 公式]]中总结了对于 With model RL 的基本原则和更新方式，但是现实中的问题通常不能保证有严谨的公式。本笔记想对 model free RL的基本方法和思想进行一些概括和总结，好对后面的[[Actor Critic]]等算法打好基础。此处所说的 model 指的是**环境模型**，即对环境的**状态转移概率**和**奖励机制**进行建模。在拥有 Model 的情况下（类似围棋），智能体可以预知执行某个动作后环境会发生怎样的变化（确定性或随机性）；而 Model-free RL 则是要在不掌握这些环境动态规律的情况下，直接通过与环境的交互来学习策略。

## 蒙特卡洛方法

蒙特卡罗方法是所有 model free RL 的基础，因为没有了 Model 预测执行某个动作后会发生，就只能通过蒙特卡洛的概率进行预测状态转移的概率。至于蒙特卡罗方法是啥，简而言之，就是用大量采样的概率视作真实概率，像是掷硬币足够多就可以逼近 $0.5$ 的概率。

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

可以看到这个方法基本就是