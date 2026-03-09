
> [!NOTE] Reference
> [CS229笔记]([main_notes.pdf](https://cs229.stanford.edu/main_notes.pdf))第15章和[中文翻译仓库](https://github.com/Na-moe/CS229_CN/blob/main)以及[强化学习的数学原理](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)这本书
## 马可夫决策过程(MDP)
一个马可夫决策过程由一个元组定义$(S,A, \{P_{SA}\},\gamma, R)$，其中：
- $S$:是一系列状态
- $A$:是一系列动作
- $P_{SA}$:是状态转换可能，对于$s\in S, a\in A$，$P_{SA}$是当下状态所有可能转换的状态的可能性
- $\gamma\in [0,1)$:是discount factor
- $R:S\times A\mapsto R$:是奖励函数
MDP的动态过程如下：我们从某个状态$s_0$开始，然后在MDP中选择一个动作 $a_0 ∈ A$ 执行。然后 MDP 的状态随机转移到某个后继状态$s_1$，根据$s_1 ∼P_{s_0a_0}$ 抽取。然后选择另一个动作$a_1$。由于这个动作，状态再次转移，现在转移到某个 $s_2 ∼ P_{s_1a_1}$ ，依此类推。可以将这个过程表示为：
$$
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} s_3 \xrightarrow{a_3} \dots
$$
以动作序列 $a_0, a_1, \dots$ 遍历状态序列 $s_0, s_1, \dots$ 后，总收益由下式给出
$$
    R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \dots.
$$
或者，将奖励写成仅关于状态的函数时，则变为
$$
    R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots.
$$
在强化学习中，目标是随着时间推移选择动作以最大化总收益的期望值：
$$
    \text{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots\right]
$$

**状态价值函数 $V_\pi(s)$**

**定义：**
$$
V_\pi(s) = \mathbb{E}\Bigg[ \sum_{t=0}^\infty \gamma^t r_t \;\Big|\; s_0 = s,\, a_t \sim \pi \Bigg]
$$
**含义：**  
在初始状态为 $s$ 的条件下，遵循策略 $\pi$ 时，期望获得的 **累计折扣回报**。用来衡量某个状态本身的“好坏”。

**状态-动作价值函数 $Q_\pi(s,a)$**

**定义：**

$$Q_\pi(s,a) = \mathbb{E}\Bigg[ \sum_{t=0}^\infty \gamma^t r_t \;\Big|\; s_0 = s,\, a_0 = a,\, a_{1:\infty} \sim \pi \Bigg]$$

**含义：**  
在状态 $s$下先执行动作 $a$，之后遵循策略 $\pi$，期望获得的 **累计折扣回报**。用来衡量某个动作在某个状态下的“好坏”。

**策略目标函数 $J(\pi)$**

**定义：**

$$J(\pi) = \mathbb{E}_{s_0 \sim \rho_0}\Big[ V_\pi(s_0) \Big] = \mathbb{E}_{s_0:\infty \sim \rho_\pi,\, a_0:\infty \sim \pi}\Bigg[ \sum_{t=0}^\infty \gamma^t r_t \Bigg]$$

**含义：**  
策略 $\pi$ 在 **整个初始状态分布 $\rho_0$** 下的期望累计回报，是强化学习中需要最大化的最终目标。
因此，我们的目标是最大化这个函数：(假定所有智能体共享同样的奖励函数)
$$
J(\pi) \triangleq \mathbb{E}_{s_{0:\infty} \sim \rho_{\pi}^{0:\infty}, a_{0:\infty} \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right].$$
同时定义$A$优势函数，对于单智能体和多智能体，定义分别为：
$$
A_{\pi}(s,a) = Q_{\pi}(s,a)-V_{\pi}(s)
$$
$$
A_{\pi}^{i1:m}\left(s, \mathbf{a}^{j1:k}, \mathbf{a}^{i1:m}\right) \triangleq Q_{\pi}^{j1:k,i1:m}\left(s, \mathbf{a}^{j1:k}, \mathbf{a}^{i1:m}\right) - Q_{\pi}^{j1:k}\left(s, \mathbf{a}^{j1:k}\right).$$

## Bellman 方程

**策略 (policy)** 是一个函数 $\pi: S \mapsto A$，它将状态映射到动作。当处于状态 $s$ 时，如果执行 (executing) 某个策略 $\pi$，则采取动作 $a = \pi(s)$。同时定义策略 $\pi$ 的价值函数 (value function) 为
$$
    V^\pi(s) = \text{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots \mid s_0 = s, \pi\right].
$$
$V^\pi(s)$ 表示从状态 $s$ 开始并按照策略 $\pi$ 采取动作所获得的折扣奖励的期望总和。\footnote{请注意，这里以 $\pi$ 为条件的写法并不完全正确，因为 $\pi$ 不是随机变量，但这在文献中是相当标准的用法。

有了这个策略后，要怎么估计一个策略的**总价值**呢，不可能真的把所有状态的价值真的按照概率加权，那样太慢了。Bellman 方程就是为了解决这个问题而产生的，其利用了 MDP 的马尔可夫性质，即本状态可以只由上个状态决定。

给定一个固定的策略 $\pi$，其价值函数 $V^\pi$ 满足贝尔曼方程 (Bellman equation)：
$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}[R_{t+1}|S_t = s] + \gamma\mathbb{E}[G_{t+1}|S_t = s], \\
&= \underbrace{\sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a)r}_{\text{mean of immediate rewards}} + \underbrace{\gamma \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} p(s'|s,a)v_\pi(s')}_{\text{mean of future rewards}} \\
&= \sum_{a \in \mathcal{A}} \pi(a|s) \left[ \sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v_\pi(s') \right], \quad \text{for all } s \in \mathcal{S}. 
\end{aligned}
$$
这表明从状态 $s$ 开始的折扣奖励期望总和 $V^\pi(s)$ 由两部分组成：第一部分是从状态 $s$ 开始即刻获得的即时奖励 (immediate reward) $R(s)$；第二部分是未来折扣奖励的期望总和。仔细考察第二项，可以看到上面的求和项可以重写为 $\text{E}_{s' \sim P_{s\pi(s)}}[V^\pi(s')]$。这是从状态 $s'$ 开始的折扣奖励的期望总和，其中 $s'$ 的分布由 $P_{s\pi(s)}$ 给出，也就是在 MDP 中从状态 $s$ 执行第一个动作 $\pi(s)$ 后将到达的状态分布。因此，上面的第二项给出的是在 MDP 中执行第一步后获得的折扣奖励的期望总和。

贝尔曼公式利用了**Bootstrap(自举)** 的思想，它不再依赖长期问题，而是将长期问题分解为下一状态的价值，变成了一种递归的求发。具体来说，其核心为**用当前已有的价值估计，去更新另一个价值估计。** 最后可以有效收敛成正确的奖励。

贝尔曼方程可以有效地用于求解 $V^\pi$。具体来说，在一个有限状态 MDP ($|S| < \infty$) 中，可以为每个状态 $s$ 写出一个关于 $V^\pi(s)$ 的方程。这给出了 $|S|$ 个线性方程组，其中包含 $|S|$ 个变量（未知的 $V^\pi(s)$），可以有效地求解这些变量。

为了表示线性公式，可以把贝尔曼方程拆成：
$$
\begin{aligned}
v_{\pi}(s) &= r_{\pi}(s) + \gamma \sum_{s^{\prime} \in \mathcal{S}} p_{\pi}\left(s^{\prime} \mid s\right) v_{\pi}\left(s^{\prime}\right), \\
r_{\pi}(s) &\doteq \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s, a) r, \\
p_{\pi}\left(s^{\prime} \mid s\right) &\doteq \sum_{a \in \mathcal{A}} \pi(a \mid s) p\left(s^{\prime} \mid s, a\right).
\end{aligned}
$$
此时，对于如下图所示系统，可以写出每个状态对应本策略的总奖励：
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260307121626257.png)
$$
\begin{aligned}
v_{\pi}(s_i) &= r_{\pi}(s_i) + \gamma \sum_{s_j \in \mathcal{S}} p_{\pi}(s_j \mid s_i) v_{\pi}(s_j) \\
v_{\pi} &= r_{\pi} + \gamma P_{\pi} v_{\pi}
\end{aligned}
$$
$$
\underbrace{
\begin{bmatrix}
v_{\pi}(s_1) \\
v_{\pi}(s_2) \\
v_{\pi}(s_3) \\
v_{\pi}(s_4)
\end{bmatrix}
}_{v_{\pi}}
=
\underbrace{
\begin{bmatrix}
r_{\pi}(s_1) \\
r_{\pi}(s_2) \\
r_{\pi}(s_3) \\
r_{\pi}(s_4)
\end{bmatrix}
}_{r_{\pi}}
+
\gamma
\underbrace{
\begin{bmatrix}
p_{\pi}(s_1|s_1) & p_{\pi}(s_2|s_1) & p_{\pi}(s_3|s_1) & p_{\pi}(s_4|s_1) \\
p_{\pi}(s_1|s_2) & p_{\pi}(s_2|s_2) & p_{\pi}(s_3|s_2) & p_{\pi}(s_4|s_2) \\
p_{\pi}(s_1|s_3) & p_{\pi}(s_2|s_3) & p_{\pi}(s_3|s_3) & p_{\pi}(s_4|s_3) \\
p_{\pi}(s_1|s_4) & p_{\pi}(s_2|s_4) & p_{\pi}(s_3|s_4) & p_{\pi}(s_4|s_4)
\end{bmatrix}
}_{P_{\pi}}
\underbrace{
\begin{bmatrix}
v_{\pi}(s_1) \\
v_{\pi}(s_2) \\
v_{\pi}(s_3) \\
v_{\pi}(s_4)
\end{bmatrix}
}_{v_{\pi}}.
$$

$$
\begin{bmatrix}
v_{\pi}(s_1) \\
v_{\pi}(s_2) \\
v_{\pi}(s_3) \\
v_{\pi}(s_4)
\end{bmatrix}
=
\begin{bmatrix}
0.5(0) + 0.5(-1) \\
1 \\
1 \\
1
\end{bmatrix}
+
\gamma
\begin{bmatrix}
0 & 0.5 & 0.5 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
v_{\pi}(s_1) \\
v_{\pi}(s_2) \\
v_{\pi}(s_3) \\
v_{\pi}(s_4)
\end{bmatrix}.
$$
$P_{\pi}$ 矩阵需要满足两个性质：
- 非负，因为概率不可能是负数： $P_{\pi}>0$
- 每行的和为1，因为每个状态的转移概率总和为1 ： $P_{\pi}1=1$

但是现实中不会直接求闭式解，因为求逆会消耗很大算力。实际上是使用迭代的方式，最开始先随便赋值 $v_0$，然后通过：
$$
v_{k+1}=r_\pi+\gamma P_\pi v_k,\quad k=0,1,2,\dots
$$
来不断更新总价值。最后可以迭代收敛到真正的 $v_\pi$。

同样地，定义最优价值函数 (optimal value function)为
$$
    V^*(s) = \max_{\pi} V^\pi(s).
$$
换句话说，这是使用任何策略可以达到的最佳期望折扣奖励总和。对于最优价值函数，也有一个贝尔曼方程：
$$
    V^*(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s').
$$
上面的第一项是即时奖励。第二项是在执行动作 $a$ 之后获得的期望未来折扣奖励总和在所有动作 $a$ 上的最大值。应该确保理解这个方程及其合理性。
同时定义策略 $\pi^*: S \mapsto A$ 如下：
$$
    \pi^*(s) = \arg \max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s').
$$
注意，$\pi^*(s)$ 给出了在方程eq:15.2中的 "max" 中达到最大值的动作 $a$。

事实证明，对于每一个状态 $s$ 和每一个策略 $\pi$，有
$$
    V^*(s) = V^{\pi^*}(s) \ge V^\pi(s).
$$
第一个等号表示，对于每个状态 $s$，策略 $\pi^*$ 的价值函数 $V^{\pi^*}$ 都等于最优价值函数 $V^*$。此外，不等号表示 $\pi^*$ 的价值至少与任何其他策略的价值一样大。换句话说，方程所定义的 $\pi^*$ 是最优策略 (optimal policy)。

## BOE
Bellman Optimal Equation，用于表示某个状态或者动作是最优的，即所获得的即时奖励加上后续状态的折扣最优价值的期望是最低的。

具有两种形态：
- 状态价值函数（$V$）：在状态 $s$ 下，最优价值等于遍历所有动作 $a$，取「即时奖励 + 折扣后继状态最优价值期望」的最大值。
	$$
	V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' | s, a) \, V^*(s') \right]
	$$
- 动作价值函数（$Q$）：在状态 $s$ 执行动作 $a$ 的最优价值，等于即时奖励加上到达下一个状态后、再选最优动作的折扣期望价值。
	$$
	Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q^*(s', a')
	$$
两者的关系很直接：
$$
V^*(s) = \max_a Q^*(s, a)
$$
至于为什么是最优，最优是否是唯一的等问题，可以看书。

随之可以自然地推出 Value iteration 和 Policy iteration 的方程：
$$
V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \, V_k(s') \right]
$$
通过这个 Value iteration 方程迭代后可以用 Policy iteration 同时更新局部最佳策略：
$$
\pi_{k+1}(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \, V^{\pi_k}(s') \right]
$$
最后当两个值都不再变化后再