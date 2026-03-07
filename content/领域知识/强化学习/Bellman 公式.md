
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
## Bellman 方程

**策略 (policy)** 是一个函数 $\pi: S \mapsto A$，它将状态映射到动作。当处于状态 $s$ 时，如果执行 (executing) 某个策略 $\pi$，则采取动作 $a = \pi(s)$。同时定义策略 $\pi$ 的价值函数 (value function) 为
$$
    V^\pi(s) = \text{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots \mid s_0 = s, \pi\right].
$$
$V^\pi(s)$ 表示从状态 $s$ 开始并按照策略 $\pi$ 采取动作所获得的折扣奖励的期望总和。\footnote{请注意，这里以 $\pi$ 为条件的写法并不完全正确，因为 $\pi$ 不是随机变量，但这在文献中是相当标准的用法。

有了这个策略后，要怎么估计一个策略的**总价值**呢，不可能真的把所有状态的价值真的按照概率加权，那样太慢了。Bellman 方程就是为了解决这个问题而产生的。

给定一个固定的策略 $\pi$，其价值函数 $V^\pi$ 满足贝尔曼方程 (Bellman equation)：
$$
    V^\pi(s) = R(s) + \gamma \sum_{s' \in S} P_{s\pi(s)}(s') V^\pi(s').
$$
这表明从状态 $s$ 开始的折扣奖励期望总和 $V^\pi(s)$ 由两部分组成：第一部分是从状态 $s$ 开始即刻获得的即时奖励 (immediate reward) $R(s)$；第二部分是未来折扣奖励的期望总和。仔细考察第二项，可以看到上面的求和项可以重写为 $\text{E}_{s' \sim P_{s\pi(s)}}[V^\pi(s')]$。这是从状态 $s'$ 开始的折扣奖励的期望总和，其中 $s'$ 的分布由 $P_{s\pi(s)}$ 给出，也就是在 MDP 中从状态 $s$ 执行第一个动作 $\pi(s)$ 后将到达的状态分布。因此，上面的第二项给出的是在 MDP 中执行第一步后获得的折扣奖励的期望总和。

贝尔曼方程可以有效地用于求解 $V^\pi$。具体来说，在一个有限状态 MDP ($|S| < \infty$) 中，可以为每个状态 $s$ 写出一个关于 $V^\pi(s)$ 的方程。这给出了 $|S|$ 个线性方程组，其中包含 $|S|$ 个变量（未知的 $V^\pi(s)$），可以有效地求解这些变量。

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

