## 多步Q-learning
先讲解基础Q-learning：这是一种基于bootstrapped(自举)实现的算法，即基于已有估计值去更新自己的估计值，而不是等到完整的真实回报（Return）才能更新。实际例子可以看西湖大学的《强化学习的数学原理》，其中有比较详细的说明。
Q-learning 的核心更新公式是：

$$ Q^\pi(s, a) \leftarrow r(s, a) + \gamma \mathbb{E}_{s' \sim p(s' \mid s, a)} \left[ Q^\pi(s', \pi(s')) \right] $$
但是我们没办法求出下一步的状态动作价值函数，所以可以依靠用此步骤估计下一步，本步和下步的贝尔曼误差是：
$$ \mathcal{E} = \frac{1}{2} \mathbb{E}_{(s,a) \sim \beta} \left[ \left( Q_\phi(s, a) - \left[ r(s, a) + \gamma \max_{a'} Q_\phi(s', a') \right] \right)^2 \right] $$

在拟合 Q 迭代的每一步中，该误差被近似为 $\sum_i |Q_\phi(s_i, a_i) - y_i|^2$。若 $\mathcal{E} = 0$，则有 $Q_\phi(s, a) = r(s, a) + \gamma \max_{a'} Q_\phi(s', a')$。这对应于最优策略 $\pi'$ 的最优 Q 函数。然而，当我们离开表格型情形时，大多数理论保证将不再成立。（表格型情景：像DP那样用表储存每一种的数据，而非用模型拟合）

我们可以将拟合 Q 迭代（fitted Q-iteration）转换为一个在线版本，我们称之为 Q-learning。

---

**算法 11 在线 Q-迭代**

1: **loop**  
2: 执行某个动作 $a_t$，并观察 $(s_i, a_i, s'_i, r_i)$  
3: $y_i = r(s_i, a_i) + \gamma \max_{a'} Q_\phi(s'_i, a')$  （$y$为目标$Q^*(s,a)$）
4: $\phi \leftarrow \phi - \alpha \frac{\partial Q_\phi}{\partial \phi}(s_i, a_i) (Q_\phi(s_i, a_i) - y_i)$  
5: **end loop**

---

我们将 $(Q_\phi(s_i, a_i) - y_i)$ 称为时序差分（TD）误差。在 Q-learning 中，如果我们仅基于 argmax 策略进行探索，可能会被困在环境的某个子集内。
那么它会始终选择当前 Q 函数认为“最优”的动作，但在训练初期，Q 函数往往还不准确。
于是：
- 如果智能体偶然早期“高估”了某个动作；
- 它就会一直重复选择那个动作；
- 因为它不再尝试其他动作；
- 就永远看不到潜在更优的状态或奖励。
- 从而错过能带来更高奖励的状态和动作。
一些常用的探索策略包括 $\varepsilon$-贪婪策略：

$$ \pi(a_t \mid s_t) = \begin{cases} 1 - \epsilon, & a_t = \arg\max_{a_t} Q_\phi(s_t, a_t) \ \epsilon / (|\mathcal{A}| - 1), & \text{otherwise} \end{cases} $$

以及 Boltzmann 探索策略：

$$ \pi(a_t \mid s_t) \propto \exp\left(Q_\phi(s_t, a_t)\right) $$

### TD learning
**TD Learning（Temporal Difference Learning，时序差分学习）**  
是强化学习（RL）中一种核心算法思想，用于在**无需等待完整回合结束**的情况下**估计状态价值（Value）或动作价值（Q-value）**。  
它结合了**蒙特卡洛方法**（MC）和**动态规划**（DP）的优点。

---
MC（蒙特卡洛方法）：对于围棋这样的任务，有明确的数据的，可以采用Policy Iteration这类算法，但是对于如robotics之类的显示任务，没有明确的棋盘格model，限定你下一步可以走向前后左右，所以没办法估计这步的状态奖励。因此MC采用以采样代替计算，用大数定律采集这个状态的状态奖励。
一、基本思想

在 RL 中，我们关心的是状态的“长期收益”：
$$
V^\pi(s) = \mathbb{E}_\pi [R_t | s_t = s]
$$
但要精确知道这个值，必须运行到终止状态才能计算总回报。

TD Learning 的关键创新是：
> “不等回合结束，就能用下一步的估计值来更新当前的价值估计。”

因此叫做 **Temporal Difference（时间差分）**——  
因为它用**时间上相邻的两个状态价值的差**来进行学习。

二、基本更新公式

最核心的 TD(0) 更新公式如下：

$$
V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
$$

含义：
- $V(s_t)$：当前状态的价值估计；
- $\alpha$：学习率；
- $r_{t+1}$：从 $s_t$ 到 $s_{t+1}$ 获得的即时奖励；
- $\gamma$：折扣因子；
- $V(s_{t+1})$：下一状态的估计值；
- 方括号内的部分：
  $$
  \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
  $$
  称为 **TD 误差（TD error）**。
- $$\bar{v_t} = r_{t+1}+\gamma v(s_{t+1})$$
	称为TD target。目标是让$v(s_t)$更趋近$\bar{v_t}$
实际上TD算法和with model的Policy Iteration非常相似。

这其实就是一种**自举**(bootstrapping)更新——当前的估计用下一步的估计来改进。

 三、与其他方法的关系

| 方法 | 是否自举 | 是否使用整回合 | 是否需要模型 |
|------|------------|----------------|----------------|
| **蒙特卡洛（MC）** | ❌ 否 | ✅ 是 | ❌ 否 |
| **动态规划（DP）** | ✅ 是 | ❌ 否 | ✅ 是（知道转移概率） |
| **TD Learning** | ✅ 是 | ❌ 否 | ❌ 否 |

TD Learning 处于两者中间，  
既能在**未知环境下直接学习**，又能**边交互边更新**，因此特别高效。

四、TD 的几种主要形式
1. TD(0)
	最基本形式，只看一步：

$$
V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
$$
2. n-step TD
	延迟 n 步更新：
$$
V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma r_{t+2} + \dots + \gamma^n V(s_{t+n}) - V(s_t)]
$$
	兼顾短期与长期信息。
3. TD(λ)
	引入“迹（eligibility trace）”机制，将所有过去的状态都按照衰减系数 λ 更新：
$$
V(s_t) \leftarrow V(s_t) + \alpha \, \delta_t \, e_t
$$
	其中 $e_t = \gamma \lambda e_{t-1} + 1$。
	它在 λ=0 时退化为 TD(0)，在 λ=1 时近似 MC → 即 **TD(λ)** 连接了两者。

五、用于 Q-Learning / SARSA 的 TD 版本

当学习动作价值函数时，TD 也衍生出两种常见算法：

| 算法                          | 更新方式                                                                         | 特点                              |
| --------------------------- | ---------------------------------------------------------------------------- | ------------------------------- |
| **SARSA (on-policy)**       | $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$           | 更新用的是当前策略产生的下一个动作 \(a'\)，稳定但收敛慢 |
| **Q-Learning (off-policy)** | $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ | 用最优动作估计更新，偏向贪心，更高效但可能不稳定        |

六、TD误差的直觉

TD 误差 $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$

- 如果 $δ_t > 0$ → 表示当前状态比预期更好 → **增加** $V(s_t)$；
- 如果 $δ_t < 0$ → 当前状态比预期更差 → **减少** $V(s_t)$。

神经科学上，这个信号被认为与**多巴胺神经元的激活模式**相对应。

七、在 Actor-Critic 框架中的作用

TD Learning 通常用于 **Critic（价值估计器）**：
$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

然后用该 TD 误差去更新 **Actor（策略）**：
$$
\nabla_\theta J(\theta) \approx \mathbb{E}[\delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)]
$$

→ Critic 用 TD 学习更新；Actor 用 TD 误差指导策略提升。
八、小结

| 方面 | 内容 |
|------|------|
| 核心思想 | 通过“下一个状态的估计”来更新当前价值 |
| 优点 | 样本效率高，不需完整回合，实时学习 |
| 缺点 | 依赖估计值（有偏），需要平衡稳定性与收敛性 |
| 应用 | SARSA、Q-Learning、Actor-Critic、A3C、DDPG、SAC 等 |
### Double Q-learning(Double DQN)
标准 Q-learning 的目标：

$$ y = r + \gamma \max_{a'} Q(s', a') $$

由于 $\max$ 是凸函数，$\mathbb{E}[\max X] \geq \max \mathbb{E}[X]$（Jensen 不等式），样本噪声会被 $\max$ 放大，导致系统性过估计。

在DQN中，维护两套估计器 $Q_A$、$Q_B$。更新其中一个时：
- 用 $A$ 选择动作（$\arg\max$），用 $B$ 评价该动作（取值），或反之。

两种等价写法（对称交替）：

**更新 $Q_A$：**

$$ a^* = \arg\max_{a'} Q_A(s', a'), \quad y = r + \gamma Q_B(s', a^*) $$

$$ Q_A(s, a) \leftarrow Q_A(s, a) + \alpha \left[ y - Q_A(s, a) \right] $$

**更新 $Q_B$：**

$$ a^* = \arg\max_{a'} Q_B(s', a'), \quad y = r + \gamma Q_A(s', a^*) $$

$$ Q_B(s, a) \leftarrow Q_B(s, a) + \alpha \left[ y - Q_B(s, a) \right] $$

直觉：一个网络“挑动作”，另一个网络“报分数”，互相牵制，减少由 $\max$ 引起的乐观偏差。

	而在deep dqn中在，用 online 网络选动作、用 target 网络评价：

$$ a^* = \arg\max_{a'} Q_{\text{online}}(s', a') $$
$$ y = r + \gamma Q_{\text{target}}(s', a^*) $$

|项|Q-learning|Double Q-learning|
|---|---|---|
|目标|$r + \gamma \max_{a'} Q(s', a')$|$r + \gamma Q_{\text{other}}(s', \arg\max_{a'} Q_{\text{self}}(s', a'))$|
|偏差|易过估计|明显缓解过估计|
|估计器|1 套|2 套（交替更新）|
|稳定性|较差（尤其函数逼近时）|更稳（Deep 版本即 Double DQN）|
关于$\epsilon-greedy$ 行为策略代码：
```
初始化 Q_A, Q_B (全零或小随机)
for 每个episode:
  s = s0
  while not terminal:
    以 ε-贪婪 按 Q_组合(s,·) 选 a    # 例如 Q_组合=Q_A+Q_B 或随机择一
    执行动作，得到 r, s'
    随机选择要更新的表：若 coin()==A:
        a_star = argmax_a' Q_A(s', a')
        y = r + γ * Q_B(s', a_star)
        Q_A(s,a) ← Q_A(s,a) + α * (y - Q_A(s,a))
    否则（更新 B）同理交换 A/B
    s = s'
```
在浏览作业的时候，可以着重了解一下以下的代码逻辑，不需要手搓：
- Exploration scheduling for ϵ-greedy actor
- Learning rate scheduling
- Gradient clipping
完成以下代码：
- 完成`update_critic`中的DQN部分critic
- 完成`get_action`中的$\epsilon-greddy$
- 完成`run_hw3_dqn.py`中的TODO部分
- 调用被要求的`update`，并在必要时更新目标critic
