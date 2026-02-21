要求完成：
- `cs285/scripts/run_hw2.py`
- `cs285/agents/pg_agent.py`
- `cs285/networks/policies.py`   
- `cs285/networks/critics.py`    
- `cs285/infrastructure/utils.py`
## 强化学习中的策略梯度方法（Policy Gradient） 
回顾强化学习的目标是学习一个最优策略参数 $\theta^*$，使得目标函数最大化： 
$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[r(\tau)] $$ 其中每个轨迹 $\tau$ 的长度为 $T$，定义如下： $$ \pi_\theta(\tau) = p(s_0, a_0, \ldots, s_{T-1}, a_{T-1}) = p(s_0)\pi_\theta(a_0|s_0)\prod_{t=1}^{T-1} p(s_t|s_{t-1}, a_{t-1})\pi_\theta(a_t|s_t) $$ 以及奖励函数： $$ r(\tau) = r(s_0, a_0, \ldots, s_{T-1}, a_{T-1}) = \sum_{t=0}^{T-1} r(s_t, a_t). $$ 策略梯度方法直接对上述目标函数求梯度： 
$$ 
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \int \pi_\theta(\tau) r(\tau) d\tau  \\
 &= \int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) r(\tau) d\tau. \\
 & = \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[\nabla_\theta \log \pi_\theta(\tau) r(\tau)] 
\end{aligned}
$$ 实际应用中的近似计算 在实际中，轨迹 $\tau$ 上的期望可以通过采样一批 $N$ 条轨迹来近似： 
$$
\begin{aligned}
\nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log \pi_\theta(\tau_i) r(\tau_i)\\
&  = \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{it}|s_{it}) \right) \left( \sum_{t=0}^{T-1} r(s_{it}, a_{it}) \right). 
\end{aligned}
$$我们可以看到，策略 $\pi_\theta$ 是在给定状态 $s_t$ 下，动作空间上的概率分布。在智能体与环境的交互循环中，智能体会从 $\pi_\theta(\cdot|s_t)$ 中采样动作 $a_t$，而环境则会以奖励 $r(s_t, a_t)$ 做出响应。

## 方差降低（Variance Reduction） 
### 未来奖励（Reward-to-go） 
一种降低策略梯度方差的方法是利用“因果性”：策略无法影响过去已发生的奖励。这引出了一个修改后的目标函数，其中奖励的总和不包括在查询策略时刻之前已经获得的奖励。这个奖励之和是对 $Q$ 函数的一个采样估计，被称为“未来奖励”（reward-to-go）。 $$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{it}|s_{it}) \left( \sum_{t'=t}^{T-1} r(s_{it'}, a_{it'}) \right) $$
### 折扣因子（Discounting） 
将折扣因子 $\gamma$ 应用于奖励可以被理解为鼓励智能体更关注时间上更近的奖励，而减少对遥远未来的奖励的关注。这也可视为一种降低方差的手段（因为越远的未来具有更大的不确定性，从而导致更高的方差）。 我们在课程中了解到，折扣因子可以通过两种方式引入： 第一种方式是对整条轨迹的奖励进行折扣： $$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{it}|s_{it}) \right) \left( \sum_{t'=0}^{T-1} \gamma^{t'-t} r(s_{it'}, a_{it'}) \right) $$ 第二种方式是在“未来奖励”上应用折扣： $$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{it}|s_{it}) \left( \sum_{t'=t}^{T-1} \gamma^{t'-t} r(s_{it'}, a_{it'}) \right) $$
### 基线（Baseline） 
另一种降低方差的方法是从奖励总和中减去一个基线（即关于轨迹 $\tau$ 的常数项）： $$ \nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ r(\tau) - b \right] $$ 这样不会改变策略梯度的无偏性，因为： $$ \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta(\tau)} [b] = \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ \nabla_\theta \log \pi_\theta(\tau) \cdot b \right] = 0 $$
有了无偏性，可以利用这个项来降低方差，即求：
$$
Var[u_t(G_t-b)]\quad\quad最小
$$
在本任务中，我们将实现一个值函数 $V_\phi^\pi$，作为**状态依赖的基线**。该值函数将被训练以近似从某个特定状态开始的未来奖励总和： 
$$ V_\phi^\pi(s_t)= \frac{\mathbb{E}[(G_t)||u_t||^2\,|s_t]}{\mathbb{E}[||u_t||^2\,|s_t]} \approx \sum_{t'=t}^{T-1} \mathbb{E}_{\pi_\theta} \left[ r(s_{t'}, a_{t'}) \mid s_t \right] $$ 因此，近似的策略梯度变为如下形式： $$ \begin{aligned} \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{it}|s_{it}) \\ &\quad \left( \sum_{t'=t}^{T-1} \gamma^{t'-t} r(s_{it'}, a_{it'}) - V_\phi^\pi(s_{it}) \right) \end{aligned} $$
### 广义优势估计（Generalized Advantage Estimation） 
在之前策略梯度表达式中（为清晰起见省略了索引 $i$）的量： $$ \left( \sum_{t'=t}^{T-1} \gamma^{t'-t} r(s_{t'}, a_{t'}) \right) - V_\phi^\pi(s_t) $$ 可以被解释为对优势函数 $A^\pi(s_t, a_t)$ 的估计： $$ A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) $$ 其中 $Q^\pi(s_t, a_t)$ 通过蒙特卡洛回报进行估计，而 $V^\pi(s_t)$ 则使用学习到的价值函数 $V_\phi^\pi$ 进行估计。我们可以进一步降低方差，通过使用 $V_\phi^\pi$ 在蒙特卡洛回报中的估计来估算优势函数： $$ A^\pi(s_t, a_t) \approx \delta_t = r(s_t, a_t) + \gamma V_\phi^\pi(s_{t+1}) - V_\phi^\pi(s_t) $$ 其中边界情况为 $\delta_{T-1} = r(s_{T-1}, a_{T-1}) - V_\phi^\pi(s_{T-1})$。然而，这可能会以引入偏差为代价影响我们对策略梯度的估计，因为 $V_\phi^\pi$ 是从数据中学习得到的。 我们可以改用 $n$ 步蒙特卡洛回报和 $V_\phi^\pi$ 的组合来估计优势函数： $$ A_n^\pi(s_t, a_t) = \sum_{t'=t}^{t+n} \gamma^{t'-t} r(s_{t'}, a_{t'}) + \gamma^n V_\phi^\pi(s_{t+n+1}) - V_\phi^\pi(s_t) $$ 增加 $n$ 会使蒙特卡洛回报在优势估计中占比更大，从而降低偏差但增加方差；减小 $n$ 则相反。注意当 $n = T - t - 1$ 时，恢复为无偏但高方差的蒙特卡洛优势估计（如公式 (13) 所示）；而当 $n = 0$ 时，则恢复为低方差但高偏差的优势估计 $\delta_t$。 我们可以将多个 $n$ 步优势估计以指数加权的方式组合起来，这种方法被称为广义优势估计（GAE）。令 $\lambda \in [0, 1]$，则定义： $$ A_{GAE}^\pi(s_t, a_t) = \frac{1 - \lambda^{T-t-1}}{1 - \lambda} \sum_{n=1}^{T-t-1} \lambda^{n-1} A_n^\pi(s_t, a_t) $$ 其中 $\frac{1 - \lambda^{T-t-1}}{1 - \lambda}$ 是一个归一化常数。注意：$\lambda$ 越大，越强调具有更高 $n$ 值的优势估计；$\lambda$ 越小则相反。因此，$\lambda$ 作为偏差-方差权衡的控制参数：增大 $\lambda$ 会降低偏差但增加方差。 在无限时间范围的情况（$T = \infty$）下，我们有： $$ \begin{aligned} A_{GAE}^\pi(s_t, a_t) &= \frac{1}{1 - \lambda} \sum_{n=1}^{\infty} \lambda^{n-1} A_n^\pi(s_t, a_t) \\ &= \sum_{t'=t}^{\infty} (\gamma \lambda)^{t'-t} \delta_{t'} \end{aligned} $$其中我们省略了推导过程以简洁起见（详见 [GAE 论文](https://arxiv.org/pdf/1506.02438.pdf)）。 在有限时间范围的情况下，我们可以写成： 
$$ A_{GAE}^\pi(s_t, a_t) = \sum_{t'=t}^{T-1} (\gamma \lambda)^{t'-t} \delta_{t'} $$
这提供了一种高效实现广义优势估计的方法，因为我们可以通过递归计算： $$ A_{GAE}^\pi(s_t, a_t) = \delta_t + \gamma \lambda A_{GAE}^\pi(s_{t+1}, a_{t+1}) $$
