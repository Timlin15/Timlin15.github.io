---
date: 2026-03-07
lastmod: 2026-03-16
---
在多个智能体的情况下进行强化学习，MDP和policy gradiant都不能很好胜任，因此[Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning | PDF](https://arxiv.org/pdf/2109.11251)提出HATRPO/HAPPO训练方式。同时参考了Numerical Optimization

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
### 信赖域算法
信赖域方法是一类用于数值优化的迭代算法。它的核心思想是：  
- 在每一步迭代时，不是直接在全局搜索下降方向，而是在当前点附近建立一个**局部近似模型**（通常是二次模型），然后只在一个“可信”的区域（trust region）内对这个近似模型进行优化。
定理： 设 $\pi$ 是当前策略，$\bar{\pi}$ 是下一个候选策略。我们定义 $L_\pi(\bar{\pi}) = J(\pi) + \mathbb{E}_{s\sim\rho_\pi, a\sim\bar{\pi}} [A_\pi(s, a)]$， $D^{max}_{KL}(\pi, \bar{\pi}) = \max_s D_{KL}(\pi(\cdot|s), \bar{\pi}(\cdot|s))$。$L$是用上一步的策略来构造近似下一步的奖励函数；$D$ 指 **KL 散度**（常写 $D_{\mathrm{KL}}$​），用来度量**新旧策略的分布差**，在信赖域法里作为**约束/惩罚**控制更新“别走太远”。
那么以下不等式成立： $$J(\bar{\pi}) \geq L_\pi(\bar{\pi}) - C D^{max}_{KL}(\pi, \bar{\pi})$$ 其中 $$C = \frac{4\gamma \max_{s,a} |A_\pi(s,a)|}{(1-\gamma)^2}$$
所以当当前策略$\pi$和下一步策略$\bar{\pi}$距离很近的时候，只根据上一步推断出来的$L_{\pi}(\bar{\pi})$会和$J(\bar{\pi})$非常接近。所以agent可以通过信赖域来迭代其策略：
$$
π_{k+1} = \arg \max_{\pi} \left( L_{\pi_k}(\pi) - C \mathbf{D}_{KL}^{\max}(\pi_k, \pi) \right).
$$
但是这种方法并不实用，计算困难，论文提出TRPO算法，即：
$$θ_{k+1} = \arg \max_{θ} L_{π_{θ_k}}(π_θ), \quad \text{subject to } \mathbb{E}_{s \sim ρ_{π_{θ_k}}} \left[ D_{KL}(π_{θ_k}, π_θ) \right] ≤ δ.$$
每一次迭代，TRPO在策略$\pi_{\theta_k}$构建一个KL球$\mathcal{B}_\delta(\pi_{\theta_k})$，使得$L{\pi_{\theta_k}}(\pi_{\theta})$和真实奖励函数$J(\pi_{\theta})$相近。为了减轻计算散度的期望的计算负担，论文提出了PPO算法：
$$L_{π_{θ_k}}^{PPO}(π_θ) = \mathbb{E}_{s \sim ρ_{π_{θ_k}}, a \sim π_{θ_k}} \left[ \min \left( \frac{π_θ(a|s)}{π_{θ_k}(a|s)} A_{π_{θ_k}}(s, a), \text{clip} \left( \frac{π_θ(a|s)}{π_{θ_k}(a|s)}, 1 - ε, 1 + ε \right) A_{π_{θ_k}}(s, a) \right) \right].$$
- $r_\theta(s,a)=\frac{π_θ(a|s)}{π_{θ_k}(a|s)} A_{π_{θ_k}}(s, a)$：**策略比**（新/旧策略在同一 $(s,a)$ 上的相对概率）。
- $A_{\pi_{\theta_k}}(s,a)$：**优势函数**（常用 GAE 估计）。
- $\mathrm{clip}(r,1\!\pm\!\epsilon)=\min(\max(r,1-\epsilon),1+\epsilon)$：把 $r$限制在 $[1-\epsilon,1+\epsilon]$。
- 外层 $\min(\cdot,\cdot)$：在“未裁剪值”和“裁剪后值”之间取**更保守**的那个，避免过度乐观。

- $A>0$（动作优于平均）：希望 **增大** 其概率（$r\uparrow$）。若 r>1+ϵr>1+\epsilon，被截断为 $(1+\epsilon)A$，通过 $⁡\min$ **限制上涨幅度**。
- $A<0$（动作劣于平均）：希望 **降低** 其概率（$r\downarrow$）。若 $r<1-\epsilon$，被截断为 $(1-\epsilon)A$，通过 $\min$ **限制下跌幅度**。
### 信赖域算法在MARL中的应用
一种原始的应用方法是直接共享参数，用聚合轨迹进行策略训练，这个方法由MAPPO提出：
$$L_{π_{θ_k}}^{MAPPO}(π_θ) = \sum_{i=1}^{n} \mathbb{E}_{s \sim ρ_{π_{θ_k}}, \mathbf{a} \sim π_{θ_k}} \left[ \min \left( \frac{π_θ(\mathbf{a}^i|s)}{π_{θ_k}(\mathbf{a}^i|s)} A_{π_{θ_k}}(s, \mathbf{a}), \text{clip} \left( \frac{π_θ(\mathbf{a}^i|s)}{π_{θ_k}(\mathbf{a}^i|s)}, 1 - ε, 1 + ε \right) A_{π_{θ_k}}(s, \mathbf{a}) \right) \right].$$
但是MAPPO有致命的缺陷：参数共享决定了智能体只能有相同的action space，可能导致并不能找到最优策略。因此论文提出可以使用HAPPO和HATRPO算法。

**多智能体的优势函数**
在任何一个合作马可夫游戏中，给定一个联合策略$\pi$，对于任何状态$s$，以及任何智能体子集$i_{1:m}$，定义如下方程：
$$
\begin{aligned}
A_{π}^{i1:m}(s, a^{i1:m}) &= \sum_{j=1}^{m} A_{π}^{ij}(s, a^{i1:j-1}, a^{ij}).\\
A_{π}^{ij}(s, a^{i1:j-1}, a^{ij}) &= Q_{π}(s, \mathbf{a}(\{i1:j\})) - Q_{π}(s, \mathbf{a}(\{i1:j-1\})).
\end{aligned}
$$
- 求和符号右侧式子表示**一组代理 $i_{1:m}$​** 同时把动作从“旧策略的基线动作”换成给定的新动作 $a^{i_{1:m}}$​ 时产生的**联合优势**（对旧策略 $\pi$ 而言）。
- 等式右侧表示前 $j-1$ 个代理已用新动作 $a^{i_{1:j-1}}$​，再让第 $j$ 个代理把动作改为 $a^{i_j}$​ 所带来的**边际优势**；把这些边际优势从 $j=1$ 到 $m$ 加起来，恰好等于“所有人一起改”的联合优势。
设 π 是一个联合策略，$\bar{\pi}^{i1:m-1} = \prod_{j=1}^{m-1} \bar{\pi}^{ij}$ 是其他代理 $i_{1:m-1}$ 的某个其他联合策略，而 $\hat{\pi}^{im}$ 是代理 $i_m$ 的某个其他策略。那么 
$$L_{\pi}^{i1:m}\left(\bar{\pi}^{i1:m-1}, \hat{\pi}^{im}\right) \triangleq \mathbb{E}_{s \sim \rho_{\pi}, \mathbf{a}^{i1:m-1} \sim \bar{\pi}^{i1:m-1}, a^{im} \sim \hat{\pi}^{im}}\left[A_{\pi}^{im}\left(s, \mathbf{a}^{i1:m-1}, a^{im}\right)\right]$$
请注意，对于任何 $\bar{\pi}^{i1:m-1}$，我们有 
$$
\begin{aligned}
L_{\pi}^{i1:m}\left(\bar{\pi}^{i1:m-1}, \pi^{im}\right) &= \mathbb{E}_{s \sim \rho_{\pi}, \mathbf{a}^{i1:m-1} \sim \bar{\pi}^{i1:m-1}, a^{im} \sim \pi^{im}}\left[A_{\pi}^{im}\left(s, \mathbf{a}^{i1:m-1}, a^{im}\right)\right]\\
&= \mathbb{E}_{s \sim \rho_{\pi}, \mathbf{a}^{i1:m-1} \sim \bar{\pi}^{i1:m-1}}\left[\mathbb{E}_{a^{im} \sim \pi^{im}}\left[A_{\pi}^{im}\left(s, \mathbf{a}^{i1:m-1}, a^{im}\right)\right]\right] = 0
\end{aligned}
$$
- 含义：在旧策略的状态分布 $π\rho_\pi$​ 下，让前 $m−1$ 个代理按 $\bar\pi$ 出动作，第 $i_m$​ 个代理按 $\hat\pi$ 出动作，计算“**第 $i_m$​ 个代理的边际优势**”的期望。它是一个**局部/代理目标**，衡量“把第 $i_m$​ 个体从旧策略换成 $\hat\pi$ 的收益”，条件是其他体用 $\bar\pi$。即用每个智能体的更新策略的优势函数加和表示代理函数（在常见实践中）。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20250920162012.png)
### HATRPO/HAPPO
算法1使用的是散度$D_{KL}^{max}$，难估计且不光滑。同TRPO中的方法，将这个约束转为
$$
\mathbb{E}_{s \sim \rho_{π_{θ_k}}} \left[ D_{KL}(\pi_{θ_k}^{i_m}(·|s) \| π_{θ}^{i_m}(·|s)) \right] ≤ δ.$$
最后的目标变成了求以下这个目标的最大值：
$$
\begin{aligned}
θ_{k+1}^{i_m} &= \arg \max_{θ^{i_m}} \mathbb{E}_{s \sim ρ_{π_{θ_k}}, \mathbf{a}^{i1:m-1} \sim π^{i1:m-1}, a^{im} \sim π_{θ^{i_m}}^{im}} \left[ A_{π_{θ_k}}^{im}(s, \mathbf{a}^{i1:m-1}, a^{im}) \right], \\&subject\,\, to\,\, \mathbb{E}_{s \sim ρ_{π_{θ_k}}} \left[ D_{KL}(π_{θ_k}^{im}(·|s) \| π_{θ^{i_m}}^{im}(·|s)) \right] ≤ δ.
\end{aligned}
$$
然后同TRPO一样：
- 把目标在$\theta=\theta_k^{\,i_m}$​​ 处做**一阶近似**，梯度记为 $\mathbf g_k^{\,i_m}$；
- 把期望 $KL$ 在该点做**二阶近似**，Hessian 即 Fisher 信息矩阵 $\mathbf H_k^{\,i_m}​​$。$H_{k}^{i_m} = \nabla_{\theta^{i_m}}^2 \mathbb{E}_{s \sim \rho_{\pi_{\theta_k}}} \left[ D_{KL}(\pi_{\theta^{i_m}_k}^{i_m}(\cdot|s), \pi_{\theta^{i_m}}^{i_m}(\cdot|s)) \right] \bigg|_{\theta^{i_m}=\theta_k^{i_m}}$

$$
θ_{k+1}^{i_m} = θ_k^{i_m} + α^j \sqrt{\frac{2δ}{g_k^{i_m} (H_k^{i_m})^{-1} g_k^{i_m}}} .
$$
最后一步是求$\mathbb{E}_{\mathbf{a}^{i1:m-1} \sim \pi_{\theta_k^{i1:m-1}}, a^{im} \sim \pi_{\theta^{im}}^{im}} \left[ A_{\pi_{\theta_k}}^{im}(s, \mathbf{a}^{i1:m-1}, a^{im}) \right],$之后没看懂总之
HAPPO的目标是最大化
$$\mathbb{E}_{s \sim \rho_{\pi_{\theta_k}}, \mathbf{a} \sim \pi_{\theta_k}} \left[ \min \left( \frac{\pi_{\theta^{i_m}}^{i_m}(a^{i_m}|s)}{\pi_{\theta_k^{i_m}}^{i_m}(a^{i_m}|s)} M^{i1:m}(s, \mathbf{a}), \text{clip}\left(\frac{\pi_{\theta^{i_m}}^{i_m}(a^{i_m}|s)}{\pi_{\theta_k^{i_m}}^{i_m}(a^{i_m}|s)}, 1 \pm \epsilon \right) M^{i1:m}(s, \mathbf{a}) \right) \right].$$


