[CS285_Notes.pdf](https://patrickyin.me/undergrad_notes/img/CS285_Notes.pdf)
考虑一个离散MDP过程，有时间跨度$T$，存在一个专家策略$\pi^*$，模仿学习的目标是学习一个策略$\pi^{\theta}$，尽量模仿专家，使其满足：
$$
\mathbb{E}_{p_{\pi^*}(s)} \pi_\theta(a \neq \pi^*(s) \mid s) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}_{p_{\pi^*}(s_t)} \pi_\theta(a_t \neq \pi^*(s_t) \mid s_t) \leq \varepsilon,
$$
- $p_{\pi^*}(s_t)$：表示在专家策略 $\pi^*$ 下，第 $t$ 步的状态分布。  
- $\pi_\theta(a_t \neq \pi^*(s_t) \mid s_t)$：表示在状态 $s_t$ 下，模仿策略与专家动作不同的概率。

整体式子的意思是：

> 在专家生成的状态分布下，模仿策略 $\pi_\theta$ 选择与专家动作不一致的概率，在所有时间步上的平均值，不超过某个小误差 $\varepsilon$。

也就是说，$\pi_\theta$ 学得“看起来和专家差不多”，但允许有少量误差。

求证：$\sum_{s_t}|p_{\pi_{\theta}}(s_t)-p_{\pi^*}(s_t) |\leq T\epsilon$.
证明思路：耦合+并集界
1. 由题设得到每步“分歧事件”的概率约束
	令
	$$E_i = \{a_i^{\theta} \neq \pi^*(s_i)\},\quad s_i~p_{\pi^*}(s_i).
	$$
	题设给出：
	$$\frac 1T\sum_{i=1}^T \Pr(E_i)\leq \epsilon$$
	故$$\sum_{i=1}^T\Pr(E_i)\leq T\epsilon.$$
2. 耦合引理把分布差异变成“轨迹是否分叉”的概率  
	考虑一对耦合轨迹 $(s_1^*, a_1^*, \ldots, s_t^*)$ 与$(s_1^\theta, a_1^\theta, \ldots, s_t^\theta)$，  
	其中环境随机性共享。若在前 $t-1$ 步里从未发生分歧事件 $E_i$，  
	则两条轨迹的状态在第 $t$ 步必相同：

$$
\neg\left(\bigcup_{i=1}^{t-1} E_i\right) \Rightarrow s_t^\theta = s_t^*.
$$

	由耦合引理（全变差距离上界）：

$$
\frac{1}{2} \sum_{s_t} \left| p_{\pi_\theta}(s_t) - p_{\pi^*}(s_t) \right| \leq \Pr\left[\bigcup_{i=1}^{t-1} E_i\right].
$$

3. 并集界（union bound）

$$
\Pr\left[\bigcup_{i=1}^{t-1} E_i\right] \leq \sum_{i=1}^{t-1} \Pr(E_i) \leq \sum_{i=1}^{T} \Pr(E_i) \overset{(\star)}{\leq} T\varepsilon.
$$

4. 合并得到结论

$$
\sum_{s_t} \left| p_{\pi_\theta}(s_t) - p_{\pi^*}(s_t) \right| \leq 2\Pr\left[\bigcup_{i=1}^{t-1} E_i\right] \leq 2T\varepsilon.
$$

这就证明了在专家分布下“平均每步分歧概率” $\leq \varepsilon$ 时，任意时刻 $t$ 的状态分布 $L_1$ 距离被 $2T\varepsilon$ 上界控制。

## 代码部分
CS285的hw1的starter code为OpanAI gym中的Mujoco任务给到了一个专家策略，填充代码中的TODO部分以实现一个模仿学习。
推荐按照顺序阅读以下几个文件：
- `scripts/run_hw1.py`（训练循环）  
- `policies/MLP_policy.py`（策略定义）  
- `infrastructure/replay_buffer.py`（存储训练轨迹）  
- `infrastructure/utils.py`（从策略中采样轨迹的工具函数）  
- `infrastructure/pytorch_utils.py`（用于在 NumPy 和 PyTorch 之间转换的工具函数）

对于某些文件，部分重要功能缺失，并用 `TODO` 标记。具体来说，你需要实现以下内容：

- `policies/MLP_policy.py`：`forward` 和 `update` 函数  
- `infrastructure/utils.py`：`sample_trajectory` 函数  
- `scripts/run_hw1.py`：`run_training_loop` 函数（你的大部分代码将写在这里）

在MLP_policy.py中，用深度神经网络生成一个平均值和方差，用于生成正态分布并采用。然后在update中，采用MSE作为损失函数，把policy根据观察生成的action和expert生成的action进行损失计算，再用AdamW优化器优化，最后返回这个损失函数。

在`utils.py`中要求补全`sample_trajectory`这个函数，其中先是由模拟环境中初始化，然后根据模拟环境的观察输入policy中得到action拼成轨迹

其中由于`Dagger`要不断将新的数据纳入训练集，所以不能像普通LM那样采用静态的训练集，所以采用一个缓冲区来储存数据，在`replay_buffer.py`中
