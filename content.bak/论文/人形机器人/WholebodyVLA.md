见[arxiv网址](http://arxiv.org/abs/2512.11047)

本篇论文主要是为了解决两个问题：
- 人形机器人遥操作数据缺失导致的loco-manipulate（移动-操作）知识稀缺，即全身协调能力不足
- RL算法准确度稳定性缺失导致的无法可靠执行移动操作
为了解决这两个问题，论文提出了
- 使用基于VQ-VAE的LAM模型来从大量的人类数据和机器人数据中学习，并用LAM(latent action model)模型训练VLA模型
- 采用LMO RL策略来优化全身控制，获得稳定的移动-操作动作

## 模型架构
为了将大量的人类动作egocentric画面转化为可供VLA训练的数据，论文采用将人类egocentric画面转化为**discrete latent space**(离散潜在空间)的方式来转化为训练数据。
为了增强模型在移动和操作两方面的能力，同时因为移动和操作两种数据有很大差距（e.g. 操作中镜头为静态的，而移动中镜头经常移动），文章训练了两种LAM来将不同类型的数据转为离散潜在动作。

为了让VLM能通过两种LAM生成的数据进行训练，论文将VLM训练成能同时生成两种潜在动作，使得可以使用两个LAM生成的数据进行训练。对于locomotion的LAM，主要使用人类egocentric的画面训练，而manipulate的LAM主要使用AgiBot的数据
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260203210924907.png)

可以看到VLM生成同时生成了manipulate和locomotion两种latent action，之后经过Action Decoder后，上半身（基本等同manipulate）的动作直接由Action Decoder输出，而下半身locomotion的数据会经过一个LMO RL算法之后再输入下半身。

### LAM设计
此处没有完全明白，未来要用到可以再细化理解，先简单介绍一下。
总的来说，LAM的设计采用了VQ-VAE架构，在经典VAE上面加上了一个codebook。
```
输入 x → [Encoder] → 连续潜在向量 z (高斯分布采样) → [Decoder] → 重建 x̂
```
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260203220159198.png)
将连续的帧$(o_t,o_{t+k})$输入encoder DINOv2后生成一个连续的隐含向量（latent vector)$z_t=E_i(o_t, o){t+k}$，之后在codebook找最近近邻：
$$
c_t^i = \arg \min_{c \in \mathcal{C}^i} \|z_t - c\|_2, \quad c_t \in \mathcal{C}_i. \quad i \in \{\text{mani, loco}\}
$$
为了训练Encoder和Codebook，Decoder会获取到前一帧和量化的latent action，然后训练着去重建下一帧，重建用标准VQ-VAE损失优化。

然后便可以用LAM去优化VLA的策略，来基于视觉观察和prompt预测locomotion和manipulation的latent action并用交叉熵进行优化，即最大化：
$$
\pi_{\theta}(c_t^{mani},c_t^{loco}|o_t, \ell)
$$
然后用一个decoder转化为机器人指令，论文中没有强调action decoder的架构，只说了这是一个轻量级（lightweight）的decoder。
### 离散化的动作设计
之所以采用离散化的action token设计是因为论文认为：
- 普通locomotion RL是基于追踪连续的速度的，这需要学习无限多种可能性
- 这个目标超过了loco-manipulation的实际需要，尽管可能符合locomotion的需要
- 预测连续的速度使得模型更难以训练并更不可靠了。
同时，采用离散化的动作token设计可以极大加快收敛速度，如PI0FAST中的这个图
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260203213755966.png)

### LMO RL 策略
同上文所讲，该论文放弃了用RL输出特定的速度，而是采用离散化的指令：
$$
u_t = [s_x, s_y, s_ψ, h*] ∈ {-1, 0, 1}³ × ℝ
$$
其中： 
- sx: 前进/后退 (-1, 0, 1) 
- sy: 左移/右移 (-1, 0, 1) 
- sψ: 左转/右转 (-1, 0, 1) 
- h*: 站立高度 (连续值，用于蹲下)

观测空间为：
$$
O_t = [u_t, ω_t, g_t, q_t, q̇_t, a_{t-1}]
$$
- ut: 命令 
- ωt: 基座角速度 
- gt: 重力向量
- qt, q̇t: 关节位置和速度 
- at-1: 上一步动作

采用两阶段训练：

| **阶段**       | **目标** | **方法**                      |
| ------------ | ------ | --------------------------- |
| **Stage I**  | 基础步态   | 随机采样速度目标，上半身跟踪随机姿态，逐渐放宽关节限制 |
| **Stage II** | 精度+稳定性 | 固定巡航速度，加入方向精度惩罚，注入真实操作扰动    |
