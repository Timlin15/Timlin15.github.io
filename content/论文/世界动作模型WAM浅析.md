---
date: 2026-07-20
lastmod: 2026-07-20
---
本文根据[Awesome-WAM GitHub仓库/论文](https://github.com/OpenMOSS/Awesome-WAM)进行整理阅读WAM模型。该论文将WAM分为两类，分别是
- **Cascaded WAM**：显式因式分解 $p(o^′,a|o,l)=p(a|o^′,o,l)·p(o^′|o,l)$，先合成未来状态表征，再从中推导动作，两阶段解耦。其中再细分为
	- **Explicit**：以渲染出的像素/几何作为未来中介
	- **Implicit**：以隐/预测性表征而非渲染像素作为未来，用于条件化策略
- **Joint WAM** —— 直接建模联合分布 $p(o^′,a|o,l)$，状态预测与动作生成在共享表征空间中协同优化。
	- **Autoregressive Generation**：离散 token 自回归
	- Diffusion-based Generation：分为Unified Stream 和 MultiStream
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260720181709150.png)
相关：[[Diffusion & Flow model]], [[CS336 Assignment 1]]
## Cascaded WAM
### Implicit(LAPA, villa-X)
Implicit：以隐/预测性表征而非渲染像素作为未来，用于条件化策略。训练出世界模型后使用其对数据集进行标注，最后使用标注的模型训练动作生成模型。

以LAPA为例，这种模型可以看到比较明显的 VQ-VAE 模型的特点，即运用一个 encoder 和 decoder，将动作化为 Codebook 中的一个离散化 token，同时训练两个模型，Encoder 采用 C-ViViT（时空 Transformer）变体，接收当前帧 $x_t$ 和未来帧 $x_{t+H}$ 经交叉注意力（令 $z_t$​ 关注 $x_t$​，而非简单加性嵌入）+ NSVQ 量化输出离散 $z_t$​；decoder 由 $z_t$​ 与 $x_t$​ 重建 $x_{t+H}$
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260720211152645.png)
模型训练分为三部分：
1. latent 量化：使用语言标注的视频
2. latent 预训练：冻结 encoder 作 IDM 标注视频，训 VLM 预测 $z_t$
3. 动作微调：丢弃 latent head，用少量真实动作数据训新 head，这里的动作效仿 OpenVLA，将动作化为离散的bin值

villa-X 在 LAPA 上则是进行了些许改进，具体来说是补足了 LAPA 仅仅依赖视觉短板，增加了对机器人本体关节状态的建模。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260720212258365.png)
除了 IDM（逆向运动学模型/Encoder）和FDM（正向运动学模型/Decoder）外，模型还涉及了一个proprio FDM。其是两层 MLP，预测未来 K 时间的本体感知状态与动作，$c_e$ 是数据集和控制频率编码的一个向量，以剥离平台相关特征。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260720212911341.png)
ACT-robot 通过单向注意力**显式条件于** ACT-latent 的中间特征，且 latent 与真实动作序列长度不要求相等。训练时对 latent-to-robot 注意力做 50% 整体屏蔽 + 50% token 级随机屏蔽，防止真实动作专家退化到只依赖 latent 通道。

训练是采用开始训练的 LAM 标记的数据集给进行训练集标注，然后使用标注数据对 VLA 进行标注。

### Explicit(UniPi, AVDC)
以渲染出的像素/几何作为未来中介， 显性地从未来的预测生成动作。这两个工作也是相互继承的关系。

UniPi 把策略编码为像素空间的视频轨迹。视频（图像序列）被当作跨环境、跨具身的通用接口——用它同时承载观测与动作行为，动作信息隐式蕴含在帧间变化里，而非显式变量。

该模型第一阶段是一个文本条件视频扩散模型充当轨迹规划器，输入当前帧 + T5 文本目标特征，生成"完成该目标应有的轨迹视频"；视频经由时序超分逐级细化。第二阶段是一个任务独立、轻量的 inverse dynamics model（3×3 卷积 + 残差卷积 + 全局均值池化 + MLP，回归 7 维控制量），从相邻生成帧中回归底层动作。IDM 与规划器解耦，可在更小的仿真数据上单独训练。

并且使用了很多相关技巧比如首帧约束，Tiling 和 Hirerarchial Planning 优化生成效果，保证生成的图片不偏移初始状态。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260720224105099.png)

AVDC 的大致思路相似，但是使用了光流作为关键表征