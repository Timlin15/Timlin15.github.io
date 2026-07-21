---
date: 2026-07-20
lastmod: 2026-07-21
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

AVDC 的大致思路相似，但是使用了光流作为关键表征，整套流程可只用 RGB(-D) 视频训练，完全不需要动作标签。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260720232734890.png)
(a) 输入当前 RGBD 观测 + 文本目标；
(b) 文本条件视频扩散模型合成一段"想象执行"视频；
(c) 光流预测模型估计相邻帧间的光流；
(d) 以光流作为帧间稠密对应，结合首帧深度，在刚体假设下反解出目标物体的 SE(3) 变换，进而换算为机械臂控制指令。为抑制光流/位置误差累积，引入闭环 **replanning** 策略（失败即重新生成视频与动作）。

## Joint WAM
直接建模联合分布 $p(o^′,a|o,l)$，状态预测与动作生成在共享表征空间中协同优化。一般这种 WAM 需要兼顾以下五种类型的模型：
1. forward dynamics，前向动力学，给定当前观测和动作，预测未来观测。
2. inverse dynamics，逆向动力学，给定当前与未来观测，反推中间的动作。
3. marginal action distribution (policy)，边际动作分布，把未来 $o'$ 边缘化掉，只由当前观测出动作。这就是标准策略，也是 VLA 的形态。
4. marginal image distribution (video generative model)，边际画面分布，把动作边缘化掉,只预测未来画面。这就是（动作无关的）视频生成 / 纯世界预测。
5. joint distribution，联合分布，同时生成未来观测与动作，两者都不作为条件而是共同被预测。

主要分为 Diffusion base 和 自回归系的

### Diffusion base
UWM, Unified World Stream 采用双流扩散模型，同时具有 action 和 video 模态，具有独立的时间步 $\tau_a$、$\tau_v$。
![1784563065576533859.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260720235746238.png)
核心是利用 $\tau=T$（纯噪声）等价于对该模态边缘化（不提供），$\tau=0$（无噪声）等价于将其作为条件输入。通过独立调节两个时间步，可在 policy（video→action）、forward dynamics（action→video）、video prediction 等模式间连续切换，无需为每种模式单独设计架构。

UWM 有两个输出头以及两个模态输入：  
- 一个头输出action 的去噪预测  
- 一个头输出video/image 的去噪预测  
  
当图像时间固定并且清晰，则可以预测动作action，就变成了普通的policy模型。  
当图像时间固定为 $T$，则变成了逆向动力学模型，即给定当前状态和动作预测下一帧。

在训练中，使用对两个时间步随机采样，随机混合的方式进行训练。

Motus 采用 MoT（Mixture-of-Transformer）设计，并且使用光流作为表征。本文章也是直接继承自上一篇UWM。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260721101721736.png)
MoT 设计集成 understanding、action、video generation 三个专家，经 tri-model joint attention 交互。采用 UniDiffuser 风格调度器，video 与 action 各自拥有独立时间步 $\tau_v$、$\tau_a$​

其中，模型对光流采用类似 VAE 风格的 压缩处理，将不同的光流压缩到codebook中，并且在 Action Expert中使用 latent action 进行训练
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260721102546025.png)


WorldVLA，使用Transformer base 的方法，将图像、文本、动作各用**独立 tokenizer** 编码，但三种模态 token **共享同一个词表空间**，从而可在统一的 LLM 式架构中直接做下一 token 预测，无需单独的 latent action 中间层。

单一自回归 Transformer，以OpenVLA的方式将动作变为 Token，使用VQ-GAN编码图片。通过数据配比的方式，同时对世界模型和动作模型类型数据进行训练：
```
样本类型 A（Action Model 数据）：
[BOS]{text: "What action should robot take to <task>?"}[BOI]{image}...{image}[EOI][EOS]
[BOA]{action}...{action}[EOA][EOS]
   → 只在这些 action token 位置计算 loss

样本类型 B（World Model 数据）：
[BOS]{text: "Generate the next frame..."}[BOI]{image}[EOI][BOA]{action}[EOA][EOS]
[BOI]{image}[EOI][EOS]
   → 只在这些 image token 位置计算 loss
```
