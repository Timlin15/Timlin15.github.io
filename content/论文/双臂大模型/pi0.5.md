$\pi_{0.5}$的目的是构建一个健壮的VLA模型，能应对zero-shot场景。
先从模型架构说起，$\pi_{0.5}$是基于$\pi_0$的模型架构，即一个VLM加动作专家模组。pi0的部分有待之后补上
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260121161301023.png)
pi0.5在这基础上改进了pi0的模型，给它加上了高阶和低阶的部分，使得更好地进行规划：
$$
\pi_\theta(\mathbf{a}_{t:t+H}, \hat{\ell}|\mathbf{o}_t, \ell) = \pi_\theta(\mathbf{a}_{t:t+H}|\mathbf{o}_t, \hat{\ell})\pi_\theta(\hat{\ell}|\mathbf{o}_t, \ell)
$$
$\ell$是总体的任务提示词，$\hat{\ell}$是模型tokenized文字输出，$\mathbf{o}_t$是摄像头和关节信息，既可以是预测的高阶语义任务，也可以是针对vlm训练的答案。模型将动作分布分为上式中的两个部分，使得动作分布不依赖$\ell$（任务提示词）而是依赖于$\hat{\ell}$（文字输出）。这看着很反常，动作怎么可以不依赖提示词呢。其实这是为了实现长程任务做的设计。
对于不同的token，模型会使用不同的的encode方式，然后接入不同的专家权重，类似MoE？
由于训练过程中离散化的动作更快，但是推理中连续的动作更快，因此模型设计兼顾了自回归token化的动作和迭代式的flow model，训练则用
$$
\mathbb{E}_{D,\tau,\omega} \left[ H(x_{1:M}, f_\theta^\ell(\mathbf{o}_t, \ell)) \right. \left. + \alpha \left\| \omega - \mathbf{a}_{t:t+H} - f_\theta^a(\mathbf{a}_{t:t+H}^{\tau,\omega}, \mathbf{o}_t, \ell) \right\|^2 \right]
$$
其中$H(x_{1:M},y_{1:M}^{\ell})$是文本token和预测logits交叉熵损失，$y_{1:H}^a=f^a_{\theta}(a_{t:t+H}^{\tau,\omega},\mathbf{o}_t,\ell)$是动作专家的输出，其中$\alpha$是一个权衡两者损失的参数。
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260121170621517.png)
运行时在推理阶段，模型先推断语义上的子任务，推断下一个适合的动作，最后交给低阶动作模块。
$\pi_{0.5}$的训练也很关键：其先在异质信息上进行训练，然后再低阶动作信息和高阶语义信息上进行训练。同时为了减少计算量，其在预训练上用了自回归式token化的动作训练，后训练则使用flow model
因此可以说pi0.5的模型的关键创新一是特定配比的异信息进行训练，而是在前人work的基础上引入了语义层和 动作层用于表示思考过程。用了这么多异质训练数据，pi0.5是如何防止灾难性遗忘的呢，其在每一个训练阶段都纳入了语义训练信息和动作训练信息，防止灾难性遗忘。

其核心贡献为：在真实的随机全新场景中检测了vla的泛化能力；采用异质数据进行训练（co-training），包括非机器的信息，同时强调特定配比的数据的重要性；用高阶语义信息作为思考和规划，低阶动作信息用于传入动作专家执行。