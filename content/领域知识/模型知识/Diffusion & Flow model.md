> [!NOTE] Reference
> 参考了[Step-by-Step Diffusion: An Elementary Tutorial | PDF](https://arxiv.org/pdf/2406.08929)，[Flow Matching Guide and Code | PDF](https://arxiv.org/pdf/2412.06264)，[An Introduction to Flow Matching and Diffusion Models | PDF](https://arxiv.org/pdf/2506.02070)

# Diffusion model
生成式模型的**目标**是：
> 从多个在未知分布$p^*(x)$中独立同分布的数据中，提取出一个新的样本，是从分布$p^*$中提取的

Diffusion model也一样，它想从目标分布（如狗的图像）$p^*$中采样出一个点。但是从初始的简单分布（如高斯分布）中推导到复杂的目标分布根本不可能。所以Diffusion先反过来，研究怎么将一个多峰的复杂分布简化为一个简单的正态分布

![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20251025160740.png)

![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/20251025161523.png)
