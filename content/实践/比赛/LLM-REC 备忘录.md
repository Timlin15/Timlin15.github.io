---
date: 2026-07-12
lastmod: 2026-07-12
---
## 背景文档/仓库
文档：[语雀文档](https://docs.qingque.cn/d/home/eZQAVR10RNyGdtIrSA08YFfr5?identityId=2J04lZy9e9D#section=h.hm62y5djc9a6)
云服务参考：[云计算资源文档](https://www.streamlake.com/document/WANQING/mq57afym1d7p20atnau)
比赛仓库：[比赛landing repo](https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/tree/main)
基模仓库：[基模huggingface](https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition/tree/main)
论文 Github：[基模huggingface](https://github.com/Kuaishou-OneRec/OpenOneRec/tree/main)
论文：[基模论文](https://arxiv.org/pdf/2606.06260)

组内记录：[云文档](https://365.kdocs.cn/l/crt5wflmDz8V)

## SFT 数据
比赛官方提供以下几类 SFT 数据： 
> 一类是 种子SFT 数据（该部分LLamaFactory框架直接可用，可在万擎平台下载），另一类是原始的用户行为序列以及物料元信息、以及通识数据（需选手自行处理，通过Huggingface下载）。其中种子SFT数据分为懂物料/懂用户/懂推荐三块，该批数据从原始的用户行为序列清洗得到，此处仅作为参考，了解OneReason的SFT数据格式，选手也可以自行进行额外清洗、配比等操作；对于原始数据，包含大量用户在短视频、电商、直播、广告四个领域的交互行为，每个行为序列的物品id经过哈希，可以通过pid2sid 表找到对应的sid，也可通过pid关联可能存在的caption和tag。详细介绍可以参考[https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition](https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition) 数据解释。选手可基于转化后的sid行为序列构造SFT数据，转换为LLamaFactory可用的数据格式，进行增量数据配比训练模型。

