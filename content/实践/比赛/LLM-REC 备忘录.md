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

## 基本概念
pid/ item id：两者是同一个东西，是真实世界里一条物料的唯一标识(样例里的 `1234`)经过哈希后的值。

sid/itemic pattern：三段式的 `[s_a, s_b, s_c]`，模型词表中真实存在的值
## SFT 数据
比赛官方提供以下几类 SFT 数据： 
> 一类是 种子SFT 数据（该部分LLamaFactory框架直接可用，可在万擎平台下载），另一类是原始的用户行为序列以及物料元信息、以及通识数据（需选手自行处理，通过Huggingface下载）。其中种子SFT数据分为懂物料/懂用户/懂推荐三块，该批数据从原始的用户行为序列清洗得到，此处仅作为参考，了解OneReason的SFT数据格式，选手也可以自行进行额外清洗、配比等操作；对于原始数据，包含大量用户在短视频、电商、直播、广告四个领域的交互行为，每个行为序列的物品id经过哈希，可以通过pid2sid 表找到对应的sid，也可通过pid关联可能存在的caption和tag。详细介绍可以参考[https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition](https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition) 数据解释。选手可基于转化后的sid行为序列构造SFT数据，转换为LLamaFactory可用的数据格式，进行增量数据配比训练模型。

**第一条：种子SFT数据（万擎平台下载，LLaMA-Factory 直接可用）**

这是官方已经从原始用户行为序列**清洗好、构造完成的成品 SFT 数据**，分三块，对应 OneReason 的三种训练目标：
- **懂物料**：物料理解类任务，让模型建立 "sid ↔ caption/tag（语义/类目）" 的对应关系，即理解每个 semantic ID 背后的内容是什么。
- **懂用户**：用户理解类任务，从行为序列中刻画用户兴趣/画像。
- **懂推荐**：推荐生成类任务，给定用户历史行为序列，预测/生成下一个要推荐的物品（next-item，以 semantic ID 形式输出）。

关键定性：这批数据**只是"参考样例"**，目的是让选手**看懂 OneReason 期望的 SFT 数据格式**。选手不必照单全收，可以自行做额外清洗、重新配比（三块之间的比例、与通识数据的比例等）。

**第二条：原始数据（HuggingFace 下载，需选手自行处理）**

这是没有加工过的底料，包含用户在**短视频 / 电商 / 直播 / 广告**四个域的交互行为序列，外加物料元信息和通识数据。核心的处理链路是：
- 序列里的物品是**哈希后的 pid**（int64），本身不可读；
- 通过 **`pid2sid` 表**把 pid 映射成三段式 semantic ID `[s_a, s_b, s_c]`（即模型词表里的 `<s_a_*><s_b_*><s_c_*>` token）；
- 也可通过 pid 关联 **caption（文本描述）** 和 **tag（三级类目）**。

Huggingface 仓库：

| File                     | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| `OneReason_UserProfile/` | Per-user multi-domain behavior records (~500k rows)   |
| `OneReason_Pid2Sid/`     | Mapping from content PID to three-segment semantic ID |
| `OneReason_Pid2Caption/` | Mapping from content PID to text caption              |
| `OneReason_Pid2Tag/`     | Mapping from content PID to level-3 category tag      |
| `OneReason_General/`     | General knowledge data                                |

## 模型测评
模型测评分为以下四个项目：

![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260712160047796.png)

### 懂物料
- **输入**:System(角色:视频数据分析专家)+ User(视频的纯文字内容描述)。模型是纯文本模态,看不到视频本身,只看描述。
- **生成**:模型用 **beam search 一次性生成 64 个候选 itemic pattern**(即 64 个 sid 串,如 `<|video_begin|><s_a_...><s_b_...><s_c_...>`)。beam search 是**单次解码**过程,通过维护束宽(beam width),在一次前向解码里就同时保留 64 条最可能的候选序列。不是把模型独立跑 64 次,而是一次解码吐出 top-64 个候选。
- **映射**:每个候选 sid 串通过官方的 `itemic pattern → item ids` 映射表,转成对应的 item id(= pid)。
- **判分**:把这 64 个预测出的 item id 和 ground truth item id 对比,**只要有一个命中就算 Pass(记 1),64 个全不中记 0**。
### 懂用户
懂用户分为
- 抽取兴趣相关行为
- 给定兴趣抽取逻辑链条
两个测试方向。
前者从用户交互历史和system prompt中提取出相关主题的动作。比如给定`主题：冬季取暖从内容接触到商品筛选并下单石墨烯电热膜`和用户交互历史，需要从交互历史中获取与主题直接相关的交互。最后和 ground truth 比较获得 F1 分数。

后者也从给定的用户交互历史和 system prompt 中提取出具有高阶逻辑演进与深度意图关联的交互行为链路。主要包括**场景需求补全**和**兴趣因果递进**这两个方面。详情见PDF。

### 懂推荐
懂推荐是**全域行为 → 预测 next item**,即给定用户在四个域(直播/电商/视频/广告)的历史行为,预测用户接下来会点击的视频。

**输入**:System("推荐系统助手,擅长根据用户属性与多域历史行为预测用户的视频偏好")+ User(分域列出的历史行为,每条是带 begin 前缀的三段 sid,末尾一句"请推断用户接下来会点击的视频")。这里有一个和别的任务不同的关键点——**输入序列里 domain 的排列顺序是有讲究的**:目标域放最后、视频域放倒数第二、广告域放倒数第三。因为目标是预测视频,所以视频域历史被刻意放在离输出最近的位置(倒数第二),让模型解码时能更强地条件在视频兴趣上;而不同目标域会有不同的历史排列模板(目标=电商时历史=直播/广告/视频/电商,等等)。

懂推荐的评分是"两路各生成 32 个候选,去重合并成 64 池,再做命中判定",这是它区别于懂物料(单纯 Pass@64)的核心设计。

**第一路:Thinking mode**。assistant 先 rollout 一条 CoT(`<think>分析用户兴趣、意图与上下文…推理过程…</think>`),然后在 think 块之后生成 **32 个 item id 候选**(每个候选是一串 sid,经映射表转成 item id)。
**第二路:Non-thinking mode**。assistant 的 think 块为空(`<think></think>`),直接生成 **32 个 item id 候选**。
**合并**:两路的结果**去重合并成一个 64 item ids 的候选池**。
**判定**:**只要这 64 个候选里有任意一个命中任意一个 ground truth item id,就算通过(记 1);全不中记 0。**

### 懂世界
培养做题区，查看世界知识

##  Tips
懂推荐：根据用户历史行为序列，深度理解并预测用户的真实需求。
不同的CoT Pattern会对模型懂推荐的能力产生较大的影响，目前 Baseline 懂推荐数据给出的 CoT Pattern整体上
是按照“兴趣归纳->行为模式->预测总结”三段式总结的，CoT Pattern存在较大优化的空间。可以参考 OneReason
技术报告和沙龙分享，也更加鼓励选手自己探索更有效的 CoT Pattern，以及更合理的 CoT 和 UnCoT 的配比。
沙龙：[新浪财经](https://finance.sina.com.cn/wm/2026-06-10/doc-iniaxvfp1173962.shtml)
获取可以去查看官方论文。


## 问题

1. 数据配比，数据质量，数据类型，多样性
2. RL