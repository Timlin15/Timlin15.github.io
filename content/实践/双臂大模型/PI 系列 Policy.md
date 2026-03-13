> [!NOTE] 注意
> 本笔记是针对 OpenPI 实验室的 PI 系列建模进行分析，更偏重理论层面，由于PI系列是基于 VLA 的模型，所以需要 Transformers [[CS336 Assignment 1]] 基础和 Flow model [[Diffusion & Flow model]] 才能看懂。
> 至于代码实现，可以查看[[PI05代码分析]] 和 [[Lerobot Policy抽象基类]]
```mermaid
flowchart TD
    %% 根节点
    Root["PI05@lerobot理解"]

    %% 左侧分支
    Theory[模型理论]
    PI_Policy[PI系列policy]
    Diffusion["Diffusion & flow model"]
    CS336[CS336 Assignment 1]

    %% 右侧分支
    Engineering[代码工程实现]
    Abstract[lerobot抽象层设计]
    CodeAnalysis[PI05代码分析]

    %% 连接关系
    Root --> Theory
    Theory --> PI_Policy
    PI_Policy --> Diffusion
    PI_Policy --> CS336

    Root --> Engineering
    Engineering --> Abstract
    Engineering --> CodeAnalysis
```

> [!NOTE] 注意
> 本文主要聚焦PI系列策略的建模，所以不会特别提及模型的训练方式和训练集，尽管这两个方向可以说和建模同等重要，这部分可以查看[[pi0.5]]，虽然这部分因为是早期文章所以稍显稚嫩，可能考虑重构一下。

