## using_smolvla_example
这个文件主要是基于真机连接的smolvla运行，用最简单的实例进行编写代码，总体而言，要实现一个最简单的真机VLA，只需要以下几个变量：
- 进行运算的元件
```
device = torch.device("mps") # or "cuda" or "cpu"
```
- 模型id，权重和预处理器/后处理器，预处理器和后处理器，负责将机器人的原始观测数据（如图像）转换为模型所需的格式，并将模型的输出转换为机器人可执行的动作
```
model_id = "lerobot/smolvla_base"
model = SmolVLAPolicy.from_pretrained(model_id) # 加预训练好的模型  

preprocess, postprocess = make_pre_post_processors(
	model.config,
	model_id,
	# This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
	preprocessor_overrides={"device_processor": {"device": str(device)}},

) 
```
- 真机的端口和id，端口告诉代码需要知道去哪里发送指令给电机，这直接指向了物理硬件接口。 ID 则会让代码去加载对应的**校准文件**（Calibration file），以确保模型输出的角度能正确映射到这台真机的物理姿态上。
```
# find ports using lerobot-find-port
follower_port = ... # something like "/dev/tty.usbmodem58760431631"

# the robot ids are used the load the right calibration files
follower_id = ... # something like "follower_so100" 
```
- 相机设置及端口id组成机械臂的config，即可创建一个机械臂实例并进行连接
```
camera_config = {
	"camera1": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
	"camera2": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
}

robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
robot = SO100Follower(robot_cfg)
robot.connect()
```
- robot type：因为训练数据可能有多种机器人，需要robot type告诉模型现在运行的是哪个机器人，以输出不同的自由度数据
- task：任务内容
- 特征转化：将**物理机器人的硬件特征**（Hardware Features，比如某个电机的实时读数是浮点数，或者某个摄像头的图像是640x480）**转换为模型和数据集所期望的标准格式**（Dataset Features）。
```
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}
```
比如说可以把包括电机当前位置（`float`）和摄像头画面尺寸（`tuple`，如 (480, 640, 3)）的输入转为字典
```
{
	"observation.state": {
		"dtype": "float32",
		"shape": (6,), # 当前机器人的姿态向量
		"names": ["shoulder_pan.pos", ...]
	},
	"observation.images.camera1": {
		"dtype": "video",
		"shape": (480, 640, 3),
		"names": ["height", "width", "channels"]
	},
	# ... 其他相机
}
```
最后把两个字典合并，供后续的函数进行操作如归一化操作及构建Pytorch Tensor传入模型

最后可以直接运行循环：
```
for _ in range(MAX_EPISODES):
	for _ in range(MAX_STEPS_PER_EPISODE):
		obs = robot.get_observation()
		obs_frame = build_inference_frame(
		observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
	)

	obs = preprocess(obs_frame)

	action = model.select_action(obs)
	action = postprocess(action)
	action = make_robot_action(action, dataset_features)
	robot.send_action(action)

print("Episode finished! Starting new episode...")
```

以上是对于一个真机的处理，对于模拟中的例子，则需要额外的操作。
首先真机中对于真机的连接，因为模拟没有端口和id，因此不需要这些操作，可以直接使用Robosuite
(Robosuite是Gymnasium的抽象层)创建环境，支持返回`obs, reward, done, truncated, info`这几个变量的返回，余下的基本上是数据上的操作。
比如需要Robosuite的观察数据格式转为Lerobot和把Lerobot动作输出转为Robosuite支持的动作格式进行交互。
```Python
# ===== 从模拟环境构造观测 =====
def env_obs_to_lerobot_obs(env_obs):
    """
    将 robosuite 观测转换为 LeRobot 期望的格式
    
    真机代码: obs = robot.get_observation()
    模拟替代: obs = env_obs_to_lerobot_obs(env_obs)
    """
    # 图像：robosuite 输出是上下翻转的，需要 [::-1]
    # 使用 .copy() 确保数组内存连续，避免 torch.from_numpy 报错
    agentview = env_obs["agentview_image"][::-1].copy().astype(np.uint8)
    robot0_eye_in_hand = env_obs["robot0_eye_in_hand_image"][::-1].copy().astype(np.uint8)
    
    # 状态：拼接末端位姿和夹爪
    state = np.concatenate([
        env_obs["robot0_eef_pos"],       # (3,)
        env_obs["robot0_eef_quat"],      # (4,)
        env_obs["robot0_gripper_qpos"],  # (2,)
    ]).astype(np.float32)
    
    return {
        "camera1": agentview,
        "camera2": robot0_eye_in_hand,
        "state": state,
    }


# ===== 将模型输出转换为环境动作 =====
def lerobot_action_to_env_action(action_dict):
    """
    将 LeRobot 输出转换为 robosuite 动作
    
    真机代码: robot.send_action(action)
    模拟替代: env.step(lerobot_action_to_env_action(action))
    """
    # action 可能是 dict 或 tensor，取决于 postprocess
    if isinstance(action_dict, dict):
        if "action" in action_dict:
            action = action_dict["action"]
        else:
            raise KeyError(f"Expected 'action' key in action_dict, got keys: {list(action_dict.keys())}")
    else:
        action = action_dict
    
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    
    # 确保是 (7,) 形状，clip 到合理范围
    action = np.array(action).flatten()
    if len(action) < 7:
        # Pad with zeros if action is smaller than 7 (e.g. 6D from SO-100 model)
        action = np.pad(action, (0, 7 - len(action)), mode="constant")
    action = action[:7]
    action = np.clip(action, -1, 1)
    
    return action
```
以及是
```Python
# 这告诉 LeRobot 你的观测和动作字段叫什么
dataset_features = {
    # 观测
    "observation.images.camera1": {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera2": {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),  # eef_pos(3) + eef_quat(4) + gripper(2)
        "names": ["state"],
    },
    # 动作
    "action": {
        "dtype": "float32",
        "shape": (7,),  # OSC_POSE: dx,dy,dz,dax,day,daz,gripper
        "names": ["action"],
    },
}
```
剩余的基本是一样的。

---

接下来解析 `lerobot/policies/smolvla` 目录下三个核心文件的结构与功能：`configuration_smolvla.py`、`modeling_smolvla.py` 和 `processor_smolvla.py`。可参照[[Lerobot Policy抽象基类]]来阅读

SmolVLA 是一个基于视觉语言模型 (VLM) 的策略，并结合了流匹配 (Flow Matching) 技术来生成动作。

### a. `configuration_smolvla.py`

此文件定义了 SmolVLA 策略的配置类 `SmolVLAConfig`，用于管理超参数和模型设置。

#### 主要类：`SmolVLAConfig`
继承自 `PreTrainedConfig`，用于确保持久化和参数验证。

##### 1. 动作分块 (Action Chunking)

```python
chunk_size: int = 50          # 模型一次预测的动作序列长度
n_action_steps: int = 50      # 实际执行多少步（≤ chunk_size）
```

模型每次推理输出 50 步动作，然后逐步取出执行。`n_action_steps ≤ chunk_size` 在 `__post_init__` 里做了校验。如果 `n_action_steps < chunk_size`，意味着还没执行完就重新预测，实现滑动窗口式的控制。

#####  2. 状态/动作 Padding

```python
max_state_dim: int = 32
max_action_dim: int = 32
```

不同机器人的状态和动作维度不一样（比如 Aloha 双臂 14 维，Franka 7 维）。SmolVLA 作为一个**通用模型**，需要统一输入维度，所以短的向量会被 **zero-pad** 到固定长度。这样同一个模型可以处理不同机器人，是 Physical Intelligence 的 π0 的设计思路。

#####  3. 图像预处理

```python
resize_imgs_with_padding: tuple[int, int] = (512, 512)  # 统一 resize 到 512x512
empty_cameras: int = 0  # 填充空白相机
```

`empty_cameras` 比较有意思——比如训练 Aloha 仿真时只有 top camera，但模型结构期望 3 个相机输入（top + left wrist + right wrist），就用空白图像填充。`validate_features` 里就是干这个的：

```python
def validate_features(self):
    for i in range(self.empty_cameras):
        self.input_features[f"observation.images.empty_camera_{i}"] = PolicyFeature(...)
```

以及有归一化的函数，用于统一数据处理的函数：

```Python
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
```

可以看到视觉不用归一化，因为用VLM自带的处理方式，剩余的状态和动作都要归一化。

#####  4. Aloha 适配

```python
adapt_to_pi_aloha: bool = False
use_delta_joint_actions_aloha: bool = False
```

SmolVLA 的基础模型继承自 π0 的权重，而 π0 训练时用的是 Physical Intelligence 内部的 Aloha 关节空间表示。如果你的 Aloha 数据用的是标准 LeRobot 空间，就需要 `adapt_to_pi_aloha=True` 做坐标转换。`use_delta_joint_actions_aloha` 是把关节角转成相对值（delta），但代码里标了 `NotImplementedError`，还没移植。

#####  5. VLM Backbone 相关

```python
vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
load_vlm_weights: bool = False
tokenizer_max_length: int = 48
add_image_special_tokens: bool = False
pad_language_to: str = "longest"
```

- `vlm_model_name`：用哪个预训练 VLM 作为 backbone（默认是 SmolVLM2 500M）
- `load_vlm_weights`：是否从头加载 VLM 权重。从预训练 SmolVLA checkpoint 恢复时设 `True`（因为权重已经包含了 VLM 部分）；如果只加载 SmolVLA 的 action expert 权重，VLM 权重需要从 HF 单独加载
- `tokenizer_max_length`：语言 token 最大长度，48 对机器人指令（如 "pick up the red block"）足够了
- `pad_language_to`：语言 padding 策略，`"longest"` 表示 pad 到 batch 内最长

#####  6. Action Expert 架构

这是 SmolVLA 最核心的设计——在 VLM 旁边挂一个 **action expert 网络**：

```python
attention_mode: str = "cross_attn"        # expert 通过交叉注意力读取 VLM 特征
num_expert_layers: int = -1               # expert 层数（-1 = 和 VLM 一样多）
num_vlm_layers: int = 16                  # 只用 VLM 前 16 层（不是全部）
self_attn_every_n_layers: int = 2         # 每 2 层插一次 self-attention
expert_width_multiplier: float = 0.75     # expert hidden size = VLM 的 75%
```

架构示意：
```
VLM (16层)          Action Expert
  Layer 1    ──cross_attn──→   Expert Layer 1 (cross-attn)
  Layer 2                      Expert Layer 2 (self-attn)   ← 每2层一次
  Layer 3    ──cross_attn──→   Expert Layer 3 (cross-attn)
  ...                          ...
```

expert 比 VLM 窄（0.75x），通过交叉注意力从 VLM 的中间表示中提取信息来预测动作。这样 expert 是轻量的，VLM 可以冻结。

#####  7. Flow Matching 解码


```python
num_steps: int = 10           # 去噪步数
min_period: float = 4e-3      # timestep 正弦编码的频率范围
max_period: float = 4.0
```

SmolVLA 用 **flow matching** 生成动作（和 π0 一样）。推理时从噪声开始，跑 10 步 ODE 求解得到动作序列。`min/max_period` 控制时间步嵌入的正弦编码频率范围。

#####  8. 微调策略

```python
freeze_vision_encoder: bool = True    # 冻结视觉编码器
train_expert_only: bool = True        # 只训练 action expert
train_state_proj: bool = True         # 也训练 state projection 层
```

默认的微调策略是**冻结整个 VLM，只训练 action expert + state projection**。这是非常高效的方式——VLM 已经有很好的视觉语言理解能力，只需要训练动作生成部分。

##### 9. 优化器预设

```python
optimizer_lr: float = 1e-4
optimizer_betas: tuple = (0.9, 0.95)      # β2=0.95 比默认 0.999 更激进
optimizer_weight_decay: float = 1e-10     # 几乎不用 weight decay
scheduler_warmup_steps: int = 1_000
scheduler_decay_steps: int = 30_000
```

注意 `weight_decay=1e-10` 几乎为零，加上 cosine decay 到 `2.5e-6` 的极小 lr，说明微调时很保守，避免破坏预训练特征。


### b. `modeling_smolvla.py`

此文件包含了策略的核心逻辑，包括模型架构、前向传播和推理算法。

#### 主要类：`SmolVLAPolicy`
继承自 `PreTrainedPolicy`，是 LeRobot 框架与模型实现的接口。

*   **`__init__`**: 初始化底层的 `VLAFlowMatching` 模型。
*   **`forward`**: 训练时调用。
    1.  预处理图像 (`prepare_images`)、状态和动作。
    2.  调用 `self.model.forward` 计算损失。
    3.  返回损失字典（包含 MSE Loss）。
*   **`select_action`**: 推理时调用。
    1.  管理动作队列 (`_queues`)。如果队列为空，调用 `_get_action_chunk` 预测新的动作块。
    2.  返回队列中的下一个动作。
*   **`predict_action_chunk`**: 执行一次完整的动作块预测。
*   **`prepare_images`**:
    *   将图像调整大小并填充到配置的尺寸。
    *   将像素值从 `[0, 1]` 归一化到 `[-1, 1]` (SigLIP 要求)。
    *   处理缺失的相机视角，填充全黑图像。

#### 主要类：`VLAFlowMatching`
继承自 `nn.Module`，实现了结合 VLM 和流匹配的具体算法。

*   **组件**:
    *   `vlm_with_expert`: 包含 VLM 骨干网和动作专家层。
    *   `state_proj`, `action_in_proj`, `action_out_proj`: 线性投影层，用于将状态和动作映射到 VLM 的隐空间维度。
*   **`forward` (训练)**:
    1.  **Embed Prefix**: 将图像、语言指令和机器人状态编码为 Embeddings（前缀）。
    2.  **Embed Suffix**: 将带噪声的动作 (Noisy Actions) 和时间步 (Timestep) 编码为 Embeddings（后缀）。
    3.  **VLM Forward**: 将前缀和后缀拼接，传入 VLM。
    4.  **Loss**: 提取后缀部分的输出，通过投影层得到预测的速度场 $v_t$，与目标 $u_t$ (噪声 - 动作) 计算 MSE 损失。
*   **`sample_actions` (推理)**:
    1.  从高斯噪声开始。
    2.  使用欧拉法 (Euler step) 或其他求解器，循环执行去噪步骤 (`denoise_step`)。
    3.  在每一步中，VLM 根据当前的噪声动作预测更新方向，逐步还原出真实动作。
*   **`embed_prefix` / `embed_suffix`**:
    *   负责构建 Attention Mask，确保因果关系（例如图像不应关注到动作）。
    *   使用正弦位置编码 (`create_sinusoidal_pos_embedding`) 对时间步进行编码。

3. `processor_smolvla.py`

此文件定义了数据处理流水线，负责将原始数据集转换为模型可输入的格式。

#### 主要函数：`make_smolvla_pre_post_processors`
构建并返回预处理 (`input_steps`) 和后处理 (`output_steps`) 流水线。

1.  **预处理流水线**:
    *   `AddBatchDimensionProcessorStep`: 增加 Batch 维度。
    *   `SmolVLANewLineProcessor`: **关键步骤**，确保语言指令 (`task`) 以换行符 `\n` 结尾（这对某些 VLM 如 PaliGemma/SmolVLM 是必须的）。
    *   `TokenizerProcessorStep`: 使用 VLM 对应的 Tokenizer 对文本进行编码。
    *   `DeviceProcessorStep`: 将数据移动到指定设备 (GPU/CPU)。
    *   `NormalizerProcessorStep`: 根据数据集统计信息对状态和动作进行归一化。

2.  **后处理流水线**:
    *   `UnnormalizerProcessorStep`: 将模型输出的归一化动作还原为原始物理单位。
    *   `DeviceProcessorStep`: 移回 CPU。

#### 辅助类：`SmolVLANewLineProcessor`
*   检查数据中的 `task` 字段。
*   如果是字符串或字符串列表，确保其以 `\n` 结尾。这是为了适配预训练模型的 Prompt 格式。


## 数据类型
```
[Input Data Creation]
  observation.images.cam1: torch.Size([1, 3, 224, 224]) (Random Noise)
  observation.images.cam2: torch.Size([1, 3, 224, 224]) (Random Noise)
  observation.images.cam3: torch.Size([1, 3, 224, 224]) (Random Noise)
  observation.state: torch.Size([1, 6]) (Random State)
  Prompt: 'Pick up the red apple.'
  Tokenizing prompt...
  observation.language.tokens: torch.Size([1, 6])
  observation.language.attention_mask: torch.Size([1, 6]) (casted to bool)
  [Debug] lm_expert type: <class 'transformers.models.llama.modeling_llama.LlamaModel'>
  [Debug] lm_expert has 16 layers.

[Running Inference: select_action]

=== Inspection Results ===

[Intermediate Layers]
  Layer: state_embedding_proj
    shape: (1, 960)
    mean: -0.009457314386963844
    std: 0.23588013648986816
    dtype: torch.float32
  Layer: expert_layer_0_norm
    shape: (1, 10, 720)
    mean: 0.02460503950715065
    std: 0.9994508624076843
    dtype: torch.float32
  Layer: action_output_proj
    shape: (1, 10, 32)
    mean: 0.011173387058079243
    std: 0.6361964344978333
    dtype: torch.float32

[Output Data]
  Action Shape: torch.Size([1, 6])
  Action Values (First 5): [0.1507483273744583, 0.20595519244670868, 0.3171220123767853, -2.04714298248291, 0.7914081811904907]

=== Done ===
```
其中obs在真机中来自`robot.get_observation()`返回一个`dic[str, num]`，实际上返回了一个关于几号camera，关节的输入。
```
(Pdb) pp obs
{'camera1': array([[[116, 114, 111],
        [115, 114, 110],
        [115, 113, 110],
        ...,
        [116, 113, 110],
        [116, 114, 110],
        [116, 115, 111]],

       [[117, 115, 111],
        [116, 114, 111],
        [116, 114, 110],
        ...,
        [115, 113, 110],
        [116, 114, 111],
        [117, 115, 112]],

       [[117, 115, 111],
        [116, 115, 111],
        [116, 114, 111],
        ...,
        [116, 114, 111],
        [117, 115, 112],
        [117, 116, 112]],

       ...,

       [[241, 241, 234],
        [248, 245, 241],
        [248, 247, 243],
        ...,
        [251, 250, 246],
        [252, 250, 246],
        [253, 252, 248]],

       [[247, 245, 240],
        [250, 247, 244],
        [247, 246, 242],
        ...,
        [252, 250, 247],
        [252, 251, 246],
        [253, 251, 247]],

       [[250, 247, 243],
        [248, 246, 243],
        [245, 244, 241],
        ...,
        [252, 250, 247],
        [252, 251, 247],
        [252, 251, 247]]], shape=(480, 640, 3), dtype=uint8),
 'camera2': array([[[252, 252, 247],
        [251, 251, 248],
        [253, 252, 248],
        ...,
        [238, 239, 234],
        [240, 239, 235],
        [241, 241, 235]],

       [[251, 250, 246],
        [251, 250, 246],
        [252, 250, 246],
        ...,
        [241, 241, 235],
        [241, 240, 234],
        [242, 241, 235]],

       [[252, 250, 247],
        [252, 251, 247],
        [251, 251, 247],
        ...,
        [239, 239, 233],
        [239, 239, 233],
        [241, 242, 236]],

       ...,

       [[ 82,  82,  82],
        [ 82,  82,  82],
        [ 82,  82,  82],
        ...,
        [ 81,  81,  81],
        [ 81,  81,  81],
        [ 81,  81,  81]],

       [[ 82,  82,  82],
        [ 82,  82,  82],
        [ 82,  82,  82],
        ...,
        [ 82,  82,  82],
        [ 82,  82,  82],
        [ 82,  82,  82]],

       [[ 82,  82,  82],
        [ 82,  82,  82],
        [ 82,  82,  82],
        ...,
        [ 82,  82,  82],
        [ 82,  82,  82],
        [ 82,  82,  82]]], shape=(480, 640, 3), dtype=uint8),
 'state': array([-0.0991365 , -0.01267833,  1.0168837 ,  0.997682  , -0.02249813,
        0.06418103,  0.00229299,  0.020833  , -0.020833  ], dtype=float32)}
```
`lerobot.policies.utils` 中`build_interface_frame()`则将观察，任务，设备，计算设备，数据维度展位一个字典：
```
{'observation.images.camera1': tensor([[[[0.4549, 0.4510, 0.4510,  ..., 0.4549, 0.4549, 0.4549],
          [0.4588, 0.4549, 0.4549,  ..., 0.4510, 0.4549, 0.4588],
          [0.4588, 0.4549, 0.4549,  ..., 0.4549, 0.4588, 0.4588],
          ...,
          [0.9451, 0.9725, 0.9725,  ..., 0.9843, 0.9882, 0.9922],
          [0.9686, 0.9804, 0.9686,  ..., 0.9882, 0.9882, 0.9922],
          [0.9804, 0.9725, 0.9608,  ..., 0.9882, 0.9882, 0.9882]],

         [[0.4471, 0.4471, 0.4431,  ..., 0.4431, 0.4471, 0.4510],
          [0.4510, 0.4471, 0.4471,  ..., 0.4431, 0.4471, 0.4510],
          [0.4510, 0.4510, 0.4471,  ..., 0.4471, 0.4510, 0.4549],
          ...,
          [0.9451, 0.9608, 0.9686,  ..., 0.9804, 0.9804, 0.9882],
          [0.9608, 0.9686, 0.9647,  ..., 0.9804, 0.9843, 0.9843],
          [0.9686, 0.9647, 0.9569,  ..., 0.9804, 0.9843, 0.9843]],

         [[0.4353, 0.4314, 0.4314,  ..., 0.4314, 0.4314, 0.4353],
          [0.4353, 0.4353, 0.4314,  ..., 0.4314, 0.4353, 0.4392],
          [0.4353, 0.4353, 0.4353,  ..., 0.4353, 0.4392, 0.4392],
          ...,
          [0.9176, 0.9451, 0.9529,  ..., 0.9647, 0.9647, 0.9725],
          [0.9412, 0.9569, 0.9490,  ..., 0.9686, 0.9647, 0.9686],
          [0.9529, 0.9529, 0.9451,  ..., 0.9686, 0.9686, 0.9686]]]],
       device='cuda:0'),
 'observation.images.camera2': tensor([[[[0.9882, 0.9843, 0.9922,  ..., 0.9333, 0.9412, 0.9451],
          [0.9843, 0.9843, 0.9882,  ..., 0.9451, 0.9451, 0.9490],
          [0.9882, 0.9882, 0.9843,  ..., 0.9373, 0.9373, 0.9451],
          ...,
          [0.3216, 0.3216, 0.3216,  ..., 0.3176, 0.3176, 0.3176],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216]],

         [[0.9882, 0.9843, 0.9882,  ..., 0.9373, 0.9373, 0.9451],
          [0.9804, 0.9804, 0.9804,  ..., 0.9451, 0.9412, 0.9451],
          [0.9804, 0.9843, 0.9843,  ..., 0.9373, 0.9373, 0.9490],
          ...,
          [0.3216, 0.3216, 0.3216,  ..., 0.3176, 0.3176, 0.3176],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216]],

         [[0.9686, 0.9725, 0.9725,  ..., 0.9176, 0.9216, 0.9216],
          [0.9647, 0.9647, 0.9647,  ..., 0.9216, 0.9176, 0.9216],
          [0.9686, 0.9686, 0.9686,  ..., 0.9137, 0.9137, 0.9255],
          ...,
          [0.3216, 0.3216, 0.3216,  ..., 0.3176, 0.3176, 0.3176],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216]]]],
       device='cuda:0'),
 'observation.state': tensor([[[-0.0991, -0.0127,  1.0169,  0.9977, -0.0225,  0.0642,  0.0023,
           0.0208, -0.0208]]], device='cuda:0'),
 'robot_type': 'panda',
 'task': 'pick up the cube'}
 
 ---
 
 dataset_features = {
    # 观测
    "observation.images.camera1": {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera2": {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),  # eef_pos(3) + eef_quat(4) + gripper(2)
        "names": ["state"],
    },
    # 动作
    "action": {
        "dtype": "float32",
        "shape": (7,),  # OSC_POSE: dx,dy,dz,dax,day,daz,gripper
        "names": ["action"],
    },
}
```
经过预处理后则加入了关于环境的信息和mask信息，并将task经过分词器转为词id：
```
(Pdb) pp obs_processed
{'action': None,
 'info': {},
 'next.done': False,
 'next.reward': 0.0,
 'next.truncated': False,
 'observation.images.camera1': tensor([[[[0.4549, 0.4510, 0.4510,  ..., 0.4549, 0.4549, 0.4549],
          [0.4588, 0.4549, 0.4549,  ..., 0.4510, 0.4549, 0.4588],
          [0.4588, 0.4549, 0.4549,  ..., 0.4549, 0.4588, 0.4588],
          ...,
          [0.9451, 0.9725, 0.9725,  ..., 0.9843, 0.9882, 0.9922],
          [0.9686, 0.9804, 0.9686,  ..., 0.9882, 0.9882, 0.9922],
          [0.9804, 0.9725, 0.9608,  ..., 0.9882, 0.9882, 0.9882]],

         [[0.4471, 0.4471, 0.4431,  ..., 0.4431, 0.4471, 0.4510],
          [0.4510, 0.4471, 0.4471,  ..., 0.4431, 0.4471, 0.4510],
          [0.4510, 0.4510, 0.4471,  ..., 0.4471, 0.4510, 0.4549],
          ...,
          [0.9451, 0.9608, 0.9686,  ..., 0.9804, 0.9804, 0.9882],
          [0.9608, 0.9686, 0.9647,  ..., 0.9804, 0.9843, 0.9843],
          [0.9686, 0.9647, 0.9569,  ..., 0.9804, 0.9843, 0.9843]],

         [[0.4353, 0.4314, 0.4314,  ..., 0.4314, 0.4314, 0.4353],
          [0.4353, 0.4353, 0.4314,  ..., 0.4314, 0.4353, 0.4392],
          [0.4353, 0.4353, 0.4353,  ..., 0.4353, 0.4392, 0.4392],
          ...,
          [0.9176, 0.9451, 0.9529,  ..., 0.9647, 0.9647, 0.9725],
          [0.9412, 0.9569, 0.9490,  ..., 0.9686, 0.9647, 0.9686],
          [0.9529, 0.9529, 0.9451,  ..., 0.9686, 0.9686, 0.9686]]]],
       device='cuda:0'),
 'observation.images.camera2': tensor([[[[0.9882, 0.9843, 0.9922,  ..., 0.9333, 0.9412, 0.9451],
          [0.9843, 0.9843, 0.9882,  ..., 0.9451, 0.9451, 0.9490],
          [0.9882, 0.9882, 0.9843,  ..., 0.9373, 0.9373, 0.9451],
          ...,
          [0.3216, 0.3216, 0.3216,  ..., 0.3176, 0.3176, 0.3176],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216]],

         [[0.9882, 0.9843, 0.9882,  ..., 0.9373, 0.9373, 0.9451],
          [0.9804, 0.9804, 0.9804,  ..., 0.9451, 0.9412, 0.9451],
          [0.9804, 0.9843, 0.9843,  ..., 0.9373, 0.9373, 0.9490],
          ...,
          [0.3216, 0.3216, 0.3216,  ..., 0.3176, 0.3176, 0.3176],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216]],

         [[0.9686, 0.9725, 0.9725,  ..., 0.9176, 0.9216, 0.9216],
          [0.9647, 0.9647, 0.9647,  ..., 0.9216, 0.9176, 0.9216],
          [0.9686, 0.9686, 0.9686,  ..., 0.9137, 0.9137, 0.9255],
          ...,
          [0.3216, 0.3216, 0.3216,  ..., 0.3176, 0.3176, 0.3176],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216],
          [0.3216, 0.3216, 0.3216,  ..., 0.3216, 0.3216, 0.3216]]]],
       device='cuda:0'),
 'observation.language.attention_mask': tensor([[ True,  True,  True,  True,  True, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False]],
       device='cuda:0'),
 'observation.language.tokens': tensor([[18188,   614,   260, 20636,   198,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2]],
       device='cuda:0'),
 'observation.state': tensor([[[-0.0991, -0.0127,  1.0169,  0.9977, -0.0225,  0.0642,  0.0023,
           0.0208, -0.0208]]], device='cuda:0'),
 'task': ['pick up the cube\n']}
```
经过`select_action`后就变成一个动作向量了：
```
tensor([[0.3547, 0.0511, 0.9123, 0.9800, 0.1543, 0.3194]], device='cuda:0')
```
经过后处理就变成纯向量了：
```
tensor([[0.3547, 0.0511, 0.9123, 0.9800, 0.1543, 0.3194]])
```
