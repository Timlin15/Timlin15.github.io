# Policy 抽象层分析：极简实现指南

要在 `lerobot` 中实现一个新的策略 (Policy)，你需要定义两个主要类：一个 **配置 (Configuration)** 类和一个 **策略 (Policy)** 类。这两个类分别对应训练和运行策略所需的数据结构和逻辑。

基类提供了抽象层，它们位于：
- `lerobot.configs.policies.PreTrainedConfig`: 基础配置类。
- `lerobot.policies.pretrained.PreTrainedPolicy`: 基础策略类。

以下是每个类极简实现的详细要求分解。

## 1. 配置类 (`MyPolicyConfig`)

这个类定义了你的策略所需的超参数和配置。它必须继承自 `PreTrainedConfig`。LeRobot 使用 `draccus` 进行配置管理，这使得可以从命令行参数进行类型安全的解析。

### 位置
通常放置在 `lerobot/policies/<policy_name>/configuration_<policy_name>.py`。

### 必要的继承
```python
from lerobot.configs.policies import PreTrainedConfig
from dataclasses import dataclass

@dataclass
class MyPolicyConfig(PreTrainedConfig):
    ...
```

### 必须实现的抽象方法与属性
你必须实现以下方法以满足 `PreTrainedConfig` 抽象类的要求：

1.  **`observation_delta_indices`** (属性):
    -   返回观测特征中代表增量 (delta) 而非绝对值的索引列表（通常用于归一化目的）。如果不适用，则返回 `None`。
    -   签名: `@property def observation_delta_indices(self) -> list | None:`

2.  **`action_delta_indices`** (属性):
    -   返回动作空间中是增量的索引或键的列表。如果不适用，则返回 `None`。
    -   签名: `@property def action_delta_indices(self) -> list | None:`

3.  **`reward_delta_indices`** (属性):
    -   同上，但是针对奖励 (reward)。如果不适用，则返回 `None`。
    -   签名: `@property def reward_delta_indices(self) -> list | None:`

4.  **`validate_features`**:
    -   用于验证配置中的输入/输出特征是否符合策略预期的方法。如果验证失败，应抛出错误。
    -   签名: `def validate_features(self) -> None:`

5.  **`get_optimizer_preset`**:
    -   返回默认的优化器配置（例如，带有特定学习率的 AdamW）。
    -   签名: `def get_optimizer_preset(self) -> OptimizerConfig:`

6.  **`get_scheduler_preset`**:
    -   返回默认的学习率调度器配置。如果默认不使用调度器，则返回 `None`。
    -   签名: `def get_scheduler_preset(self) -> LRSchedulerConfig | None:`

### 隐式要求
-   你的类名应该通过继承 `PreTrainedConfig` 自动注册到 `draccus`。`type` 属性会根据类名或注册键自动处理，但你应该确保命名唯一。

---

## 2. 策略类 (`MyPolicy`)

这个类包含模型架构以及前向传播、损失计算和动作推理的逻辑。它必须继承自 `PreTrainedPolicy`。

### 位置
通常放置在 `lerobot/policies/<policy_name>/modeling_<policy_name>.py`。

### 必要的继承
```python
from lerobot.policies.pretrained import PreTrainedPolicy
from torch import Tensor, nn

class MyPolicy(PreTrainedPolicy):
    ...
```

### 必要的类属性
1.  **`config_class`**:
    -   必须指向你的配置类（例如 `MyPolicyConfig`）。
    -   `config_class = MyPolicyConfig`
2.  **`name`**:
    -   你的策略的字符串标识符（例如 "my_policy"）。
    -   `name = "my_policy"`

### 必须实现的抽象方法
你必须实现以下方法以满足 `PreTrainedPolicy` 抽象类的要求：

1.  **`__init__`**:
    -   使用提供的 `config` 初始化模型组件（backbones, heads 等）。
    -   调用 `super().__init__(config)`。

2.  **`reset`**:
    -   当环境重置时调用，用于重置内部状态（例如，对于循环策略或动作分块队列）。
    -   签名: `def reset(self):`

3.  **`get_optim_params`**:
    -   返回一个传递给优化器的参数字典或列表。这允许设置不同的学习率（例如，backbone 和 head 使用不同的学习率）。
    -   签名: `def get_optim_params(self) -> dict:`

4.  **`forward`**:
    -   主要的训练步骤。接收一批数据，计算模型输出，并计算损失。
    -   **输入**: `batch` (Tensor 字典)。
    -   **返回**: 一个元组 `(loss, output_dict)`。`loss` 必须是用于反向传播的标量 Tensor。`output_dict` 包含用于记录日志的指标。
    -   签名: `def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:`

5.  **`predict_action_chunk`**:
    -   给定一批观测值，预测一系列（分块/chunk）动作。理想情况下在 `eval` 模式下运行。
    -   **输入**: `batch` (Tensor 字典)。
    -   **返回**: 形状为 `(batch_size, chunk_size, action_dim)` 的 Tensor。
    -   签名: `def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:`

6.  **`select_action`**:
    -   返回一个要在环境中执行的单一动作。此方法通常处理诸如 **时间集成 (temporal ensembling)** 或 **动作队列 (action queueing)** 的逻辑（如果模型预测分块）。
    -   **输入**: `batch` (Tensor 字典)。
    -   **返回**: 代表单一动作的 Tensor。
    -   签名: `def select_action(self, batch: dict[str, Tensor]) -> Tensor:`

---

## 代码骨架

```python
class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        # Create base kwargs
        kwargs = {"strict": strict}

        # Add device parameter for newer versions that support it
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location

        # Load the model with appropriate kwargs
        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
        log_model_loading_keys(missing_keys, unexpected_keys)

        # For older versions, manually move to device if needed
        if "device" not in kwargs and map_location != "cpu":
            logging.warning(
                "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                " This means that the model is loaded on 'cpu' first and then copied to the device."
                " This leads to a slower loading time."
                " Please update safetensors to version 0.4.3 or above for improved performance."
            )
            model.to(map_location)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Returns the action chunk (for action chunking policies) for a given observation, potentially in batch mode.

        Child classes using action chunking should use this method within `select_action` to form the action chunk
        cached for selection.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError

    def push_model_to_hub(
        self,
        cfg: TrainPipelineConfig,
    ):
        api = HfApi()
        repo_id = api.create_repo(
            repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
        ).repo_id

        # Push the files to the repo in a single commit
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id

            self.save_pretrained(saved_path)  # Calls _save_pretrained and stores model tensors

            card = self.generate_model_card(
                cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
            )
            card.save(str(saved_path / "README.md"))

            cfg.save_pretrained(saved_path)  # Calls _save_pretrained and stores train config

            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload policy weights, train config and readme",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=["*.tmp", "*.log"],
            )

            logging.info(f"Model pushed to {commit_info.repo_url.url}")

    def generate_model_card(
        self, dataset_repo_id: str, model_type: str, license: str | None, tags: list[str] | None
    ) -> ModelCard:
        base_model = "lerobot/smolvla_base" if model_type == "smolvla" else None  # Set a base model

        card_data = ModelCardData(
            license=license or "apache-2.0",
            library_name="lerobot",
            pipeline_tag="robotics",
            tags=list(set(tags or []).union({"robotics", "lerobot", model_type})),
            model_name=model_type,
            datasets=dataset_repo_id,
            base_model=base_model,
        )

        template_card = (
            files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text(encoding="utf-8")
        )
        card = ModelCard.from_template(card_data, template_str=template_card)
        card.validate()
        return card

```

此代码来自policies文件夹中的`pretrained.py`这个文件，主要是一个抽象基类，规定了policy需要有什么基本函数。其中最重要的

| 方法                                     | 作用                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| `forward(batch)` → `(loss, info)`        | 训练时的前向传播，返回 loss                                  |
| `select_action(batch)` → `Tensor`        | **推理核心**：给定观测，返回单步动作，负责处理历史缓存和 action chunking 的调度 |
| `predict_action_chunk(batch)` → `Tensor` | 预测一个动作块（action chunk），被 `select_action` 内部调用  |
| `reset()`                                | 环境 reset 时清空缓存（如 action chunk 队列、观测历史等）    |
| `get_optim_params()`                     | 返回优化器参数组（允许不同模块用不同 lr）                    |

`select_action` 和 `predict_action_chunk` 的分离设计是关键——`predict_action_chunk` 负责模型推理出一整段动作序列，`select_action` 负责从缓存中逐步取出单步动作，实现了 **action chunking** 的统一抽象。



```Python
@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus issue
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: use a different name to avoid override
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    private: bool | None = None
    # Add tags to your policy on the hub.
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg] #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: Parse the original config to get the config subclass, so that we can
        # apply cli overrides.
        # This is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, config_file, args=[])

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            config = json.load(f)

        config.pop("type")
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)

```

这段代码来自`lerobot/configs/policies.py`中，是所有策略配置的基类，本质上是一个 **dataclass + HuggingFace Hub 集成 + 注册机制**，用来描述"一个策略需要知道的所有静态信息"。

### 核心字段

```python
n_obs_steps: int = 1                          # 观测窗口长度（用几步历史观测）
input_features: dict[str, PolicyFeature]       # 输入特征定义（图像、关节状态等）
output_features: dict[str, PolicyFeature]      # 输出特征定义（动作）
device: str | None = None                      # 运行设备
use_amp: bool = False                          # 是否用混合精度训练
```

`input_features` 和 `output_features` 是最关键的——它们定义了策略的输入输出"模态接口"，比如哪些 key 是图像、哪些是机器人状态、动作维度是多少等。

### 重要的便捷属性

这些 property 从 `input_features` / `output_features` 中按类型过滤，方便子类和训练代码快速获取：

| 属性                  | 作用                           |
| --------------------- | ------------------------------ |
| `robot_state_feature` | 提取机器人本体状态（关节角等） |
| `image_features`      | 提取所有视觉输入               |
| `action_feature`      | 提取动作输出特征               |
| `env_state_feature`   | 提取环境状态（如果有）         |

### 子类必须实现的抽象方法

python

```python
# delta indices —— 哪些维度需要做差分归一化（位置→速度）
observation_delta_indices → list | None
action_delta_indices → list | None
reward_delta_indices → list | None

# 优化器/调度器预设（不同策略用不同的lr、weight_decay等）
get_optimizer_preset() → OptimizerConfig
get_scheduler_preset() → LRSchedulerConfig | None

# 检查 features 配置是否合法
validate_features() → None
```

这意味着每个具体策略（ACT、Diffusion、VLA）都要声明自己需要什么样的优化配置和归一化方式。

几个带有delta的函数是**"差分归一化"**——指定哪些维度在归一化时用**相邻帧的差值**而不是绝对值。

举个例子，假设动作是 `[x, y, z, gripper]` 共 4 维：

- `action_delta_indices = [0, 1, 2]` 表示 x/y/z 位置维度要做差分（`a_t - a_{t-1}`），转化为类似速度的量再归一化
- `gripper`（第3维）不在列表里，因为它是开合状态（0/1），差分没意义

为什么要这样做？因为**绝对位置**的分布范围可能很大且依赖初始位置，差分后数据分布更稳定，归一化效果更好。返回 `None` 表示该策略不用差分归一化。



而优化器/调度器预设是因为不同策略的训练超参差异很大：

- **ACT**：用 AdamW，lr=1e-5，较小的 weight_decay
- **Diffusion Policy**：lr 可能 1e-4，配合 cosine scheduler
- **VLA 微调**：可能冻结 backbone，只有 action head 用较大 lr

把这些 "推荐配置" 封装在 config 里，训练循环不需要针对每种策略写不同的优化器代码，直接调用即可。



最后一个函数在构建策略前检查 `input_features` / `output_features` 的配置是否合法。比如：

- ACT 要求必须有且仅有一个 `ACTION` 类型的输出
- 某些策略要求图像输入必须是特定分辨率
- SmolVLA 可能要求必须有至少一个视觉输入

如果配置不合法就直接报错，**fail fast**，避免跑到训练中途才崩。

### `from_pretrained` —— 比较 hacky 但重要

这个方法的流程：

1. 从本地或 HF Hub 下载 `config.json`
2. 先用 draccus 解析一遍，拿到具体子类类型（比如 `ACTConfig`）
3. 再用那个子类重新解析，支持 CLI 覆盖参数

代码里自己也标了 `# HACK`，说明这块是为了解决 draccus 不支持"从文件推断子类类型"的限制而做的 workaround。

### `draccus.ChoiceRegistry` 的作用

这是注册机制的核心——所有 config 子类通过这个 registry 注册后，可以用字符串名字（如 `"act"`、`"diffusion"`）来实例化对应配置。`type` 属性就是返回这个注册名。这让 YAML/CLI 配置能通过 `policy.type: act` 自动选择正确的 config 类。

### 总结

`PreTrainedConfig` 的角色是：**策略的完整元数据描述** —— 输入输出是什么、怎么归一化、用什么优化器、跑在什么设备上。它和 `PreTrainedPolicy` 配合，实现了"config 描述策略 → policy 执行策略"的分离，同时通过注册机制让整个框架可以用字符串配置驱动。