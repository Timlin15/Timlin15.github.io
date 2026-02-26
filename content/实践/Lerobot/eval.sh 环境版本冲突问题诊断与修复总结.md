> 本文完全由Claude Opus4.5撰写
## 问题描述

运行 `./eval.sh` 时出现环境版本冲突，脚本无法正常执行。

## 根本原因分析

共发现 **3 个问题**，按出现顺序排列：

### 问题 1：安装了错误的 libero 包（主要问题）

**现象**：`ModuleNotFoundError: No module named 'libero'`

**原因**：
- 环境中以 editable 模式安装了原始的 `libero==0.1.0`（来源：`/mnt/data1/linjianqi/LIBERO`）
- 但 lerobot 项目实际需要的是 HuggingFace 维护的 `hf-libero>=0.1.3,<0.2.0`（见 `pyproject.toml` 第 165 行）
- 原始 `libero` 包的 editable install 机制已损坏：`__editable___libero_0_1_0_finder.py` 中的 `MAPPING` 字典为空，导致 Python 无法找到 `libero` 模块
- 原始 `libero` 包依赖极度过时的版本（numpy\==1.22.4, transformers\==4.21.1, gym\==0.25.2），与 lerobot 的依赖严重冲突

**解决方法**：
```bash
# 卸载错误的 libero 包
pip uninstall libero

# 安装正确的 hf-libero 包
# 注意：由于 robomimic==0.2.0 依赖 egl_probe，而 egl_probe 从源码编译有 cmake 兼容性问题，
# 但 hf-egl-probe 已提供等效模块，所以需要分步安装
pip install robosuite==1.4.0 --no-deps
pip install robomimic==0.2.0 --no-deps
pip install "hf-libero>=0.1.3,<0.2.0" --no-deps
pip install "hydra-core>=1.2,<1.4" "bddl==1.0.1" easydict thop "matplotlib>=3.5.3" "future>=0.18.2" h5py tensorboard tensorboardX
```

### 问题 2：numpy 版本过低

**现象**：`pip check` 报告 `opencv-python-headless` 和 `rerun-sdk` 需要 `numpy>=2`，但安装的是 `numpy==1.26.4`

**解决方法**：
```bash
pip install "numpy>=2,<2.3.0"
```

### 问题 3：robosuite 日志文件权限冲突

**现象**：`PermissionError: [Errno 13] Permission denied: '/tmp/robosuite.log'`

**原因**：`robosuite==1.4.0` 在 `log_utils.py` 中硬编码了日志路径 `/tmp/robosuite.log`，该文件已被其他用户 (`linmin`) 创建，当前用户 (`liufuweijia`) 无写入权限。

**解决方法**：
修改 `robosuite/utils/log_utils.py` 第 71 行，将日志路径改为用户独立的路径：
```python
# 修改前
fh = logging.FileHandler("/tmp/robosuite.log")

# 修改后
import os
fh = logging.FileHandler(os.path.join("/tmp", f"robosuite_{os.getuid()}.log"))
```

### 附加问题：hf-libero 缺少 assets 目录

**现象**：`FileNotFoundError: No such file or directory: '.../libero/libero/assets/scenes/libero_living_room_tabletop_base_style.xml'`

**原因**：`hf-libero` PyPI 包未包含场景资源文件（XML、模型等），需要从原始 LIBERO 仓库链接。

**解决方法**：
```bash
ln -s /mnt/data1/linjianqi/LIBERO/libero/libero/assets \
      /mnt/data1/linjianqi/conda/lerobot/lib/python3.10/site-packages/libero/libero/assets
```

---

## 第二轮修复

在解决上述环境版本冲突后，再次运行 `./eval.sh` 时遇到了新的问题。输出中有大量 EGL 相关的错误日志，但经分析 **EGL 错误并非真正的致命错误**。

### 关于 EGL 错误的说明

运行时输出中大量出现的 `EGLError: EGL_NOT_INITIALIZED` 错误全部发生在 Python 对象的 `__del__` 清理方法中（`MjRenderContext.__del__`、`EGLGLContext.__del__`），属于**进程退出时的无害清理噪音**。这些错误的出现是因为：
- 进程因其他原因崩溃后，Python 在退出时尝试清理 EGL/OpenGL 资源
- 此时 EGL display 已经被反初始化，导致清理调用（`eglDestroyContext`、`eglMakeCurrent`）失败
- 这些错误被标记为 `Exception ignored in:`，即 Python 自身忽略了这些异常

**这些 EGL 错误不需要修复**，它们不影响程序功能，只是视觉噪音。

### 问题 4：Feature name mismatch（策略与环境的特征键名不匹配）

**现象**：
```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb', 'observation.images.right_wrist_0_rgb']
- Extra features: ['observation.images.image', 'observation.images.image2']
```

**原因**：
- pi05 策略预训练时使用了 3 个摄像头：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`
- LIBERO 环境只提供 2 个摄像头：`image`（agentview）、`image2`（eye_in_hand）
- 键名和数量都不匹配，需要手动映射并填充缺失的空摄像头

**解决方法**：
在 `eval.sh` 中添加 `--rename_map` 和 `--policy.empty_cameras` 参数：
```bash
lerobot-eval \
	--policy.path=../pi05_base \
	--policy.n_action_steps=10 \
	--env.type=libero \
	--env.task=libero_10 \
	--eval.batch_size=1 \
	--eval.n_episodes=10 \
	--output_dir=./eval_logs/pi05_libero10 \
	--env.max_parallel_tasks=1 \
	--policy.empty_cameras=1 \
	--rename_map='{"observation.images.image": "observation.images.base_0_rgb", "observation.images.image2": "observation.images.right_wrist_0_rgb"}'
```

其中：
- `--rename_map` 将环境的 `image` 映射到策略的 `base_0_rgb`，`image2` 映射到 `right_wrist_0_rgb`
- `--policy.empty_cameras=1` 告知策略有 1 个摄像头（`left_wrist_0_rgb`）将使用空白填充

### 问题 5：HuggingFace Gated Model 访问权限不足

**现象**：
```
403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings to view this repository.
OSError: We couldn't connect to 'https://hf-mirror.com' to load the files
```
（此错误在修复 Feature mismatch 后暴露）

**原因**：
- pi05 策略需要下载 `google/paligemma-3b-pt-224` 的 tokenizer
- PaliGemma 是一个 **gated model**（受限模型），需要在 HuggingFace 上同意使用协议后才能下载
- 当前 HuggingFace token 没有访问 gated repos 的权限，或者尚未在 PaliGemma 模型页面接受协议

**解决方法**：
1. 前往 [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) 模型页面，接受使用协议
2. 确保 HuggingFace token 具有访问 gated repos 的权限：
   - 登录 [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - 如果使用 fine-grained token，勾选 "Access public gated repositories" 权限
3. 确认 token 配置正确：
   ```bash
   huggingface-cli whoami  # 确认已登录
   ```

## 修复后状态

- 环境版本冲突（问题 1-3）：已全部修复
- eval.sh 配置问题（问题 4）：已修复，添加了 `--rename_map` 和 `--policy.empty_cameras`
- HuggingFace 权限问题（问题 5）：需要用户在 HuggingFace 网站上手动操作

## 经验教训

1. lerobot 的 `libero` extra 依赖 `hf-libero`（HuggingFace 的 fork），而非原始 `libero` 包。安装前应查看 `pyproject.toml` 确认正确的包名。
2. 多用户共享的 `/tmp` 目录下的固定路径文件容易产生权限冲突。
3. 使用 `pip install -e ".[libero]"` 是安装 lerobot libero 支持的推荐方式，可避免手动处理依赖。
4. 运行日志中大量出现的 `__del__` 阶段 EGL 错误是清理噪音，不是真正的错误根因，分析时应先过滤掉这些噪音，找到实际的致命异常。
5. 使用需要特殊权限的 gated model 前，需提前在 HuggingFace 接受协议并配置好 token 权限。
