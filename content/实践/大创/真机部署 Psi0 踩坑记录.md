# Psi0 在宇树 G1 上的部署整合总结

## 1. 目标与当前状态

目标是将已训练完成的 ACT 权重部署到宇树 G1 真机，使用 Psi0 的 infra 跑通推理服务端、机器人控制客户端和相机服务端三端链路。

截至 2026-04-07，当前状态如下：
- 推理服务端已基本就绪，使用 `Psi0/.venv`
- 机器人控制客户端已基本就绪，使用 conda 环境 `psi_deploy`
- 相机服务端已就绪，最终改用 `teleimager` 环境运行 `realsense_server.py`
- 三端环境边界已经明确，后续重点转向整机联调与安全验证

## 2. 部署对象与基础信息

### 2.1 硬件与网络

- 宇树 G1 通过网线连接本机 `eno2`
- 电机 IP：`192.168.123.161`
- G1 开发板 IP：`192.168.123.164`

### 2.2 模型与代码

- ACT 权重已整理至 `/mnt/data2/linjianqi/ckpt`
- 本地项目仓库为 `Psi0`
- 推理虚拟环境位于 `Psi0/.venv`

### 2.3 开发板现状

- 开发板上已有多个历史环境，包括 `vision` 和 `teleimager`
- 相机服务脚本为 `~/realsense_server.py`
- 最终可工作的 RealSense 运行环境不是 `vision`，而是 `teleimager`

## 3. 三端职责划分

| 端 | 环境 | 作用 |
|------|------|------|
| 推理服务端 | `Psi0/.venv`（uv 管理） | 加载 ACT 模型并提供 HTTP 推理接口 |
| 机器人控制客户端 | `psi_deploy`（conda 管理，位于 `/data2/yangky/miniconda3/envs/psi_deploy`） | 运行 teleop、DDS、机器人控制逻辑 |
| 相机服务端 | G1 开发板 `teleimager` 环境 | 负责 RealSense D435I 图像采集与传输 |

## 4. 最终确认的部署方案

### 4.1 推理服务端

- 使用 `Psi0/.venv`
- 由 `uv` 管理
- 运行脚本：`src/act/deploy/act_g1_serve_simple.py`

### 4.2 机器人控制客户端

- 使用 conda 环境 `psi_deploy`
- 运行脚本：`real/deploy/act_inference.py`
- 启动前必须执行 `export PYTHONPATH=""`，避免 ROS Humble 污染 Python 包解析

### 4.3 相机服务端

- 在 G1 开发板上运行 `~/realsense_server.py`
- 最终运行解释器为 `/home/unitree/miniconda3/envs/teleimager/bin/python`
- 不再依赖 `vision` 环境

## 5. 关键排查结论

1. 权重目录问题已解决
- 可用目录为 `/mnt/data2/linjianqi/ckpt`
- 部署所需配置已整理好，不需要再重建 run 目录

2. uv 环境位置已确认
- 使用 `Psi0/.venv`
- 不是此前假设的 `.venv-psi`

3. 推理端与控制端不能强行共用一套环境
- 推理端偏 Python 服务依赖
- 控制端依赖 `pinocchio`、`casadi`、`pink`、DDS 等机器人学组件
- 最稳定方案是两套环境拆开管理

4. ROS Humble 会污染 conda 环境
- 会导致 ROS 自带 `pinocchio` 覆盖 conda 中带 `casadi` bindings 的版本
- 运行客户端前必须清空 `PYTHONPATH`

5. 客户端代码原本存在多处对当前工作目录的隐式依赖
- 相对路径导致 URDF、TorchScript 文件在不同启动目录下找不到
- 已整理出基于 `__file__` 的根本修复方案

6. RealSense 问题不是单一 bug，而是“平台兼容 + 历史环境污染 + 环境选型错误”的叠加
- Jetson 上 `pyrealsense2` 不能默认相信 pip wheel
- `vision` 环境里的历史改动让 RealSense Python 绑定进一步变得不可用
- 同机已有的 `teleimager` 环境反而是现成可用的工作状态

## 6. 踩坑与解决方案总表

| 编号 | 问题 | 根因 | 解决方案 |
|------|------|------|----------|
| 1 | `uv sync` 后依赖互相丢失 | `uv sync` 会收敛到指定 group 的精确状态 | `uv sync --group psi --group serve` |
| 2 | `pip install` 装进 `.local` | `.venv` 里没有可用的 `pip`，实际调用到系统 pip | 改用 `uv pip install` |
| 3 | teleop 不适合放进 uv 环境 | 机器人学依赖更适合 conda-forge 二进制包 | 控制端单独用 conda `psi_deploy` |
| 4 | `__archspec` 错误 | 导出的 yaml 带了本机不满足的微架构约束 | 删除 `_x86_64-microarch-level` |
| 5 | `Failed to build 'av'` | `av`/`aiortc` 依赖 ffmpeg 开发库，且 teleop 不需要 | 从 yaml 中移除 `av` 与 `aiortc` |
| 6 | `uninstall-no-record-file` | pip 试图卸载 conda 已安装的重复包 | 删除 pip section 中和 conda 重复的包 |
| 7 | `pinocchio.casadi` 导入失败 | ROS Humble 覆盖了 conda 版本 | `export PYTHONPATH=""` |
| 8 | `psi_deploy` 缺少间接依赖 | 导出的 yaml 不完整，且依赖链被破坏 | 手动补装缺失包 |
| 9 | NumPy 版本冲突 | cv2 针对 NumPy 1.x 编译，环境装成了 2.x | 降级到 `numpy<2` |
| 10 | `cyclonedds` 与 `unitree-sdk2py` 不兼容 | 默认安装了过新的 `cyclonedds` | 固定安装 `cyclonedds==0.10.2` |
| 11 | URDF 路径依赖 CWD | 代码使用相对工作目录路径 | 改为基于 `__file__` 的绝对路径 |
| 12 | `adapter_jit.pt` / `amo_jit.pt` 找不到 | TorchScript 文件加载路径依赖 CWD | 改为基于 `__file__` 的绝对路径 |
| 13 | 无灵巧手时客户端永久阻塞 | DDS 手部控制器无超时等待 | 增加超时，并回退到 `DummyHandController` |
| 14 | 客户端请求了错误服务端端口 | `act_inference.py` 中 URL 是旧端口 | 改成 `http://localhost:22085/act` |
| 15 | Jetson 上 `pyrealsense2` 无法稳定识别 D435I | pip wheel、系统库版本、Jetson USB 栈和历史环境修改叠加 | 放弃 `vision`，改用已验证可用的 `teleimager` 环境 |
| 16 | 固件升级前 RGB 传感器报错 | 固件版本与当前硬件/驱动不兼容 | 升级固件到 `5.17.0.10` |
| 17 | 客户端启动即发 DDS 电机指令，机器人抽搐 | 控制线程启动过早，依赖尚未就绪 | 调整启动顺序，并在机器人站稳后再启动客户端 |

## 7. 关键问题的具体处理

### 7.1 推理服务端环境问题

问题集中在 `uv` 使用方式：
- `uv sync` 分组会互相覆盖
- `.venv` 中不能假设 `pip` 可直接用

最终做法：

```bash
uv sync --group psi --group serve
uv pip install -e ./real
uv pip install -e ./unitree_sdk2_python
```

### 7.2 `psi_deploy` 环境重建问题

原始 `psi_deploy_env.yaml` 直接复用失败，主要包括：
- CPU 架构虚拟包不兼容
- `av` 和 `aiortc` 构建失败
- conda 和 pip 混装导致重复卸载
- 导出文件缺少部分间接依赖
- NumPy 主版本与 cv2 不兼容

处理策略不是“原样重建”，而是“按当前机器裁剪 yaml，再手动补依赖”。

典型修复命令：

```bash
sed '/^\s*- _x86_64-microarch-level/d' psi_deploy_env.yaml > psi_deploy_env_fixed.yaml
sed -i '/^\s*- av==/d; /^\s*- aiortc/d' psi_deploy_env_fixed.yaml
sed -i '/^\s*- numpy==/d; /^\s*- pin\b/d; /^\s*- pin==/d; /^\s*- pin-pink/d; /^\s*- setuptools==/d; /^\s*- scipy==/d; /^\s*- packaging==/d' psi_deploy_env_fixed.yaml
```

补装缺失依赖：

```bash
conda run -n psi_deploy pip install urllib3 requests glfw absl-py pyopengl 'etils[epath]' \
    pyparsing matplotlib prompt_toolkit jinja2 networkx 'sympy>=1.13.3'
conda run -n psi_deploy pip install cyclonedds==0.10.2
```

NumPy 版本修复：

```bash
/data2/yangky/miniconda3/bin/conda install -n psi_deploy 'numpy<2' -c conda-forge -y
```

### 7.3 ROS 污染问题

现象是 `from pinocchio import casadi as cpin` 失败，实际导入的是：

`/opt/ros/humble/lib/python3.10/site-packages/pinocchio`

这说明系统 ROS 路径抢在 conda 环境前面。处理方式很直接：

```bash
export PYTHONPATH=""
```

### 7.4 客户端代码路径问题

客户端若从错误目录启动，会导致：
- URDF 文件找不到
- `adapter_jit.pt` 找不到
- `amo_jit.pt` 找不到

根本修复方式：
- 将所有资源路径改为基于 `__file__` 计算
- 不再依赖当前工作目录

临时运行规避方式：

```bash
cd /data2/linjianqi/Psi0/real/deploy
python act_inference.py
```

### 7.5 无灵巧手硬件时的阻塞问题

没有 Dex3_1 手部硬件时，DDS 手部控制器会无限等待订阅成功，导致客户端卡死。

最终方案：
- 在 `robot_hand_unitree.py` 中加入超时
- 在 `master_whole_body.py` 中加入 `DummyHandController`
- 初始化失败后回退到 Dummy，避免全流程被手部硬件阻塞

这意味着：
- 手部动作指令会被静默忽略
- `hand_joints` 观测为全零
- 不影响 arm、leg、waist 的基本控制链路

### 7.6 相机服务问题总览

相机侧最终不是“修一个包”这么简单，而是经历了两轮定位：

1. 第一轮定位得到 Jetson 上 `pyrealsense2` 的通用根因分析；
2. 第二轮定位发现这台开发板上现成的 `teleimager` 环境才是最稳的工作状态。

下面按时间顺序整理。

## 8. 相机服务故障排查与最终修复

### 8.1 初始现象

`realsense_server.py` 在开发板上启动时反复报错：

```text
Server started, waiting for client requests...
Failed to start RealSense: No device connected
```

脚本本身不崩溃，但每次收到请求都只能返回空数据。

### 8.2 第一轮分析：Jetson 上的通用根因

这部分结论来自对 `RealSense_Jetson_故障排查报告.md` 的整理，反映的是 Jetson 平台上 `pyrealsense2` 常见的两层问题。

#### 根因一：缺少 udev rules

先检查系统是否安装了 RealSense 的 udev 规则：

```bash
ls /etc/udev/rules.d/ | grep realsense
# （无输出）
```

`/dev/bus/usb/002/003` 的原始权限为 `crw-rw-r--`，属主为 `root:plugdev`。普通用户无法直接打开该 USB 设备。

最容易误判的一点是：`pyrealsense2` 在无法打开设备时，不会明确报权限错误，而是直接表现为：

```text
No device connected
```

修复方法：

```bash
sudo cp ~/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

修复后设备权限放开，但问题仍未完全解决，说明还有第二层问题。

#### 根因二：pip wheel 与系统编译版本不匹配

原生命令可以正常看到设备：

```bash
rs-enumerate-devices --version
# rs-enumerate-devices  version: 2.57.5.0
# （可以正常枚举到 D435I）
```

但 Python 环境里的包版本是：

```text
pip show pyrealsense2
# Version: 2.55.1.6486
```

对应关系如下：

| 组件 | 版本 | 来源 |
|------|------|------|
| 系统 librealsense | 2.57.5.0 | 从 `~/librealsense` 源码编译安装 |
| pip pyrealsense2 | 2.55.1.6486 | PyPI wheel（自带旧版 librealsense） |

关键点是：`pip install pyrealsense2` 拿到的 wheel 内部自带 librealsense 动态库，因此 Jetson 上 Python 实际调用的可能不是系统源码编译的库，而是 pip wheel 里内嵌的旧版库。这个旧版库在 Jetson USB 3.2 栈上无法正确枚举 D435I，不报错，只返回空列表。

这也是为什么：
- `lsusb` 正常
- `rs-enumerate-devices` 正常
- Python 里却还是 `0 devices`

#### 第一轮通用修复思路

按这个分析，Jetson 上正确的通用做法应当是：

1. 使用源码编译的 librealsense
2. 安装对应的 udev rules
3. 使用源码编译生成的 Python 绑定
4. 避免直接依赖 PyPI 提供的 `pyrealsense2` wheel

对应验证步骤：

```bash
find ~/librealsense/build -name "pyrealsense2*.so"
pip uninstall pyrealsense2 -y
cp /home/unitree/librealsense/build/Release/pyrealsense2.cpython-38-aarch64-linux-gnu.so \
   /home/unitree/miniconda3/envs/vision/lib/python3.8/site-packages/
python3 -c "import pyrealsense2 as rs; d = rs.context().query_devices(); print('Devices:', len(d))"
```

这部分结论作为 Jetson 平台的通用经验仍然成立。

### 8.3 第二轮分析：这台开发板上的实际问题比“版本不匹配”更复杂

继续往下排查后发现，这台开发板并不是只有“pip wheel 版本不对”这么简单，而是还有历史环境污染和版本兼容差异。

#### `vision` 环境已损坏

此前调试中曾通过 `patchelf` 强制修改 `vision` 环境中 `pyrealsense2.cpython-38-aarch64-linux-gnu.so` 的 RUNPATH，导致它去链接 apt 版本的 librealsense 2.57.7，而它自身编译时使用的却是 2.57.5 头文件，最终触发 ABI 不匹配，运行时出现：

```text
corrupted double-linked list
```

这说明 `vision` 环境已经不再是可信的排查基线。

#### 同机已有 `teleimager` 环境完全正常

开发板上已有 `unitreerobotics/teleimager` 仓库及其配套 conda 环境。直接验证发现：

```bash
conda activate teleimager
python -c "
import pyrealsense2 as rs
ctx = rs.context()
d = ctx.query_devices()
print('Devices:', len(d))
"
# Devices: 1
```

并且三个传感器全部可见：
- Stereo Module
- RGB Camera
- Motion Module

进一步使用仓库内置测试脚本验证彩色出流：

```bash
cd ~/teleimager && python test_rs.py
# [测试] 尝试分辨率 848x480 @ 60 FPS [仅彩色]...
# ==========> ✅ 测试成功！相机满血复活！ <==========
```

### 8.4 更深入的兼容性结论

在对比多个环境后，得到如下结论：

| 层 | 细节 |
|---|---|
| 系统 librealsense | `/usr/local/lib/librealsense2.so.2.50`（v2.50，能正常枚举 D435I） |
| `teleimager` pyrealsense2 | Python 3.10，链接 `/usr/local/lib/librealsense2.so.2.50`，正常 |
| `vision` pyrealsense2 | Python 3.8，链接 librealsense 2.57.5/2.57.7，无法稳定枚举 RGB Camera |

可以看出：
- 问题与 Python 3.8/3.10 本身无关
- 根本差异在于底层 librealsense 版本组合
- 这块 Jetson（JetPack 6.2，L4T 5.15.148-tegra）上，librealsense **2.57.x** 与 D435I 的兼容性明显差于 **2.50**

### 8.5 放弃源码重编路线

尝试为 `vision` 环境重编 Python 3.8 binding 时又遭遇了一系列链式问题：

- `nlohmann/json 3.10.5` 不支持 `basic_json` 的 `CustomBaseClass` 模板参数（需要 ≥ 3.11）
- `pybind11 3.0` 移除了 `def_property + keep_alive` 组合（需要降回 2.x）
- librealsense 2.57.5 wrapper 源码调用了 `rs2_get_udp_ttl` 等 2.50 中不存在的符号，无法与 `/usr/local` 2.50 库链接
- librealsense 2.50.x wrapper 源码又与 `/usr/local/include` 的头文件存在 API 小版本差异

这说明从源码重编去“修活 vision”虽然理论可行，但在当前机器上并不是成本最低、风险最低的方案。

### 8.6 最终可用修复方案

`realsense_server.py` 实际只依赖：
- `pyrealsense2`
- `cv2`
- `numpy`
- `zmq`

这些包在 `teleimager` 环境中都已具备，而且相机已验证可正常工作。因此最终选择直接让脚本使用 `teleimager` 环境解释器。

处理方式：

```bash
sed -i '1s|^|#!/home/unitree/miniconda3/envs/teleimager/bin/python\n|' ~/realsense_server.py
chmod +x ~/realsense_server.py
```

验证：

```bash
conda activate teleimager
timeout 15 python ~/realsense_server.py
# Server started, waiting for client requests...
# RealSense: RGB + IR + Depth active.
```

也可以直接执行：

```bash
./realsense_server.py
```

### 8.7 相机端最终结论

相机问题分成两层理解：

1. Jetson 平台上的通用经验：
- 不要默认相信 pip 的 `pyrealsense2` wheel
- 要检查 udev rules
- 要优先使用和系统 librealsense 对齐的 Python 绑定

2. 这台开发板上的最终落地方案：
- 不再继续修 `vision`
- 直接复用现成可用的 `teleimager` 环境
- 当前相机端已恢复，RGB + IR + Depth 全部正常

## 9. 最终推荐启动顺序

### Step 1：G1 开发板相机服务

```bash
# 方式一：直接执行（shebang 已指定 teleimager Python）
~/realsense_server.py

# 方式二：显式指定解释器
/home/unitree/miniconda3/envs/teleimager/bin/python ~/realsense_server.py
```

### Step 2：本机推理服务端

```bash
cd /data2/linjianqi/Psi0
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src"
python src/act/deploy/act_g1_serve_simple.py \
    --host=0.0.0.0 \
    --port=22085 \
    --run-dir=/data2/linjianqi/ckpt \
    --ckpt-step=42000
```

### Step 3：本机机器人控制客户端

机器人必须先处于正常站立状态，再启动客户端：

```bash
conda activate psi_deploy
export PYTHONPATH=""
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>eno2</NetworkInterfaceAddress></General></Domain></CycloneDDS>"
cd /data2/linjianqi/Psi0/real/deploy
python act_inference.py
```

## 10. 安全注意事项

1. 客户端一旦初始化成功，就可能立即开始发送 DDS 电机控制指令
2. 启动客户端前，必须确认机器人已经站稳
3. 启动客户端前，必须确认相机服务端和推理服务端已经就绪
4. 在相机服务或推理服务未就绪时直接拉起客户端，存在机器人抽搐或异常动作风险

## 11. 已知限制

| 项目 | 状态 | 说明 |
|------|------|------|
| 灵巧手（Dex3_1） | 无硬件，使用 `DummyHandController` 兜底 | 手部动作被忽略，`hand_joints` 为全零，可能轻微影响模型表现 |
| 模型 `action_dim=36` | 与无手配置有功能偏差 | 前 14 维手部动作会被丢弃，但不影响 arm/leg/waist 控制 |
| `vision` 环境中的 RealSense | 不再作为运行基线 | 历史改动与版本兼容问题过多，当前以 `teleimager` 为准 |

## 12. 附录：踩坑分类与经验教训（2026-04-07）

### 12.1 踩坑分类

将全部 17 条问题按性质分类，区分“本可避免”与“项目或平台本身带来的”。

#### A 类：工具使用不熟悉

| # | 问题 | 本质 |
|---|------|------|
| 1 | `uv sync` 互相覆盖 | 没搞清楚 uv sync 的“精确收敛”语义 |
| 2 | `pip install` 装进 `.local` | 没意识到 `.venv` 里要用 `uv pip` 而不是系统 `pip` |
| 6 | `uninstall-no-record-file` | conda + pip 混装时没清理 pip section 里的重复包 |

共同点：这三个问题在工具文档的第一屏或 FAQ 里都有说明，是“没读文档就上手”的典型代价。

#### B 类：跨机器迁移的系统性问题

| # | 问题 | 本质 |
|---|------|------|
| 4 | `__archspec` 错误 | `conda env export` 的完整输出包含平台相关虚包，不适合跨机器使用 |
| 5 | `av`/`aiortc` 构建失败 | 目标机器没有 ffmpeg dev 库，而源 yaml 包含非必要的编译依赖 |
| 8 | `psi_deploy` 缺少间接依赖 | 完整 export 的 yaml 跨平台复现能力差，间接依赖在不同机器上解析路径不同 |
| 9 | NumPy 版本冲突 | 迁移时没约束已知的版本上限 |

共同点：都源于“把别人机器上的完整环境 snapshot 直接搬到新机器”。正确做法是用 `--from-history` 导出显式安装列表，再手动清理平台相关约束。

#### C 类：代码设计缺陷

| # | 问题 | 本质 |
|---|------|------|
| 11 | URDF 路径依赖 CWD | 代码里用相对路径加载资源，换目录就崩 |
| 12 | TorchScript 路径依赖 CWD | 同上 |
| 13 | 无灵巧手时永久阻塞 | 可选硬件初始化没有超时和降级机制 |
| 14 | 客户端请求旧端口 | 硬编码了过期配置，没有单一配置来源 |
| 17 | 启动即发电机指令，机器人抽搐 | 控制线程在依赖就绪前就开始发真实指令，缺少 ready state 检查 |

共同点：修复成本都不高，但每次部署都会踩一遍，属于本应在代码设计阶段解决的问题。其中 #17 还有安全风险。

#### D 类：架构决策

| # | 问题 | 本质 |
|---|------|------|
| 3 | teleop 不适合放进 uv 环境 | pinocchio、casadi 等机器人学库的分发形式与 pip 生态不兼容 |

这不是单点错误，而是项目初期就应该划清的边界：推理服务端适合 pip/uv，控制端适合 conda-forge。

#### E 类：系统或平台兼容性

| # | 问题 | 本质 |
|---|------|------|
| 7 | ROS 污染 `PYTHONPATH` | ROS Humble `setup.bash` 注入路径是其已知行为，conda 不能自动隔离 |
| 10 | `cyclonedds` 版本不兼容 | `unitree-sdk2py` 上游没有锁定 cyclonedds 版本 |
| 15 | `pyrealsense2` 与 Jetson/RealSense 组合不稳定 | Jetson 上 native extension、系统库和 USB 栈兼容性复杂 |
| 16 | 固件版本导致 RGB 报错 | 固件与驱动版本不对齐，属于硬件环境准备不完整 |

这类问题里，#7、#10 是已知坑但文档没说清；#15、#16 更接近平台兼容性问题。

### 12.2 经验教训

#### 经验一：工具要先读文档，尤其是同类工具

`uv` 和 `pip` 看起来都在“装包”，但 `uv sync` 是声明式收敛，`pip install` 是追加式。用任何新工具前，先搞清楚核心操作语义，能省掉大量调试时间。

#### 经验二：conda yaml 跨机器迁移有固定清洗流程

迁移之前应固定做以下步骤：

1. 去掉 `_x86_64-microarch-level` 等平台虚包
2. 去掉目标机器上不需要或无法构建的编译依赖
3. 去掉 pip section 里与 conda 重复的包
4. 约束已知的版本上限，例如 `numpy<2`

这不是“这次特殊情况”，而是跨机器迁移的标准处理。

#### 经验三：机器人学项目的依赖天然是异构的

这个项目的依赖分三层：
- 推理层：PyTorch、HTTP 服务，适合 pip/uv
- 控制层：pinocchio、casadi、DDS，适合 conda-forge
- 驱动层：librealsense、固件、系统库，属于系统级安装

越早承认这个分层，越早拆环境，后续迁移成本越低。

#### 经验四：部署代码里不允许有 CWD 依赖

任何需要部署的脚本，资源路径都必须基于 `Path(__file__).parent` 计算，而不是依赖当前工作目录。相关问题在本次占了两条，而修复只需要几行代码。

#### 经验五：控制真实硬件的代码必须有 ready state 和降级策略

- 控制线程在所有依赖就绪前不能发真实指令
- 可选硬件初始化必须有超时和降级路径

这两点处理不好，代价就是机器人抽搐、卡死或需要人工复位。

#### 经验六：系统级污染要写进启动脚本，不能靠口头约定

ROS Humble 的 `PYTHONPATH` 污染、`cyclonedds` 版本锁定、`CYCLONEDDS_URI` 网卡绑定，都是每次启动前必须执行的动作，应写入脚本。

#### 经验七：在 Jetson 上，任何 native extension 都不能默认相信 pip wheel

Jetson 是 aarch64 + JetPack 定制内核，pip 上的 wheel 往往不覆盖 Jetson 的 CUDA、V4L2、USB 栈细节。对 `pyrealsense2`、`torch`、`cv2` 这类包，必须优先检查：

1. 是否已有 Jetson 专用预编译版本
2. 同机现有环境里是否已经存在工作的版本

#### 经验八：遇到“从头重建”类问题，先找“已有的工作状态”

这次 `pyrealsense2` 排查的最大教训是：与其从零开始重编，不如先找机器上已经能工作的环境。最终解决方案并不是“把 vision 修好”，而是“发现 teleimager 已经是工作的状态，然后直接复用它”。

## 13. 最终总结

这次部署的核心经验不是“把所有依赖都装进一个环境”，而是：

- 按模块职责拆环境
- 不直接相信跨机器导出的 conda yaml
- 对系统级污染源做显式隔离
- 运行时资源路径不能依赖当前工作目录
- 对可选硬件依赖必须提供超时和降级策略
- Jetson 上优先复用已知可工作的 native 环境，而不是默认从零编译

从部署完成度看（截至 2026-04-07）：
- 推理端基本就绪
- 机器人控制客户端基本就绪
- 相机端已就绪

后续优先级建议：
1. 联调三端链路，确认推理端能收到相机帧
2. 在确保机器人站稳和安全的前提下做真机动作验证
