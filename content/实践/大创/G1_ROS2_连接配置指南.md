# Unitree G1 ROS2 连接配置指南

## 背景知识

### SSH vs ROS2

- **SSH**：远程登录工具，连上后获得 G1 内部 Orin 计算机的终端 shell，可以装软件、改配置、看日志，但本身不提供与机器人关节/传感器交互的能力。
- **ROS2**：机器人通信框架，解决控制程序如何以实时、结构化的方式与机器人电机控制器和传感器交互的问题。底层基于 DDS（CycloneDDS）协议。

两者配合使用：SSH 进 Orin 做环境配置和 debug，ROS2 在外部工作站上跑控制算法并实时收发数据。

### G1 内部网络架构

| 设备 | IP 地址 |
|------|---------|
| 你的电脑 | `192.168.123.162`（网口 `eno2`） |
| RockChip 控制板 | `192.168.123.161` |
| Orin 开发板 | `192.168.123.164` |
| LiDAR | `192.168.123.120` |

## 配置步骤

### 1. 确认网络连接

用网线连接电脑和 G1，通过 `ifconfig` 找到连接 G1 的网口（IP 在 `192.168.123.x` 网段的那个）：

```bash
ifconfig
```

确认网口名称（本例为 `eno2`，IP 为 `192.168.123.162`）。如果还没配静态 IP，在网络设置中将该接口的 IPv4 模式改为手动，设置 IP 为 `192.168.123.99` 或其他未占用的地址，子网掩码 `255.255.255.0`。

验证连通性：

```bash
ping 192.168.123.161  # RockChip
ping 192.168.123.164  # Orin
```

### 2. 确认 ROS2 基础环境

检查 `.bashrc` 中是否已有 ROS2 环境：

```bash
cat ~/.bashrc | grep -i -E "ros|cyclone|rmw|unitree|dds"
```

如果已有 `source /opt/ros/humble/setup.bash`，则不需要再配。验证：

```bash
echo $ROS_DISTRO
# 应输出 humble
```

### 3. 克隆并编译 unitree_ros2

```bash
cd /data2/你的用户名
git clone https://github.com/unitreerobotics/unitree_ros2.git
cd unitree_ros2
```

安装所有依赖（避免逐个手动安装）：

```bash
sudo rosdep init     # 如果报错说已初始化过，跳过
rosdep update
rosdep install --from-paths . --ignore-src -r -y
```

如果缺少特定包（如 `rosidl_default_generators`），也可以手动装：

```bash
# 规律：包名下划线换横杠，加 ros-humble- 前缀
sudo apt install ros-humble-rosidl-default-generators
```

安装 CycloneDDS RMW 包（必须）：

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

编译 CycloneDDS：

```bash
cd cyclonedds_ws
source /opt/ros/humble/setup.bash
colcon build
cd ..
```

编译主 workspace：

```bash
source /opt/ros/humble/setup.bash
colcon build
```

### 4. 配置 setup.sh

编辑仓库根目录下的 `setup.sh`，修改以下内容：

```bash
#!/bin/bash
echo "Setup unitree ros2 environment"
source /opt/ros/humble/setup.bash
source /data2/你的用户名/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="eno2" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
```

修改要点：

- `foxy` → `humble`（匹配你安装的 ROS2 版本）
- `$HOME/unitree_ros2/` → 你的实际路径
- `NetworkInterface name=""` → `name="eno2"`（你连接 G1 的网口名）

### 5. 验证连接

```bash
source /data2/你的用户名/unitree_ros2/setup.sh
ros2 topic list
```

如果只看到 `/parameter_events` 和 `/rosout`，先停掉 daemon 再试：

```bash
ros2 daemon stop
ros2 topic list
```

成功后应能看到 G1 发布的大量 topic，包括：

- `/lowstate` — 关节低级状态
- `/sportmodestate` — 运动模式状态
- `/dex3/left/state`、`/dex3/right/state` — Dex3 手部状态
- `/utlidar/cloud_livox_mid360` — LiDAR 点云
- 以及其他控制和传感器 topic

### 6. 查看数据

```bash
ros2 topic echo /lowstate          # 关节状态
ros2 topic echo /sportmodestate    # 运动模式状态
ros2 topic echo /dex3/right/state  # 右手状态
```

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `find_package` 报错缺包 | `sudo apt install ros-humble-包名`（下划线换横杠） |
| `librmw_cyclonedds_cpp.so` 找不到 | `sudo apt install ros-humble-rmw-cyclonedds-cpp` |
| `ros2 topic list` 只显示两个默认 topic | `ros2 daemon stop` 后重试 |
| ping 不通 G1 | 检查网线连接、网口 IP 配置是否在 `192.168.123.x` 网段 |
| CycloneDDS 不支持 `unitree_hg` 格式 | 卸载旧版 CycloneDDS，按 unitree_ros2 仓库指引重新编译 |

## 注意事项

- `source setup.sh` 只对当前终端生效，新终端需要重新 source
- 不建议将 unitree_ros2 workspace 的 source 写入 `.bashrc`，避免与其他用户冲突
- 系统级依赖（ROS2 安装、网络配置）共用，workspace 各建各的
