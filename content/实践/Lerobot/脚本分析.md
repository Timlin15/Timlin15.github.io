Lerobot的核心代码在`src/lerobot`中，这个文件夹下有：
- async_inference  
- **datasets**     
- model  
- outputs    
- \_\_pycache\_\_  
- scripts        
- transport
- cameras          
- envs         
- motors  
- policies   
- rl           
- teleoperators  
- utils
- configs          
- \_\_init\_\_.py  
- optim  
- processor 
- robots   
- templates   
- \_\_version\_\_.py
抛开一些Python文件，其中主要是几部分构成：

1. 核心算法与策略
	- policies/: 这是核心目录，存放各种机器人控制策略的具体实现代码（例如 ACT, Diffusion Policy, VQ-Bet 等）。每个子目录通常对应一种特定的策略模型。
	- model/: 存放通用的模型组件，目前包含 kinematics.py，主要涉及简单的运动学模型，独立于具体策略。
	- optim/: 包含优化器（Optimizers）和学习率调度器（Schedulers）的配置与实现。
	- processor/: 负责数据的预处理和后处理流水线。包含归一化（Normalization）、Tokenization、图像处理以及 Gym Action/Observation 的处理逻辑。
2. 硬件与机器人抽象
	- robots/: 定义具体机器人的配置和类。这里包含了不同机器人（如 Koch, Aloha, So-100 等）的硬件映射和初始化逻辑。
	- cameras/: 相机硬件的驱动和抽象层，用于连接和读取不同的摄像头设备（如 OpenCV 相机, Intel RealSense 等）。
	- motors/: 电机硬件的驱动和抽象层，用于控制不同的电机（如 Dynamixel, Feetech 等）。
	- teleoperators/: 用于遥操作（Teleoperation）的设备接口，支持通过手柄、VR 设备或主手（Leader Arm）来控制机器人并收集数据。
3. 数据与环境
	- datasets/: 处理数据集的加载、下载、上传和格式化。LeRobot 拥有自己的数据格式，这里的代码负责与 Hugging Face Hub 交互以及数据流的构建。
	- envs/: 包含仿真环境的封装，通常遵循 Gymnasium 接口，用于在仿真中训练或评估策略。
4. 系统功能与工具
	- configs/: 存放项目的默认配置文件（通常是 .yaml 格式），用于 Hydra 配置管理系统，涵盖默认的策略、环境和硬件配置。
	- utils/: 通用的工具函数集合，包含日志记录、随机种子设置、路径管理、可视化工具等。
	- scripts/: 包含各种实用脚本，用于项目维护、测试或一些特定的辅助任务。
	- templates/: 可能包含代码生成或项目结构的模板文件。
5. 进阶功能
	- transport/: 处理通信相关的逻辑（使用 gRPC），用于跨进程或跨设备的分布式控制与数据传输。
	- async_inference/: 支持异步推理的模块，用于在部署时提高模型推理的吞吐量和响应速度。
	- rl/: 包含强化学习（Reinforcement Learning）相关的组件，如 Actor（执行者）、Learner（学习者）、Buffer（经验回放缓冲区）等，用于在线学习或微调。
	总结来说，src/lerobot 的结构非常模块化：policies 和 datasets 是算法核心，robots、cameras、motors 构成了硬件抽象层，而其他文件夹则提供了必要的辅助功能和系统支持。

## 推理的接口
首先先在mujoco环境中创建一个Franka机械臂，然后仿照[[SmolVLA应用]]创建一个机械臂实例，对于从真机到模拟环境的改动，可以依照下面的Gymnasium的最小实现仿照完成：
```Python
# 这就是 Gymnasium 的全部核心 API 
env = gym.make("SomeEnv") 
obs, info = env.reset() # 重置，拿初始观测 
obs, reward, done, truncated, info = env.step(action) # 执行动作 
image = env.render() # 获取图像 

# 了解空间定义（用于检查维度） 
env.observation_space # 观测空间 
env.action_space # 动作空间
```

## Eval的接口
整体而言，eval文件分为以下几个部分：
```
lerobot_eval.py 
│ 
├── 导入部分 
│ ├── make_policy, make_pre_post_processors ← 模型工厂 
│ ├── make_env, make_env_pre_post_processors ← 环境工厂 
│ ├── preprocess_observation ← 观测预处理 
│ └── write_video ← 视频保存 
│ 
├── rollout() ← ⭐ 核心函数：跑一轮推理 
│ 
├── eval_policy() ← 多 episode 评估循环 
│ 
└── main() / CLI ← 命令行入口
```
先从最核心的`rollout()`函数开始，这个函数本质就是先前推理的官方实现，可以见[[SmolVLA应用]]。
但是这个官方实现采用并行的方式进行测试，加快了速度：

```Python
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
```

除此之外，大致核心逻辑是相同的：先创建一个虚拟环境，然后

```Python
    policy.reset()
    observation, info = env.reset(seed=seeds)
    ...
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
	...
	while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
    	observation = preprocess_observation(observation)
        observation = add_envs_task(env, observation)
        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)
        # Convert to CPU / numpy. 由于env.step() 只接受 numpy array，不接受 torch tensor
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
        
        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)
           
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        
        step += 1
        
	ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    
    return ret
```

剩余还要处理一些用于显示eval进度和最后输出画面的步骤。这里不详述了。

## train脚本
