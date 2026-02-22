本文档计划从简单到详细记录在PI05的一个checkpoint上进行训练中遇到的各种问题等。
## 规划
首先的问题是，任务是什么？训练集哪里来？task的质量决定了训练和方法的价值。由于本次实践的目的是熟悉Lerobot的训练脚本，看有什么潜在的坑，所以采取简单的Libero作为测试项目，并且使用社区开源数据集进行测试。目的是学习掌握
- [x]  配置wandb，观察训练期间的loss等数据
- [x] 打印一个完整 batch 的所有 key、shape、数值范围 
- [ ] 可视化归一化前后的 action 分布（画直方图） 
- [ ] 在 forward 里 print flow matching 的 timestep 采样分布 
- [ ] 比冻结/不冻结 VLM 的训练曲线和 eval 成功率 
- [ ] 记录不同 checkpoint 步数的 LIBERO 各子任务成功率 
- [ ] 用相同 checkpoint 不同 prompt 跑推理，观察动作差异 
- [ ] 修改 num_steps（flow matching 解码步数）看推理速度和质量的 trade-off

## 遇到的问题
首先就是git clone不了，机房没有外网环境，解决办法是在本机clone一遍后用
1. scp传输

   ```bash
   # 本地 → 远程
   scp file.txt user@host:/path/to/dest/
   
   # 远程 → 本地
   scp user@host:/path/to/file.txt ./local/
   
   # 传输整个目录（加 -r）
   scp -r my_folder/ user@host:/path/to/dest/
   ```

2. rsync传输，更优，支持增量传输

   ```bash
   # 同步目录
   rsync -avz my_folder/ user@host:/path/to/dest/
   
   # 加 --progress 显示进度
   rsync -avz --progress file.txt user@host:/path/
   
   # 可以排除.git文件夹中的pack文件，减少传输时间
   rsync -avz --progress ~/lerobot/ A100-36.163.20.107:/mnt/data/linjianqi/lerobot/
   ```

搞定后先是下载环境：
```Bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
pip install -r requirement-ubuntu.txt -i https://mirrors.ivolces.com/pypi/simple
pip install -e ".[pi]"
```
其中由于是安装配置文件，所以要加上`-r`的参数。同时使用镜像源加速。

## 使用mihomo获得外网环境
由于使用镜像源太过麻烦，所以打算直接使用clash内核mihomo以CLI形式获得外网环境。
由于没有sudo权限，下载二进制压缩包版本`.gz`结尾。传到主机后使用
```bash
gunzip mihomo-linux-amd64-v1.19.20.gz 
chmod +x mihomo-linux-amd64-v1.19.20 
mv mihomo-linux-amd64-v1.19.20 mihomo 
mkdir -p mihomo-config
```
这几个命令解压，赋予权限，重命名以及创建配置文件夹。再将本机的yaml配置文件上传到服务器。
创建一个tmux 窗口，由于不认识ghostty终端，需要设置环境变量。
```bash
export TERM=xterm-256color
tmux new -s lerobot
# tmux a -t lerobot 重新进入section
```
用
```bash
/mnt/data/linjianqi/mihomo -d /mnt/data/linjianqi/mihomo-config
# 或
./mihomo -d ./mihomo-config


# 然后修改指令运行的端口
export http_proxy=http://127.0.0.1:7890 
export https_proxy=http://127.0.0.1:7890

# 不用时使用指令
unset http_proxy https_proxy
```
即可开启mihomo。
可以使用`set -g mouse on`来开启鼠标滚动。
```Bash
(base) /mnt/data/linjianqi$ curl -I https://github.com
HTTP/2 200
date: Sat, 14 Feb 2026 08:39:50 GMT
```
成功。
之后就是正常进行操作了。

对于VSCode，则需要用`Ctrl+Shift+P`输入`Preferences: Open Remote Settings (SSH)`后加入
```json
{
  "terminal.integrated.env.linux": {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890"
  }
}
```
如果其他用户占用了改端口，可以在配置文件和export进来的端口中修改再启动。

用以下命令切换节点：
```bash
curl -X PUT http://127.0.0.1:19090/proxies/%F0%9F%9A%80%20%E8%8A%82%E7%82%B9%E9%80%89%E6%8B%A9 \
  -H "Content-Type: application/json" \
  -d '{"name":"🇨🇳 台湾 01"}'
```
![59f51f4ff5cdd936822dc23593f32711.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260214215709870.png)

## 配置环境
正常安装即可，首先创建conda环境，此处用3.10版本的Python，然后安装依赖。注意在存储非常紧张的情况下（这貌似还蛮常见的），需要重新指定conda的安装区域和安装缓存。
```Bash
conda create -n lerobot python=3.10
# 可以通过 -p 指定路径，此时不可指定名字
# 在 data1 创建一个新的缓存目录
mkdir -p /mnt/data1/linjianqi/conda_pkgs
# 修改 conda 配置，把这个路径设为首选缓存路径
conda config --add pkgs_dirs /mnt/data1/linjianqi/conda_pkgs
# 清理缓存
conda clean --all
# 下载
conda create -p /mnt/data1/linjianqi/conda/lerobot python=3.10
# 注意启动环境也要打绝对路径
conda activate /mnt/data1/linjianqi/conda/lerobot
```


```
pip install -r requirements-ubuntu.txt
```
安装lerobot依赖，然后使用
```Bash
pip install -e ".[pi]"
```
安装PI05所需的依赖。

| Checkpoint                      | 用途                                   | 模型大小   |
| ------------------------------- | -------------------------------------- | ---------- |
| `lerobot/pi05_base`             | 基础预训练模型，用于微调到自定义数据集 | ~4B params |
| `lerobot/pi05_libero_base`      | 在 LIBERO 上继续预训练的基础模型       | ~4B params |
| `lerobot/pi05_libero_finetuned` | 在 LIBERO 上微调好的模型，可直接评估   | ~4B params |
然后用
```bash
# 用 huggingface-cli 或 hf download
huggingface-cli download lerobot/pi05_base
# 如果不行可以使用国内镜像源
export HF_ENDPOINT=https://hf-mirror.com
hf download lerobot/pi05_base

# 或 git clone
git lfs install
git clone https://huggingface.co/lerobot/pi05_base
```
安装PI05的开源权重

### 安装wandb
直接使用pip安装然后login即可
```
pip install wandb
wandb login
```

## 进行测试
首先要设置渲染后端，对于无桌面服务器来说是必须的，同时安装在LIBERO上测试所需依赖：
```bash
export MUJOCO_GL=egl
pip install -e ".[libero]"
```
其中在配置环境的时候遇到了严重的环境问题，主要出在LIBERO环境冲突，主要原因是下载LIBERO环境的时候没有在`pyproject.toml`中查看包的版本，而是去谷歌随便搜了个环境下载，这导致了严重的版本冲突，无法启动脚本，见[[eval.sh 环境版本冲突问题诊断与修复总结]]。
同时，在使用LIBERO这个评测方案的时候也面临许多问题，包括：
1. LIBERO摄像头数量和PI05所需摄像头数量不一致，需要将一个输入摄像头用mask填充
2. LIBERO输出键名和PI05接受键名不一致
这是因为pi05_base这个权重自身导致的。如果切换到pi05_libero_finetuned这个权重就可以测出80%左右的成功率了。也就是这次训练的目标是用训练和微调解决这两个问题

| 实际含义     | libero 输出的键名         | pi05 期望的键名                      |
| ------------ | ------------------------- | ------------------------------------ |
| 主视角摄像头 | observation.images.image  | observation.images.base_0_rgb        |
| 手腕摄像头   | observation.images.image2 | observation.images.right_wrist_0_rgb |

然后使用`lerobot.eval`这个脚本测试会显示bug。
```Bash
export MUJOCO_GL=egl
lerobot-eval \
	--policy.path=lerobot/pi05_base \
	--policy.n_action_steps=10 \
	--env.type=libero \
	--env.task=libero_10 \
	--eval.batch_size=1 \
	--eval.n_episodes=10 \
	--output_dir=./eval_logs/pi05_libero10 \
	--env.max_parallel_tasks=1 \
```
```bash
Traceback (most recent call last):
  File "/mnt/data1/linjianqi/conda/lerobot/bin/lerobot-eval", line 10, in <module>
    sys.exit(main())
  File "/mnt/data1/linjianqi/lerobot/src/lerobot/scripts/lerobot_eval.py", line 809, in main
    eval_main()
  File "/mnt/data1/linjianqi/lerobot/src/lerobot/configs/parser.py", line 233, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/mnt/data1/linjianqi/lerobot/src/lerobot/scripts/lerobot_eval.py", line 528, in eval_main
    policy = make_policy(
  File "/mnt/data1/linjianqi/lerobot/src/lerobot/policies/factory.py", line 526, in make_policy
    validate_visual_features_consistency(cfg, features)
  File "/mnt/data1/linjianqi/lerobot/src/lerobot/policies/utils.py", line 249, in validate_visual_features_consistency
    raise_feature_mismatch_error(provided_visuals, expected_visuals)
  File "/mnt/data1/linjianqi/lerobot/src/lerobot/policies/utils.py", line 214, in raise_feature_mismatch_error
    raise ValueError(
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb', 'observation.images.right_wrist_0_rgb']
- Extra features: ['observation.images.image', 'observation.images.image2']

Please ensure your dataset and policy use consistent feature names.
If your dataset uses different observation keys (e.g., cameras named differently), use the `--rename_map` argument, for example:
  --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}'
```
这会显示键名不一致的问题，采用他推荐的reanem_map则可以成功运行，但是成功率是0%，需要重新训练。

## 训练及微调
为了实现PI05在LIBERO上的原生支持，首先需要修改模型的config，让其适配LIBERO环境中双摄像头的设置，同时修改摄像头分辨率等配置。

```json
{
    "type": "pi05",
    "n_obs_steps": 1,
    "input_features": {
        "observation.images.base_0_rgb": {
            "type": "VISUAL",
            "shape": [
                3,
                224,
                224
            ]
        },
        "observation.images.left_wrist_0_rgb": {
            "type": "VISUAL",
            "shape": [
                3,
                224,
                224
            ]
        },
        "observation.images.right_wrist_0_rgb": {
            "type": "VISUAL",
            "shape": [
                3,
                224,
                224
            ]
        },
        "observation.state": {
            "type": "STATE",
            "shape": [
                32
            ]
        }
    },
    "output_features": {
        "action": {
            "type": "ACTION",
            "shape": [
                32
            ]
        }
    },
    "device": "mps",
    "use_amp": false,
    "push_to_hub": true,
    "repo_id": null,
    "private": null,
    "tags": null,
    "license": null,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "dtype": "float32",
    "chunk_size": 50,
    "n_action_steps": 50,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "time_sampling_beta_alpha": 1.5,
    "time_sampling_beta_beta": 1.0,
    "min_period": 0.004,
    "max_period": 4.0,
    "image_resolution": [
        224,
        224
    ],
    "gradient_checkpointing": false,
    "compile_model": false,
    "compile_mode": "max-autotune",
    "optimizer_lr": 2.5e-05,
    "optimizer_betas": [
        0.9,
        0.95
    ],
    "optimizer_eps": 1e-08,
    "optimizer_weight_decay": 0.01,
    "optimizer_grad_clip_norm": 1.0,
    "scheduler_warmup_steps": 1000,
    "scheduler_decay_steps": 30000,
    "scheduler_decay_lr": 2.5e-06,
    "tokenizer_max_length": 200
}
```
```json
{
    "type": "pi05",
    "n_obs_steps": 1,
    "input_features": {
        "observation.images.image": {
            "type": "VISUAL",
            "shape": [
                3,
                256,
                256
            ]
        },
        "observation.images.image2": {
            "type": "VISUAL",
            "shape": [
                3,
                256,
                256
            ]
        },
        "observation.state": {
            "type": "STATE",
            "shape": [
                8
            ]
        }
    },
    "output_features": {
        "action": {
            "type": "ACTION",
            "shape": [
                7
            ]
        }
    },
    "empty_cameras": 1,
    "device": "mps",
    "use_amp": false,
    "push_to_hub": true,
    "repo_id": null,
    "private": null,
    "tags": null,
    "license": null,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "dtype": "float32",
    "chunk_size": 50,
    "n_action_steps": 10,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "time_sampling_beta_alpha": 1.5,
    "time_sampling_beta_beta": 1.0,
    "min_period": 0.004,
    "max_period": 4.0,
    "image_resolution": [
        224,
        224
    ],
    "gradient_checkpointing": false,
    "compile_model": false,
    "compile_mode": "max-autotune",
    "optimizer_lr": 2.5e-05,
    "optimizer_betas": [
        0.9,
        0.95
    ],
    "optimizer_eps": 1e-08,
    "optimizer_weight_decay": 0.01,
    "optimizer_grad_clip_norm": 1.0,
    "scheduler_warmup_steps": 1000,
    "scheduler_decay_steps": 30000,
    "scheduler_decay_lr": 2.5e-06,
    "tokenizer_max_length": 200
}
```
上下两个分别是`pi05_base`和`pi05_libero_finetuned`两个版本的config文件，可以发现除了输入的视频键值、分辨率不一样，以及有一个`"empty_cameras": 1`和输入输出的动作维度不一样。

可以直接在`train_config`中修改这些不同：
```json
{
    "dataset": {
        "repo_id": "HuggingFaceVLA/libero",
        "root": null,
        "episodes": null,
        "image_transforms": {
            "enable": false,
            "max_num_transforms": 3,
            "random_order": false,
            "tfs": {
                "brightness": {
                    "weight": 1.0,
                    "type": "ColorJitter",
                    "kwargs": {
                        "brightness": [
                            0.8,
                            1.2
                        ]
                    }
                },
                "contrast": {
                    "weight": 1.0,
                    "type": "ColorJitter",
                    "kwargs": {
                        "contrast": [
                            0.8,
                            1.2
                        ]
                    }
                },
                "saturation": {
                    "weight": 1.0,
                    "type": "ColorJitter",
                    "kwargs": {
                        "saturation": [
                            0.5,
                            1.5
                        ]
                    }
                },
                "hue": {
                    "weight": 1.0,
                    "type": "ColorJitter",
                    "kwargs": {
                        "hue": [
                            -0.05,
                            0.05
                        ]
                    }
                },
                "sharpness": {
                    "weight": 1.0,
                    "type": "SharpnessJitter",
                    "kwargs": {
                        "sharpness": [
                            0.5,
                            1.5
                        ]
                    }
                }
            }
        },
        "revision": null,
        "use_imagenet_stats": true,
        "video_backend": "torchcodec",
        "streaming": false
    },
    "env": null,
    "policy": {
        "type": "pi05",
        "n_obs_steps": 1,
        "input_features": {
            "observation.images.image": {
                "type": "VISUAL",
                "shape": [
                    3,
                    256,
                    256
                ]
            },
            "observation.images.image2": {
                "type": "VISUAL",
                "shape": [
                    3,
                    256,
                    256
                ]
            },
            "observation.state": {
                "type": "STATE",
                "shape": [
                    8
                ]
            },
            "observation.images.empty_camera_0": {
                "type": "VISUAL",
                "shape": [
                    3,
                    224,
                    224
                ]
            }
        },
        "output_features": {
            "action": {
                "type": "ACTION",
                "shape": [
                    7
                ]
            }
        },
        "device": "cuda",
        "use_amp": false,
        "push_to_hub": true,
        "repo_id": "Jianqi-Lin/pi05_libero",
        "private": null,
        "tags": null,
        "license": null,
        "pretrained_path": "lerobot/pi05_base",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "bfloat16",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "time_sampling_beta_alpha": 1.5,
        "time_sampling_beta_beta": 1.0,
        "time_sampling_scale": 0.999,
        "time_sampling_offset": 0.001,
        "min_period": 0.004,
        "max_period": 4.0,
        "image_resolution": [
            224,
            224
        ],
        "empty_cameras": 1,
        "tokenizer_max_length": 200,
        "normalization_mapping": {
            "ACTION": "MEAN_STD",
            "STATE": "MEAN_STD",
            "VISUAL": "IDENTITY"
        },
        "gradient_checkpointing": true,
        "compile_model": true,
        "compile_mode": "max-autotune",
        "optimizer_lr": 2.5e-05,
        "optimizer_betas": [
            0.9,
            0.95
        ],
        "optimizer_eps": 1e-08,
        "optimizer_weight_decay": 0.01,
        "optimizer_grad_clip_norm": 1.0,
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 6000,
        "scheduler_decay_lr": 2.5e-06
    },
    "output_dir": "/mnt/data1/linjianqi/lerobot/outputs/pi05_libero",
    "job_name": "pi05_multi_newest_8_gpu_30_9",
    "resume": false,
    "seed": 1000,
    "num_workers": 4,
    "batch_size": 16,
    "steps": 6000,
    "eval_freq": 20000,
    "log_freq": 200,
    "save_checkpoint": true,
    "save_freq": 2000,
    "use_policy_training_preset": true,
    "optimizer": {
        "type": "adamw",
        "lr": 0.0002,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "betas": [
            0.9,
            0.95
        ],
        "eps": 1e-08
    },
    "scheduler": {
        "type": "cosine_decay_with_warmup",
        "num_warmup_steps": 8000,
        "num_decay_steps": 48000,
        "peak_lr": 2.5e-05,
        "decay_lr": 2.5e-06
    },
    "eval": {
        "n_episodes": 50,
        "batch_size": 50,
        "use_async_envs": false
    },
    "wandb": {
        "enable": true,
        "disable_artifact": false,
        "project": "lerobot",
        "entity": null,
        "notes": null,
        "run_id": "rzfptgzx",
        "mode": null
    }
}
```
首先看policy部分的配置：
```json
    "policy": {
        "type": "pi05",
        "n_obs_steps": 1,
        "input_features": {
            "observation.images.image": {
                "type": "VISUAL",
                "shape": [
                    3,
                    256,
                    256
                ]
            },
            "observation.images.image2": {
                "type": "VISUAL",
                "shape": [
                    3,
                    256,
                    256
                ]
            },
            "observation.state": {
                "type": "STATE",
                "shape": [
                    8
                ]
            },
            "observation.images.empty_camera_0": {
                "type": "VISUAL",
                "shape": [
                    3,
                    224,
                    224
                ]
            }
        },
        "output_features": {
            "action": {
                "type": "ACTION",
                "shape": [
                    7
                ]
            }
        },
        "device": "cuda",
        "use_amp": false,
        "push_to_hub": true,
        "repo_id": "Jianqi-Lin/pi05_libero",
        "private": null,
        "tags": null,
        "license": null,
        "pretrained_path": "lerobot/pi05_base",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "bfloat16",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "time_sampling_beta_alpha": 1.5,
        "time_sampling_beta_beta": 1.0,
        "time_sampling_scale": 0.999,
        "time_sampling_offset": 0.001,
        "min_period": 0.004,
        "max_period": 4.0,
        "image_resolution": [
            224,
            224
        ],
        "empty_cameras": 1,
        "tokenizer_max_length": 200,
        "normalization_mapping": {
            "ACTION": "MEAN_STD",
            "STATE": "MEAN_STD",
            "VISUAL": "IDENTITY"
        },
        "gradient_checkpointing": true,
        "compile_model": true,
        "compile_mode": "max-autotune",
        "optimizer_lr": 2.5e-05,
        "optimizer_betas": [
            0.9,
            0.95
        ],
        "optimizer_eps": 1e-08,
        "optimizer_weight_decay": 0.01,
        "optimizer_grad_clip_norm": 1.0,
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 6000,
        "scheduler_decay_lr": 2.5e-06
    },
```
先是三个