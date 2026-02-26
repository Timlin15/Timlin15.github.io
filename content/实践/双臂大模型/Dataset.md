参考：![15c65b20fac07a1fc5348948ed0fe2ca.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/test/15c65b20fac07a1fc5348948ed0fe2ca.png)
1. behavior-1k
	[Dataset - BEHAVIOR](https://behavior.stanford.edu/challenge/dataset.html)是仿真数据。使用Lerobot数据格式包含语言标注，数据，元数据，视频`data(.parquet)`，`videos(.mp4*9)`,`mata,annotation(.json)`。大小大约1.5T，还有700G的[behavior-1k/2025-challenge-rawdata at main](https://huggingface.co/datasets/behavior-1k/2025-challenge-rawdata/tree/main)（.hdf5）
2. [agibot-world/AgiBotWorldChallenge-2025 · Datasets at Hugging Face](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025)
	3T的同时包含真机，仿真，onsite和世界模型的数据，格式为`video(.mp4)`,`params(.json)`和`state(.h5)`
3. 越疆和A1貌似是私有数据，网上只搜得到硬件
4. InternData有仿真数据和真实数据(上传中)。大小是1.89T的数据。结构为`data(.parquet)`，`videos(.mp4)`,`mata,annotation(.json)`。任务分为基础任务，长程任务，抓取-放置任务和articulation task。
5. https://huggingface.co/RoboCOIN/datasets 大小为2.5T，结构为`data(.parquet)`，`videos(.mp4*4)`,`mata,annotation(.json)`
6. [robotics-diffusion-transformer/rdt-ft-data · Datasets at Hugging Face](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data)，700G的.h5文件
7. lerobot的huggingface中还有一些如aloha的数据集，但是较小，大概是数十G的量级