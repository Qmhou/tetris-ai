好的，这是一个基于我们之前设计的项目结构的 `README.md` 文件内容。

```markdown
# AI驱动的俄罗斯方块游戏 (AI-Powered Tetris)

## 项目概述

本项目旨在创建一个功能丰富的俄罗斯方块游戏，包括手动操作模式、AI游戏模式以及一个可继续训练的AI训练模式。AI部分采用深度强化学习（DQN - Deep Q-Network）方法，通过神经网络评估不同落子选择的价值，从而实现智能游戏。游戏界面和逻辑基于Pygame，神经网络部分使用PyTorch。

## 主要特性

* **多种游戏模式**:
    * **手动模式 (Manual Mode)**: 玩家通过键盘手动控制方块。
    * **AI模式 (AI Mode)**: AI根据训练好的模型自动进行游戏，展示其游戏水平。
    * **训练模式 (Train Mode)**: AI进行学习训练，支持从断点继续训练，并可配置训练参数。
* **标准俄罗斯方块机制**:
    * 7-bag方块生成机制，确保方块分布的公平性。
    * 超级旋转系统 (Super Rotation System - SRS)，包括墙壁踢和地面踢。
    * 得分、消行、下一块预览、暂存块（Hold）等标准功能。
* **深度强化学习AI**:
    * 使用DQN算法，神经网络评估特定盘面状态的价值。
    * AI通过枚举当前方块所有可能的最终落点，并选择价值最高的方案。
    * 明确的状态特征工程，用于描述盘面状态：高度、空洞、广义井、平滑度。
    * 可定制的奖励函数，用于指导AI学习。
* **可配置与可扩展**:
    * 通过 `config.yaml` 文件配置游戏参数、神经网络结构和训练参数。
    * 模块化代码设计，易于理解、修改和扩展。

## 项目结构树

```
AI-Tetris/
├── weights/                  # 存放训练好的模型权重
│   └── (例如 dqn_tetris_episode_1000.pth)
├── screenshots/              # 存放训练过程中评估模式的游戏截图
│   └── (例如 eval_episode_1000_final.png)
├── logs/                     # 存放训练日志
│   └── (例如 training_log_YYYYMMDD_HHMMSS.csv)
├── config.yaml               # 配置文件 (游戏参数, AI参数, 训练参数)
├── main.py                   # 主程序入口，模式选择
├── train.py                  # AI训练脚本
├── tetris_game.py            # Tetris游戏核心逻辑模块
├── dqn_agent.py              # DQN Agent及神经网络模块
├── tetrominoes.py            # 定义方块形状、颜色及Piece类
├── srs_data.py               # SRS旋转的踢墙数据
├── operation_module.py       # (概念上)操作模块，实际逻辑可能整合在tetris_game.py
├── README.md                 # 本文件
└── requirements.txt          # (建议添加)项目依赖
```

## 技术栈

* **游戏框架**: Pygame
* **神经网络**: PyTorch
* **配置**: PyYAML
* **数值计算**: NumPy
* **核心语言**: Python 3.x

## 安装与环境设置

1.  **克隆仓库** (如果代码在git仓库中):
    ```bash
    git clone <your-repository-url>
    cd AI-Tetris
    ```

2.  **创建虚拟环境** (推荐):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  **安装依赖**:
    创建一个 `requirements.txt` 文件，内容如下:
    ```txt
    pygame
    torch
    numpy
    PyYAML
    ```
    然后运行:
    ```bash
    pip install -r requirements.txt
    ```
    *注意: PyTorch的安装可能因您的操作系统和CUDA版本而异。请参考 [PyTorch官方网站](https://pytorch.org/get-started/locally/) 获取特定安装命令。*

## 使用说明

通过 `main.py` 脚本启动游戏，并指定运行模式。

1.  **手动模式**:
    ```bash
    python main.py --mode manual
    ```
    * **操作按键**:
        * 左箭头: 左移
        * 右箭头: 右移
        * 下箭头: 软降
        * 上箭头: 顺时针旋转
        * `Z` 键: 逆时针旋转 (可选实现)
        * 空格键: 硬降
        * `C` 键: 暂存/交换方块
        * `ESC` 键: 退出游戏
        * `R` 键: (游戏结束后) 重新开始

2.  **AI游戏模式**:
    需要一个预训练的模型权重文件。脚本会尝试加载最新的模型（如果未指定）。
    ```bash
    python main.py --mode ai --model_path weights/your_trained_model.pth
    ```
    或者让脚本自动查找最新模型：
    ```bash
    python main.py --mode ai
    ```
    * AI将自动玩游戏。按 `ESC` 键退出，游戏结束后按 `R` 键重新开始。

3.  **AI训练模式**:
    ```bash
    python main.py --mode train
    ```
    * 训练过程将在后台静默运行（默认无图形界面，除非在`train.py`中为评估阶段开启）。
    * 模型权重将定期保存在 `weights/` 目录下。
    * 训练日志将保存在 `logs/` 目录下。
    * 评估阶段的游戏截图将保存在 `screenshots/` 目录下。
    * **继续训练**: `train.py` 脚本中可以添加逻辑来加载 `--model_path` 指定的权重文件，以继续之前的训练。

## 配置文件 (`config.yaml`)

`config.yaml` 文件包含了游戏、AI代理和训练过程的各项参数，例如：

* 游戏板尺寸、方块大小
* 神经网络结构 (隐藏层大小)
* 学习率、折扣因子 ($\gamma$)、$\epsilon$-greedy策略参数
* 经验回放缓冲区大小、批处理大小
* 模型保存频率、评估频率
* 奖励函数中的各启发式权重因子

可以根据需要调整这些参数以优化AI性能或改变游戏行为。

## AI状态特征与奖励机制

### 状态特征 (State Features)

传递给神经网络的特征向量，用于描述一个潜在落子动作完成后的盘面状态：

1.  **高度 (Height)**: `y_max`，盘面最高填充列的高度。
2.  **空洞 (Holes)**: `num_covered_holes`，上方有方块覆盖的空格子数量。
3.  **广义井 (Generalized Wells)**: `num_generalized_wells`，旨在惩罚所有最高点以下的未填充空间。
4.  **平滑度 (Smoothness/Bumpiness)**: `bumpiness_value`，相邻列块高度差的绝对值之和。

### 奖励机制 (Reward Function)

当一个方块固定后，根据盘面变化和启发式规则计算奖励：

`reward = (score - old_score) + height_penalty + hole_penalty + bumpiness_penalty + lines_reward + game_over_penalty + piece_drop_reward`

其中各项惩罚/奖励因子可在 `config.yaml` 中调整。

## 未来可能的改进

* 更复杂的神经网络结构 (如卷积层直接处理盘面图像)。
* 更高级的强化学习算法 (如 Rainbow DQN, PPO)。
* 实现更精细的 `OperationModule`，让AI输出底层操作序列而非直接选择最终状态。
* 在线排行榜或对战模式。
* 更完善的用户界面和用户体验。

## 贡献

欢迎通过提交 Pull Requests 或 Issues 来为项目做出贡献。

---

希望这个README对您有所帮助！
```