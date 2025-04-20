# AlphaGomoku

基于AlphaZero算法的五子棋AI，使用自我对弈、深度学习和蒙特卡洛树搜索来训练和玩五子棋。

## 功能特性

- 完整的AlphaZero算法实现
- 自我对弈生成训练数据
- 残差神经网络模型
- 蒙特卡洛树搜索（MCTS）
- 交互式GUI界面
- 可配置的训练和游戏参数

## 项目结构

```
AlphaGomoku/
├── config.yaml           # 配置文件
├── train.py              # 训练脚本
├── play.py               # 游戏脚本
├── README.md             # 说明文档
├── data/                 # 训练数据目录
├── models/               # 模型目录
└── src/                  # 源码目录
    ├── core/             # 核心模块
    │   ├── game.py       # 游戏规则
    │   ├── mcts.py       # 蒙特卡洛树搜索
    │   ├── neural_network.py # 神经网络
    │   └── self_play.py  # 自我对弈
    ├── gui/              # GUI模块
    │   └── gui.py        # 图形界面
    └── utils/            # 工具模块
        └── config_utils.py # 配置工具
```

## 环境要求

- Python 3.8+
- PyTorch 1.7+
- NumPy
- PyGame
- PyYAML
- tqdm

## 安装

1. 克隆仓库:

```bash
git clone https://github.com/yourusername/AlphaGomoku.git
cd AlphaGomoku
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python train.py --config config.yaml
```

训练过程会在`models/`目录下保存模型，在`data/`目录下保存训练数据。

### 与AI对战

```bash
python play.py --config config.yaml
```

#### 其他选项:

- `--model models/best.pt`: 指定要加载的模型文件
- `--human_first`: 设置人类玩家先手（默认AI先手）
- `--mcts_sims 400`: 调整MCTS模拟次数（影响AI强度和思考时间）

## 配置

可以在`config.yaml`文件中修改配置参数，包括：

- 棋盘大小
- 训练参数
- 神经网络参数
- MCTS参数
- GUI参数

## 如何玩游戏

1. 运行`python play.py`启动游戏
2. 在棋盘上点击放置棋子
3. 用"新游戏"按钮开始新游戏
4. 用"认输"按钮结束当前游戏

## 许可证

[MIT](LICENSE)

## 致谢

本项目基于DeepMind的AlphaZero论文和算法实现。感谢开源社区的贡献。 