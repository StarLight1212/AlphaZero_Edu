import yaml
import logging
import os
import torch

log = logging.getLogger(__name__)


class DotDict(dict):
    """
    使字典支持点号访问
    例如 d = DotDict({'foo': 3})
    d.foo  # 返回 3
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        args: 配置参数
    """
    # 检查文件是否存在
    if not os.path.exists(config_path):
        log.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载YAML文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='gbk') as f:
            config = yaml.safe_load(f)
    
    # 创建参数字典
    args = DotDict({})
    
    # 训练参数
    args.epochs = config["training"]["epochs"]
    args.batch_size = config["training"]["batch_size"]
    args.num_iterations = config["training"]["num_iterations"]
    args.num_episodes = config["training"]["num_episodes"]
    args.max_queue_length = config["training"]["max_queue_length"]
    args.num_iters_history = config["training"]["num_iters_history"]
    args.update_threshold = config["training"]["update_threshold"]
    args.arena_compare = config["training"]["arena_compare"]
    args.temp_threshold = config["training"]["temp_threshold"]
    
    # 神经网络参数
    args.num_channels = config["network"]["num_channels"]
    args.dropout = config["network"]["dropout"]
    args.learning_rate = DotDict(config["network"]["learning_rate"])
    args.grad_clip = config["network"]["grad_clip"]
    args.optimizer = config["network"]["optimizer"]
    
    # MCTS参数
    args.num_sims = config["mcts"]["num_sims"]
    args.cpuct = config["mcts"]["cpuct"]
    args.dirichlet_alpha = config["mcts"]["dirichlet_alpha"]
    args.dirichlet_epsilon = config["mcts"]["dirichlet_epsilon"]
    
    # 游戏参数
    args.board_size = config["game"]["board_size"]
    args.win_length = config["game"]["win_length"]
    
    # 系统参数
    args.cuda = config["system"]["cuda"] and torch.cuda.is_available()
    if args.cuda:
        log.info("CUDA可用，使用GPU")
    else:
        log.info("CUDA不可用或未启用，使用CPU")
    
    args.checkpoint_dir = config["system"]["checkpoint_dir"]
    args.data_dir = config["system"]["data_dir"]
    args.load_model = config["system"]["load_model"]
    args.load_folder_file = config["system"]["load_folder_file"]
    args.num_workers = config["system"]["num_workers"]
    args.use_wandb = config["system"]["use_wandb"]
    
    # GUI参数
    args.window_width = config["gui"]["window_width"]
    args.window_height = config["gui"]["window_height"]
    args.cell_size = config["gui"]["cell_size"]
    args.margin = config["gui"]["margin"]
    args.bottom_margin = config["gui"]["bottom_margin"]
    args.fps = config["gui"]["fps"]
    
    return args


def print_config(args):
    """
    打印配置参数
    
    Args:
        args: 配置参数
    """
    log.info("配置参数:")
    
    # 打印训练参数
    log.info("训练参数:")
    log.info(f"  训练轮数: {args.epochs}")
    log.info(f"  批量大小: {args.batch_size}")
    log.info(f"  迭代次数: {args.num_iterations}")
    log.info(f"  每次迭代的自我对弈次数: {args.num_episodes}")
    log.info(f"  训练样本队列最大长度: {args.max_queue_length}")
    log.info(f"  保留的历史迭代数: {args.num_iters_history}")
    log.info(f"  新模型胜率阈值: {args.update_threshold}")
    log.info(f"  竞技场比赛次数: {args.arena_compare}")
    log.info(f"  温度阈值: {args.temp_threshold}")
    
    # 打印神经网络参数
    log.info("神经网络参数:")
    log.info(f"  通道数: {args.num_channels}")
    log.info(f"  Dropout率: {args.dropout}")
    log.info(f"  学习率范围: {args.learning_rate.min} - {args.learning_rate.max}")
    log.info(f"  梯度裁剪: {args.grad_clip}")
    log.info(f"  优化器: {args.optimizer}")
    
    # 打印MCTS参数
    log.info("MCTS参数:")
    log.info(f"  模拟次数: {args.num_sims}")
    log.info(f"  PUCT常数: {args.cpuct}")
    log.info(f"  Dirichlet噪声参数: {args.dirichlet_alpha}")
    log.info(f"  Dirichlet噪声权重: {args.dirichlet_epsilon}")
    
    # 打印游戏参数
    log.info("游戏参数:")
    log.info(f"  棋盘大小: {args.board_size}")
    log.info(f"  获胜所需的连续棋子数: {args.win_length}")
    
    # 打印系统参数
    log.info("系统参数:")
    log.info(f"  使用CUDA: {args.cuda}")
    log.info(f"  检查点目录: {args.checkpoint_dir}")
    log.info(f"  数据目录: {args.data_dir}")
    log.info(f"  加载模型: {args.load_model}")
    log.info(f"  加载模型路径: {args.load_folder_file}")
    log.info(f"  工作线程数: {args.num_workers}")
    log.info(f"  使用Weights & Biases: {args.use_wandb}")
    
    # 打印GUI参数
    log.info("GUI参数:")
    log.info(f"  窗口宽度: {args.window_width}")
    log.info(f"  窗口高度: {args.window_height}")
    log.info(f"  格子大小: {args.cell_size}")
    log.info(f"  边距: {args.margin}")
    log.info(f"  底部边距: {args.bottom_margin}")
    log.info(f"  帧率: {args.fps}") 