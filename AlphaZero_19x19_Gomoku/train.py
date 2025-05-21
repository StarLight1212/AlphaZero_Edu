import argparse
import logging
import os
import sys
import torch
import multiprocessing as mp

from src.core.game import GomokuGame
from src.core.neural_network import NNetWrapper
from src.core.self_play import SelfPlay
from src.utils.config_utils import load_config, print_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode='a')
    ]
)

log = logging.getLogger(__name__)


def main():
    """训练模型的主函数"""
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn')
        log.info("设置多进程启动方法为: spawn")
    except RuntimeError:
        log.info("多进程启动方法已设置，继续使用现有方法")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练AlphaZero五子棋AI')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config
    config = load_config(config_path)
    
    # 打印配置
    print_config(config)
    
    # 创建游戏实例
    game = GomokuGame(board_size=config.board_size, win_length=config.win_length)
    
    # 创建神经网络
    nnet = NNetWrapper(game, config)
    
    # 创建自我对弈实例
    self_play = SelfPlay(game, nnet, config)
    
    # 开始训练
    try:
        self_play.learn()
    except KeyboardInterrupt:
        log.info("训练被用户中断")
        
        # 保存当前模型
        save_path = os.path.join(config.checkpoint_dir, "interrupted.pt")
        nnet.save_checkpoint(folder=config.checkpoint_dir, filename="interrupted.pt")
        log.info(f"已保存中断时的模型到 {save_path}")
        
    log.info("训练完成")
    

if __name__ == "__main__":
    main() 