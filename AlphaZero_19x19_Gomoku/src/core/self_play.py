import logging
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
from collections import deque
import random
import multiprocessing as mp
from functools import partial
from .game import GomokuGame
from .mcts import MCTS
from .neural_network import NNetWrapper

log = logging.getLogger(__name__)


class SelfPlay:
    """自我对弈类，用于生成训练数据"""

    def __init__(self, game, nnet, args):
        """
        初始化自我对弈

        Args:
            game: 游戏规则实例
            nnet: 神经网络实例
            args: 参数
        """
        self.game = game
        self.nnet = nnet
        self.args = args

        # 初始化蒙特卡洛树搜索
        self.mcts = MCTS(self.game, self.nnet, self.args)

        # 训练样本队列
        self.train_examples_history = []

        # 存储最佳模型
        # self.best_nnet = NNetWrapper(game, args)

        # 设置进程数
        self.num_processes = getattr(args, 'num_processes', 8)  # 如果配置中没有设置进程数，默认为8
        log.info(f"设置并行进程数为: {self.num_processes}")

    @staticmethod
    def _worker_func_wrapper(i, game, nnet, args):
        """静态包装函数，用于替代lambda函数，解决pickle问题"""
        # 忽略索引参数i，直接调用真正的worker函数
        return SelfPlay._execute_episode_worker(game, nnet, args)

    @staticmethod
    def _execute_episode_worker(game, nnet, args):
        """
        静态工作函数，用于并行执行自我对弈

        Args:
            game: 游戏规则实例
            nnet: 神经网络实例
            args: 参数

        Returns:
            train_examples: 训练样本列表
        """
        # 记录训练样本 [(board, current_player, pi, reward)]
        train_examples = []

        # 初始化棋盘 - 使用Board对象而不是numpy数组
        board = game.game_to_board(game.get_init_board())
        current_player = 1

        # 执行游戏直到结束
        step = 0

        # 初始化MCTS
        mcts = MCTS(game, nnet, args)

        # 重置MCTS树
        mcts.reset()

        while True:
            step += 1

            # 获取标准形式的棋盘（从当前玩家的视角）
            canonical_board = game.get_canonical_form(board.pieces, current_player)

            # 温度参数，控制探索
            # 前 temp_threshold 步使用温度=1，鼓励探索
            # 之后使用温度→0，更确定的选择
            temp = 1 if step <= args.temp_threshold else 0.5

            # 创建一个标准形式的Board对象，用于MCTS
            canonical_board_obj = game.game_to_board(canonical_board)
            # 复制历史记录到标准形式的Board对象
            if current_player == 1:
                canonical_board_obj.history = [np.copy(hist) for hist in board.history]
            else:
                # 如果是第二个玩家，需要翻转历史记录的颜色
                canonical_board_obj.history = [np.copy(hist) * -1 for hist in board.history]

            # 执行MCTS，获取策略
            pi = mcts.get_action_prob(canonical_board_obj, temp=temp, add_dirichlet_noise=True)

            # 获取有效动作
            valid_moves = game.get_valid_moves(canonical_board)
            valid_indices = np.where(valid_moves == 1)[0]  # 有效动作的索引

            # 确保策略只对有效动作有非零概率（可选，用于调试）
            pi_masked = pi * valid_moves
            if np.sum(pi_masked) > 0:
                pass
            else:
                pi_masked = valid_moves / np.sum(valid_moves)  # 均匀分布作为fallback

            # 记录训练样本 - 使用带有历史的Board对象
            # 对于对称变换，我们需要特殊处理历史记录
            board_with_history = canonical_board_obj

            # 获取基本的对称变换(不包含历史)
            sym_basic = game.get_symmetries(canonical_board, pi_masked)

            # 为每个基本对称变换重新添加历史(经过相同的变换)
            for b_basic, p in sym_basic:
                # 创建一个临时Board对象
                temp_board = game.game_to_board(b_basic)

                # 对历史记录应用相同的对称变换
                for idx, hist_board in enumerate(canonical_board_obj.history):
                    # 找到与当前board_basic相同变换的历史board
                    hist_syms = game.get_symmetries(hist_board, np.ones(game.get_action_size()))
                    for h_idx, (h_sym, _) in enumerate(hist_syms):
                        if np.array_equal(b_basic, game.get_canonical_form(canonical_board, current_player)):
                            temp_board.history.append(h_sym)
                            break

                # 添加到训练样本
                train_examples.append([temp_board, current_player, p, None])

            # 选择动作
            if step <= args.temp_threshold:
                # 探索阶段，从有效动作中按概率采样
                action = np.random.choice(valid_indices, p=pi_masked[valid_indices])
            else:
                # 利用阶段
                if np.random.random() < 0.35:    # 0.8
                    # 选择最优动作（从有效动作中）
                    valid_pi = pi_masked[valid_indices]
                    action = valid_indices[np.argmax(valid_pi)]
                else:
                    # 保留一定的探索性，从有效动作中采样
                    action = np.random.choice(valid_indices, p=pi_masked[valid_indices])

            # 执行动作，获取下一个状态
            move = game.get_coord_from_action(action)
            board.execute_move(move, current_player)
            current_player = -current_player

            # 检查游戏是否结束
            game_result = game.get_game_ended(board.pieces, current_player)

            if game_result is not None:
                # 游戏结束
                result_rewards = []

                for hist_board, hist_player, hist_pi, _ in train_examples:
                    if game_result == 0:  # 平局
                        reward = 0
                    else:
                        # 计算相对于hist_player的奖励值
                        if hist_player == current_player:
                            reward = game_result
                        else:
                            reward = -game_result
                    result_rewards.append([hist_board, hist_player, hist_pi, reward])

                return result_rewards

    def execute_episode(self):
        """
        执行一次自我对弈，生成训练样本

        Returns:
            train_examples: 训练样本列表
        """
        return self._execute_episode_worker(self.game, self.nnet, self.args)

    def learn(self):
        """
        执行AlphaZero学习算法主循环
        """
        log.info("开始训练")

        # 创建训练数据目录
        os.makedirs(self.args.data_dir, exist_ok=True)

        # 创建模型目录
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

        # 如果需要加载模型
        if self.args.load_model:
            log.info(f"加载模型: {self.args.load_folder_file}")
            model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
            if os.path.exists(model_file):
                self.nnet.load_checkpoint(self.args.load_folder_file[0], self.args.load_folder_file[1])
            else:
                log.warning(f"模型文件不存在: {model_file}")

        # 如果存在之前的训练样本，加载它们
        examples_file = os.path.join(self.args.data_dir, "train_examples.pkl")
        if os.path.exists(examples_file):
            log.info("加载之前的训练样本")
            with open(examples_file, "rb") as f:
                self.train_examples_history = pickle.load(f)
            log.info(f"加载了 {len(self.train_examples_history)} 组训练样本")

        # 主训练循环
        for iteration in range(1, self.args.num_iterations + 1):
            log.info(f"开始迭代 {iteration}/{self.args.num_iterations}")

            # 收集自我对弈的训练样本
            iteration_train_examples = []

            # 使用进程池并行执行自我对弈
            log.info(f"使用 {self.num_processes} 个进程进行并行自我对弈")
            with mp.Pool(processes=self.num_processes) as pool:
                # 准备要并行执行的任务
                num_episodes = self.args.num_episodes

                # 使用包装函数创建部分函数
                worker_func = partial(self._worker_func_wrapper, game=self.game, nnet=self.nnet, args=self.args)

                # 使用进程池的imap方法并行执行自我对弈
                results = list(tqdm(
                    pool.imap(worker_func, range(num_episodes)),
                    total=num_episodes,
                    desc="并行自我对弈"
                ))

                # 收集所有结果
                for episode_examples in results:
                    iteration_train_examples.extend(episode_examples)

            # 添加到训练样本历史
            self.train_examples_history.append(iteration_train_examples)

            # 如果历史样本超过了保留的迭代数，移除最早的样本
            if len(self.train_examples_history) > self.args.num_iters_history:
                log.info(f"移除最早的训练样本，保持历史长度为 {self.args.num_iters_history}")
                self.train_examples_history.pop(0)

            # 保存训练样本
            log.info("保存训练样本")
            with open(examples_file, "wb") as f:
                pickle.dump(self.train_examples_history, f)

            # 从历史样本中抽取训练样本
            train_examples = []
            for examples in self.train_examples_history:
                train_examples.extend(examples)

            # 打乱训练样本
            random.shuffle(train_examples)

            # 如果训练样本超过最大队列长度，截断
            if len(train_examples) > self.args.max_queue_length:
                log.info(f"截断训练样本，保持长度为 {self.args.max_queue_length}")
                train_examples = train_examples[:self.args.max_queue_length]

            # 重新格式化训练样本 [(board, pi, v)]
            # 确保board是Board对象，并保留历史记录
            train_data = []
            for x in train_examples:
                board_obj = x[0]  # 已经是Board对象
                pi = x[2]
                v = x[3]
                train_data.append((board_obj, pi, v))

            # 训练神经网络
            log.info(f"使用 {len(train_data)} 个样本训练神经网络")
            total_loss = self.nnet.train(train_data)
            log.info(f"训练完成，总损失: {total_loss:.4f}")

            # 评估新模型与最佳模型的性能
            # log.info("与最佳模型对比")
            # arena_results = self.arena_compare(self.best_nnet, self.nnet)
            # self.evaluate_against_random(self.nnet, 20)

            # 如果新模型更好，则更新最佳模型
            # win_rate = arena_results[1] / arena_results[2]
            # log.info(f"新模型的胜率: {win_rate:.4f}")

            # if win_rate >= self.args.update_threshold:
            #     log.info("新模型更好，更新最佳模型")
            #     # 复制网络权重
            #     self.best_nnet.nnet.load_state_dict(self.nnet.nnet.state_dict())
            #     # 保存最佳模型
            #     self.best_nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename="best.pt")
            #     # 保存迭代模型
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename=f"iteration_{iteration}.pt")
            # else:
            #     log.info("新模型没有显著改进，保持最佳模型不变")
            #     # 加载最佳模型的权重
            #     self.nnet.nnet.load_state_dict(self.best_nnet.nnet.state_dict())

            # 保存当前迭代的模型 (覆盖 'latest' 并保存迭代版本)
            log.info(f"保存迭代 {iteration} 的模型")
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename='best.pt')
            self.nnet.save_checkpoint(folder=self.args.checkpoint_dir, filename=f"iteration_{iteration}.pt")

            # 记录训练进度（可选，使用wandb）
            if hasattr(self.args, 'use_wandb') and self.args.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "iteration": iteration,
                        "train_loss": total_loss,
                        "num_train_samples": len(train_data),
                        # Add other relevant metrics if available
                    })
                except ImportError:
                    log.warning("wandb 未安装，跳过日志记录。")
                except Exception as e:
                    log.error(f"记录到 wandb 失败: {e}")

                # 训练完成
            log.info("所有训练迭代完成")

    def arena_compare(self, net1, net2, num_games=40):
        """
        比较两个网络的性能

        Args:
            net1: 第一个网络（最佳网络） best_net
            net2: 第二个网络（新训练的网络） n_net
            num_games: 比赛次数

        Returns:
            [net1_wins, net2_wins, total_games] - 匹配learn方法中期望的格式
        """
        # 确保使用较小的模拟次数加快评估速度
        temp_num_sims = self.args.num_sims
        self.args.num_sims = min(self.args.num_sims, 200)  # 降低模拟次数加快评估

        log.info(f"使用 {self.num_processes} 个进程进行并行竞技场评估")

        # 定义一个内部函数来处理单场比赛
        def play_game(game_idx):
            # 创建本地MCTS实例
            local_mcts1 = MCTS(self.game, net1, self.args)
            local_mcts2 = MCTS(self.game, net2, self.args)

            # 初始化棋盘
            board = self.game.get_init_board()
            current_player = 1

            # 确定本局比赛中哪个网络扮演哪个角色
            if game_idx < num_games / 2:
                # 前半部分比赛：player 1 = net1, player -1 = net2
                player_to_net = {1: net1, -1: net2}
                player_to_mcts = {1: local_mcts1, -1: local_mcts2}
            else:
                # 后半部分比赛：player 1 = net2, player -1 = net1
                player_to_net = {1: net2, -1: net1}
                player_to_mcts = {1: local_mcts2, -1: local_mcts1}

            # 重置MCTS
            local_mcts1.reset()
            local_mcts2.reset()

            # 执行游戏直到结束
            while True:
                # 获取标准形式的棋盘
                canonical_board = self.game.get_canonical_form(board, current_player)

                # 确定当前玩家使用的MCTS
                current_mcts = player_to_mcts[current_player]

                # 使用MCTS获取策略
                # 竞技场比赛使用低温度，确定性更强
                pi = current_mcts.get_action_prob(canonical_board, temp=0.1, add_dirichlet_noise=False)

                # 选择动作
                action = np.argmax(pi)  # 选择概率最高的动作

                # 执行动作
                board, current_player = self.game.get_next_state(board, current_player, action)

                # 检查游戏是否结束
                game_result = self.game.get_game_ended(board, current_player)

                if game_result is not None:
                    # 游戏结束
                    if game_result == 0:  # 平局
                        return [0, 0, 1]  # [net1_wins, net2_wins, draws]

                    # 确定谁是赢家（1表示当前玩家赢，-1表示对手赢）
                    winner = current_player if game_result == 1 else -current_player

                    # 根据赢家使用的网络更新胜场
                    if player_to_net[winner] == net1:
                        return [1, 0, 0]  # [net1_wins, net2_wins, draws]
                    else:
                        return [0, 1, 0]  # [net1_wins, net2_wins, draws]

        # 直接使用单线程执行对战，避免multiprocessing错误
        results = []
        for i in tqdm(range(num_games), desc="竞技场对比"):
            result = play_game(i)
            results.append(result)

        # 汇总结果
        net1_wins = sum(r[0] for r in results)
        net2_wins = sum(r[1] for r in results)
        draws = sum(r[2] for r in results)

        # 恢复原始模拟次数
        self.args.num_sims = temp_num_sims

        # 打印详细信息
        log.info(f"评估结果 - 最佳模型胜: {net1_wins}, 新模型胜: {net2_wins}, 平局: {draws}")

        # 返回结果 - 保持与learn方法中期望的格式一致
        return [net1_wins, net2_wins, num_games]

    def evaluate_against_random(self, net, num_games=20):
        """
        将训练好的网络与随机策略进行对比评估

        Args:
            net: 要评估的网络
            num_games: 比赛次数

        Returns:
            [net_wins, random_wins, draws, win_rate]
        """
        log.info(f"评估模型与随机策略的对战")

        # 定义一个内部函数来处理单场比赛
        def play_random_game(game_idx):
            # 创建本地MCTS实例
            local_mcts = MCTS(self.game, net, self.args)

            # 初始化棋盘
            board = self.game.get_init_board()
            current_player = 1

            # 交替先手
            # 网络是player 1还是player -1
            net_player = 1 if game_idx % 2 == 0 else -1

            # 重置MCTS
            local_mcts.reset()

            # 执行游戏直到结束
            while True:
                # 获取标准形式的棋盘
                canonical_board = self.game.get_canonical_form(board, current_player)

                if current_player == net_player:
                    # 网络的回合
                    pi = local_mcts.get_action_prob(canonical_board, temp=0.1, add_dirichlet_noise=False)
                    action = np.argmax(pi)
                else:
                    # 随机策略的回合
                    valid_moves = self.game.get_valid_moves(canonical_board)
                    valid_indices = np.where(valid_moves == 1)[0]
                    action = np.random.choice(valid_indices)

                # 执行动作
                board, current_player = self.game.get_next_state(board, current_player, action)

                # 检查游戏是否结束
                game_result = self.game.get_game_ended(board, current_player)

                if game_result is not None:
                    # 游戏结束
                    if game_result == 0:  # 平局
                        return [0, 0, 1]  # [net_wins, random_wins, draws]
                    elif (game_result == 1 and current_player == net_player) or \
                            (game_result == -1 and current_player != net_player):
                        # 网络获胜
                        return [1, 0, 0]  # [net_wins, random_wins, draws]
                    else:
                        # 随机策略获胜
                        return [0, 1, 0]  # [net_wins, random_wins, draws]

        # 直接使用单线程执行对战，避免multiprocessing错误
        results = []
        for i in tqdm(range(num_games), desc="随机策略对比"):
            result = play_random_game(i)
            results.append(result)

        # 汇总结果
        net_wins = sum(r[0] for r in results)
        random_wins = sum(r[1] for r in results)
        draws = sum(r[2] for r in results)

        # 计算胜率
        win_rate = net_wins / num_games

        # 打印详细信息
        log.info(f"随机策略评估 - 模型胜: {net_wins}, 随机胜: {random_wins}, 平局: {draws}, 胜率: {win_rate:.4f}")

        return [net_wins, random_wins, draws, win_rate]