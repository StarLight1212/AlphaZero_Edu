import logging
import math
import numpy as np

log = logging.getLogger(__name__)


class MCTS:
    """
    蒙特卡洛树搜索算法实现
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        # 保存节点统计信息
        self.Qsa = {}  # Q值 Q(s,a): 状态s下执行动作a的期望价值
        self.Nsa = {}  # 访问次数 N(s,a): 状态s下执行动作a的访问次数
        self.Ns = {}  # 状态访问次数 N(s): 状态s的访问次数
        self.Ps = {}  # 先验概率 P(s): 状态s的动作概率（由神经网络预测）

        # 游戏相关
        self.Es = {}  # 终止状态 E(s): 状态s是否为终止状态
        self.Vs = {}  # 有效动作 V(s): 状态s的有效动作

        # Dirichlet噪声参数
        self.dirichlet_alpha = args.dirichlet_alpha
        self.dirichlet_epsilon = args.dirichlet_epsilon

    def get_action_prob(self, canonicalBoard, temp=1, add_dirichlet_noise=False):
        """
        执行MCTS算法并返回动作概率

        Args:
            canonicalBoard: 标准形式的棋盘状态（可能是Board对象或numpy数组）
            temp: 温度参数，控制探索程度
                temp=1: 高探索（按访问次数比例选择）
                temp→0: 低探索（选择访问次数最多的动作）
            add_dirichlet_noise: 是否添加Dirichlet噪声（自我对弈时添加）

        Returns:
            probs: 策略向量，表示每个动作的概率
        """
        # 运行MCTS模拟
        for _ in range(self.args.num_sims):
            self.search(canonicalBoard, add_dirichlet_noise)

        # 提取board_state，用于字符串表示
        if hasattr(canonicalBoard, 'pieces'):
            # 如果是Board对象，使用pieces属性
            board_state = canonicalBoard.pieces
        else:
            # 如果是numpy数组，直接使用
            board_state = canonicalBoard
            
        # 获取当前状态的字符串表示
        s = self.game.string_representation(board_state)

        # 计算每个动作的访问次数
        counts = [
            self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())
        ]
        total_counts = sum(counts)

        # 根据温度参数计算动作概率
        if temp == 0:  # 贪婪选择
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)  # 随机选择一个最优动作
            probs = np.zeros(len(counts))
            probs[bestA] = 1
            return probs

        # 如果总访问次数为0，则返回均匀分布
        if total_counts == 0:
            valids = self.game.get_valid_moves(board_state)
            probs = valids / np.sum(valids)
            return np.array(probs)

        # 使用温度参数调整访问次数并归一化
        eps = 1e-10  # 小值防止除零
        counts_temp = [(c + eps) ** (1.0 / temp) for c in counts]
        counts_sum = sum(counts_temp)
        probs = [x / counts_sum for x in counts_temp]

        # 确保概率和为1（处理浮点误差）
        probs = np.array(probs)

        return probs

    def search(self, canonicalBoard, add_dirichlet_noise=False):
        """
        MCTS搜索算法的一次迭代

        Args:
            canonicalBoard: 标准形式的棋盘状态（可能是Board对象或numpy数组）
            add_dirichlet_noise: 是否添加Dirichlet噪声

        Returns:
            v: 当前状态的价值
        """
        # 获取状态的字符串表示
        if hasattr(canonicalBoard, 'pieces'):
            # 如果是Board对象，使用pieces属性
            board_state = canonicalBoard.pieces
        else:
            # 如果是numpy数组，直接使用
            board_state = canonicalBoard
            
        s = self.game.string_representation(board_state)

        # 检查是否是终止状态
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(board_state, 1)
        if self.Es[s] is not None:
            # 终止状态返回游戏结果
            return self.Es[s]

        # 叶子节点（未访问过的状态）
        if s not in self.Ps:
            # 使用神经网络预测策略和价值 - 传递带有历史的棋盘对象或创建临时对象
            if hasattr(canonicalBoard, 'pieces'):
                # 如果已经是带有历史的Board对象，直接使用
                self.Ps[s], v = self.nnet.predict(canonicalBoard)
            else:
                # 如果是numpy数组，创建一个临时Board对象(没有历史)
                temp_board = self.game.game_to_board(canonicalBoard)
                self.Ps[s], v = self.nnet.predict(temp_board)

            # 添加Dirichlet噪声以增加探索
            if add_dirichlet_noise:
                valids = self.game.get_valid_moves(board_state)
                valid_indices = np.where(valids == 1)[0]

                # 只对有效动作添加噪声
                if len(valid_indices) > 0:  # 确保有有效动作
                    noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_indices))

                    # 混合噪声和先验概率
                    for i, idx in enumerate(valid_indices):
                        self.Ps[s][idx] = self.Ps[s][idx] * (1 - self.dirichlet_epsilon) + noise[
                            i] * self.dirichlet_epsilon

            # 屏蔽无效动作
            valids = self.game.get_valid_moves(board_state)
            self.Ps[s] = self.Ps[s] * valids

            # 重新归一化概率
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                # 如果所有有效动作的概率都为0，则均匀分布
                log.warning("所有有效动作的概率都为0，使用均匀分布")
                self.Ps[s] = valids / np.sum(valids)

            # 保存有效动作掩码
            self.Vs[s] = valids
            # 初始化访问次数
            self.Ns[s] = 0

            return v

        # 获取有效动作掩码
        valids = self.Vs[s]

        # 选择具有最高UCB分数的动作
        cur_best = -float("inf")
        best_act = -1
        best_acts = []  # 跟踪具有相同最高分数的动作

        # 遍历所有动作
        for a in range(self.game.get_action_size()):
            if valids[a]:
                # 计算UCB分数
                # Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

                # 检查是否有先前的Q值和访问次数
                q_value = self.Qsa.get((s, a), 0)
                n_visits = self.Nsa.get((s, a), 0)

                # 计算UCB分数
                if self.Ns[s] > 0:
                    u = q_value + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + n_visits)
                else:
                    # 如果状态未被访问过，则使用先验概率
                    u = self.args.cpuct * self.Ps[s][a]

                # 更新最佳动作
                if u > cur_best + 1e-8:  # 添加小的阈值避免浮点数比较问题
                    cur_best = u
                    best_acts = [a]
                    best_act = a
                elif abs(u - cur_best) < 1e-8:  # 如果分数差异很小
                    best_acts.append(a)

        # 执行最佳动作
        if len(best_acts) > 1:
            # 如果有多个同样好的动作，选择其中最可能的那个（根据先验概率）
            prior_probs = [self.Ps[s][a] for a in best_acts]
            best_act = best_acts[np.argmax(prior_probs)]
        else:
            best_act = best_acts[0]

        # 获取下一个状态
        if hasattr(canonicalBoard, 'pieces'):
            # 如果是Board对象，使用复制历史的方式
            next_board = canonicalBoard.copy()
            move = self.game.get_coord_from_action(best_act)
            next_board.execute_move(move, 1)
            next_s = next_board.pieces
            next_s = self.game.get_canonical_form(next_s, -1)
            
            # 创建下一个状态的Board对象
            next_board_obj = self.game.game_to_board(next_s)
            # 复制并翻转历史记录
            next_board_obj.history = [np.copy(hist) * -1 for hist in next_board.history]
            
            # 递归搜索下一个状态
            v = -self.search(next_board_obj)
        else:
            # 如果是numpy数组，使用原始方法
            next_s, next_player = self.game.get_next_state(board_state, 1, best_act)
            next_s = self.game.get_canonical_form(next_s, next_player)
            # 递归搜索下一个状态
            v = -self.search(next_s)

        # 更新Q值和访问次数
        if (s, best_act) in self.Qsa:
            # 增量更新Q值
            self.Qsa[(s, best_act)] = (self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v) / (
                        self.Nsa[(s, best_act)] + 1)
            # 增加访问次数
            self.Nsa[(s, best_act)] += 1
        else:
            # 初始化Q值和访问次数
            self.Qsa[(s, best_act)] = v
            self.Nsa[(s, best_act)] = 1

        # 增加状态访问次数
        self.Ns[s] += 1

        return v

    def reset(self):
        """重置MCTS统计信息"""
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}