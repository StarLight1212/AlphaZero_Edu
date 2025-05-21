import numpy as np
import logging

log = logging.getLogger(__name__)


class Board:
    """
    五子棋棋盘类
    棋盘数据:
    1=白棋, -1=黑棋, 0=空
    """

    def __init__(self, n=15, win_length=5):
        self.n = n
        self.win_length = win_length
        # 创建空棋盘
        self.pieces = np.zeros((n, n), dtype=np.int8)
        self.last_move = None  # 记录最后一步
        self.move_count = 0  # 记录下棋数
        # 添加历史棋盘记录，最多保存20步
        self.history = []  # 存储历史棋盘状态列表

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self):
        """返回所有合法的走子位置"""
        return list(zip(*np.where(self.pieces == 0)))

    def has_legal_moves(self):
        """检查是否有合法的走子"""
        return np.any(self.pieces == 0)

    def execute_move(self, move, color):
        """在指定位置放置棋子"""
        x, y = move
        assert self[x, y] == 0, f"Position {move} is already occupied"
        # 添加当前状态到历史记录
        self.history.append(np.copy(self.pieces))
        # 如果历史记录超过20步，删除最早的记录
        if len(self.history) > 20:
            self.history.pop(0)
        # 执行当前动作
        self.pieces[x, y] = color
        self.last_move = move
        self.move_count += 1

    def get_history_planes(self):
        """获取历史棋盘平面，用于神经网络输入"""
        # 创建一个包含当前棋盘和历史的张量
        # 总共21个平面(当前+历史20步)
        history_planes = np.zeros((21, self.n, self.n), dtype=np.int8)
        
        # 设置当前棋盘状态作为第一个平面
        history_planes[0] = self.pieces
        
        # 填充历史棋盘状态
        for i, hist_board in enumerate(reversed(self.history)):
            if i < 20:  # 最多20个历史步骤
                history_planes[i+1] = hist_board
        
        return history_planes

    def is_win(self, color):
        """检查整个棋盘是否有获胜条件"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x, y] == color:
                    for dx, dy in directions:
                        count = 1
                        for step in range(1, self.win_length):
                            nx, ny = x + step * dx, y + step * dy
                            if not (0 <= nx < self.n and 0 <= ny < self.n and self.pieces[nx, ny] == color):
                                break
                            count += 1
                        if count >= self.win_length:
                            return True
        return False

    def get_winning_path(self, color):
        """返回获胜路径（用于GUI高亮显示）"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x, y] == color:
                    for dx, dy in directions:
                        path = [(x, y)]
                        for step in range(1, self.win_length):
                            nx, ny = x + step * dx, y + step * dy
                            if not (0 <= nx < self.n and 0 <= ny < self.n and self.pieces[nx, ny] == color):
                                break
                            path.append((nx, ny))
                        if len(path) >= self.win_length:
                            return path
        return []

    def copy(self):
        """创建棋盘的深拷贝"""
        new_board = Board(self.n, self.win_length)
        new_board.pieces = np.copy(self.pieces)
        new_board.last_move = self.last_move
        new_board.move_count = self.move_count
        # 复制历史记录
        new_board.history = [np.copy(board) for board in self.history]
        return new_board

    def __str__(self):
        """棋盘的字符串表示"""
        symbols = {0: '.', 1: 'O', -1: 'X'}
        board_str = ""
        for i in range(self.n):
            row = " ".join([symbols[self.pieces[i, j]] for j in range(self.n)])
            board_str += row + "\n"
        return board_str


class GomokuGame:
    """五子棋游戏规则实现"""

    def __init__(self, board_size=15, win_length=5):
        self.n = board_size
        self.win_length = win_length

    def get_init_board(self):
        """获取初始棋盘"""
        b = Board(self.n, self.win_length)
        return b.pieces

    def get_board_size(self):
        """获取棋盘尺寸"""
        return (self.n, self.n)

    def get_action_size(self):
        """获取动作空间大小"""
        return self.n * self.n

    def get_coord_from_action(self, action):
        """
        将动作索引转换为棋盘坐标
        Args:
            action: 动作索引（0到n^2-1）
            
        Returns:
            (row, col): 棋盘坐标
        """
        row = action // self.n
        col = action % self.n
        return (row, col)

    def get_action_from_coord(self, coord):
        """
        将棋盘坐标转换为动作索引
        Args:
            coord: 棋盘坐标 (row, col)
            
        Returns:
            action: 动作索引（0到n^2-1）
        """
        row, col = coord
        return row * self.n + col

    def get_next_state(self, board, player, action):
        """
        获取执行动作后的下一个状态
        Args:
            board: 当前棋盘状态
            player: 当前玩家（1或-1）
            action: 动作索引（0到n^2-1）

        Returns:
            next_board: 下一个棋盘状态
            next_player: 下一个玩家
        """
        b = Board(self.n, self.win_length)
        b.pieces = np.copy(board)
        # 将动作索引转换为坐标
        move = self.get_coord_from_action(action)
        b.execute_move(move, player)
        return b.pieces, -player

    def get_valid_moves(self, board):
        """获取有效走子的掩码"""
        b = Board(self.n, self.win_length)
        b.pieces = np.copy(board)
        valids = np.zeros(self.get_action_size(), dtype=np.int8)
        legal_moves = b.get_legal_moves()
        for x, y in legal_moves:
            valids[self.n * x + y] = 1
        return valids

    def get_game_ended(self, board, player):
        """
        检查游戏是否结束
        Returns:
            None: 游戏未结束
            1: player获胜
            -1: player失败
            0: 平局
        """
        b = Board(self.n, self.win_length)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if not b.has_legal_moves():
            return 0  # 平局
        return None  # 游戏未结束

    def get_canonical_form(self, board, player):
        """获取标准形式（从当前玩家视角）"""
        return player * board

    def get_board_with_history(self, board):
        """获取带有历史记录的棋盘状态"""
        b = Board(self.n, self.win_length)
        b.pieces = np.copy(board)
        # 尝试从历史记录加载棋盘历史(如果board是Board对象)
        if isinstance(board, np.ndarray):
            # 如果是numpy数组，则无法获取历史，返回只有当前状态的棋盘
            return b
        elif hasattr(board, 'history'):
            b.history = [np.copy(hist_board) for hist_board in board.history]
        return b

    def get_symmetries(self, board, pi):
        """
        获取状态的对称变化
        Args:
            board: 当前棋盘状态
            pi: 概率分布向量

        Returns:
            list: [(board, pi)] 对称变换后的棋盘状态和概率分布
        """
        assert len(pi) == self.n ** 2
        pi_board = np.reshape(pi, (self.n, self.n))
        symmetries = []

        # 旋转和翻转
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                symmetries.append((newB, newPi.ravel()))
        return symmetries

    def string_representation(self, board):
        """棋盘的字符串表示（用于MCTS的状态存储）"""
        return board.tobytes()

    def game_to_board(self, board_array):
        """从numpy数组创建Board对象"""
        board = Board(self.n, self.win_length)
        board.pieces = np.copy(board_array)

        # 尝试查找最后落子
        non_zero = np.where(board_array != 0)
        if len(non_zero[0]) > 0:
            # 简单启发式：最后落子可能是最右下方的棋子
            idx = np.argmax(non_zero[0] + non_zero[1])
            board.last_move = (non_zero[0][idx], non_zero[1][idx])
            board.move_count = len(non_zero[0])

        return board

    def get_winning_path(self, board, player):
        """从游戏级别调用Board的获胜路径"""
        b = Board(self.n, self.win_length)
        b.pieces = np.copy(board)
        return b.get_winning_path(player)

    @staticmethod
    def display(board):
        """打印棋盘（终端）"""
        n = board.shape[0]
        symbols = {0: '.', 1: 'O', -1: 'X'}

        # 打印列标
        print('  ', end='')
        for i in range(n):
            print(f'{i:2}', end='')
        print()

        # 打印行和棋盘内容
        for i in range(n):
            print(f'{i:2}', end='')
            for j in range(n):
                print(f'{symbols[board[i, j]]:2}', end='')
            print()