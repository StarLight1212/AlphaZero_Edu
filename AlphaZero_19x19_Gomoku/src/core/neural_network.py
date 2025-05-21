import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels, dropout=0.3):
        super(ResBlock, self).__init__()
        # 为了保持维度不变，当 kernel_size=5 时，padding 应设置为 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=False) # padding 改为 2, 建议添加 bias=False 当使用 BN 时
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=False) # padding 改为 2, 建议添加 bias=False 当使用 BN 时
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # 推荐的顺序：Conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差连接
        out += residual
        out = F.relu(out) # ReLU 放在残差连接之后
        return out


class GomokuNNet(nn.Module):
    """五子棋神经网络"""

    def __init__(self, game, args):
        super(GomokuNNet, self).__init__()

        # 游戏参数
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args
        self.channels = args.num_channels

        # 历史平面数量(当前状态+20个历史状态)
        self.history_planes = 21

        # 网络结构
        # 输入层 - 使用 kernel=5, padding=1 会减小空间维度
        self.conv_input = nn.Conv2d(self.history_planes, self.channels, kernel_size=5, stride=1, padding=1)
        self.bn_input = nn.BatchNorm2d(self.channels)

        # 计算经过 conv_input 后的空间维度
        # H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
        # W_out = floor((W_in + 2*padding - kernel_size) / stride) + 1
        self.conv_out_x = (self.board_x + 2 * 1 - 5) // 1 + 1  # 等于 self.board_x - 2
        self.conv_out_y = (self.board_y + 2 * 1 - 5) // 1 + 1  # 等于 self.board_y - 2

        if self.conv_out_x <= 0 or self.conv_out_y <= 0:
             raise ValueError(f"Board size ({self.board_x}x{self.board_y}) is too small for conv_input kernel=5, padding=1.")


        # 残差层 (使用修正后的 ResBlock, padding=2, 维度不变)
        self.res_layers = nn.ModuleList([
            ResBlock(self.channels, args.dropout) for _ in range(15)
        ])

        # 策略头
        self.conv_policy = nn.Conv2d(self.channels, 32, kernel_size=1) # kernel=1 不改变维度
        self.bn_policy = nn.BatchNorm2d(32)
        # 使用 conv_input 缩小后的维度计算扁平化尺寸
        self.fc_policy_in_features = 32 * self.conv_out_x * self.conv_out_y
        self.fc_policy = nn.Linear(self.fc_policy_in_features, self.action_size)

        # 价值头
        self.conv_value = nn.Conv2d(self.channels, 32, kernel_size=1) # kernel=1 不改变维度
        self.bn_value = nn.BatchNorm2d(32)
         # 使用 conv_input 缩小后的维度计算扁平化尺寸
        self.fc_value1_in_features = 32 * self.conv_out_x * self.conv_out_y
        self.fc_value1 = nn.Linear(self.fc_value1_in_features, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        # 输入层 - x 现在应该是形状 (batch_size, 21, board_x, board_y)
        # 输出形状 (batch, channels, board_x-2, board_y-2)
        x = F.relu(self.bn_input(self.conv_input(x)))

        # 残差层 - 维度保持 (batch, channels, board_x-2, board_y-2)
        for res_layer in self.res_layers:
            x = res_layer(x)

        # 策略头
        # 输出形状 (batch, 32, board_x-2, board_y-2)
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        # 扁平化 - 使用计算好的输入维度
        policy = policy.view(-1, self.fc_policy_in_features)
        policy = self.fc_policy(policy)

        # 价值头
        # 输出形状 (batch, 32, board_x-2, board_y-2)
        value = F.relu(self.bn_value(self.conv_value(x)))
        # 扁平化 - 使用计算好的输入维度
        value = value.view(-1, self.fc_value1_in_features)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))

        return F.log_softmax(policy, dim=1), value


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.6f}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NNetWrapper:
    """神经网络包装器，处理训练和预测"""

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = GomokuNNet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        # 使用CUDA或CPU
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")
        self.nnet.to(self.device)

        # 设置优化器
        if args.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.learning_rate.max)
        elif args.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.nnet.parameters(), lr=args.learning_rate.max, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        # 初始化当前步数和循环学习率参数
        self.current_step = 0
        self.cycle_size = None  # 在train方法中初始化
        self.lr_min = args.learning_rate.min
        self.lr_max = args.learning_rate.max
        log.info(f"设置循环学习率: 最小值={self.lr_min}, 最大值={self.lr_max}")

    def get_learning_rate(self):
        """
        实现循环学习率策略 (Cyclic LR)
        每个循环包含两个阶段:
        1. 从最小学习率逐步增加到最大学习率
        2. 从最大学习率逐步减小到最小学习率
        """
        if self.current_step >= self.total_steps:
            return self.lr_min  # 训练结束后使用最小学习率

        # 一个完整的周期大小是总步数的 1/5
        if self.cycle_size is None:
            self.cycle_size = self.total_steps // 5
            if self.cycle_size == 0:  # 防止除零错误
                self.cycle_size = 1
            log.info(f"循环学习率周期大小: {self.cycle_size} 步")

        # 计算当前在哪个周期内
        cycle = (self.current_step // self.cycle_size) % 2
        # 计算在当前周期内的位置 (0 到 1 的值)
        x = (self.current_step % self.cycle_size) / self.cycle_size

        # 如果是第一个周期 (0)，学习率从min增加到max
        # 如果是第二个周期 (1)，学习率从max减少到min
        if cycle == 0:
            # 从min线性增加到max
            return self.lr_min + (self.lr_max - self.lr_min) * x
        else:
            # 从max线性减少到min
            return self.lr_max - (self.lr_max - self.lr_min) * x

    def train(self, examples):
        """
        训练网络
        Args:
            examples: [(board, pi, v)] 训练样本列表
                board: 棋盘状态带有历史记录
                pi: 策略向量
                v: 价值标量
        """
        log.info(f"Training with {len(examples)} examples")

        # 设置为训练模式
        self.nnet.train()

        # 定义损失和指标
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        total_losses = AverageMeter()

        # 计算批次数量和总步数
        batch_count = len(examples) // self.args.batch_size
        self.total_steps = self.args.epochs * batch_count
        log.info(f"总训练步数: {self.total_steps}, 每轮次批次数: {batch_count}")

        # 重置循环学习率参数
        self.current_step = 0
        self.cycle_size = None

        # 随机打乱训练样本
        examples = [(board, pi, v) for board, pi, v in examples]
        for epoch in range(self.args.epochs):
            # 打乱训练样本
            np.random.shuffle(examples)

            t = tqdm(range(batch_count), desc=f'Epoch {epoch + 1}/{self.args.epochs}')
            for _ in t:
                # 更新学习率
                lr = self.get_learning_rate()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.current_step += 1

                # 选择一批样本
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # 转换为PyTorch张量
                try:
                    # 确保boards是正确形状的numpy数组
                    boards_array = np.array([board if isinstance(board, np.ndarray) else board.get_history_planes()
                                             for board in boards])
                    boards = torch.FloatTensor(boards_array.astype(np.float32)).to(self.device)
                    target_pis = torch.FloatTensor(np.array(pis).astype(np.float32)).to(self.device)
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float32)).to(self.device)
                except Exception as e:
                    log.error(f"Error converting data to tensors: {e}")
                    continue

                # 前向传播
                out_pi, out_v = self.nnet(boards)

                # 计算损失
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # 反向传播和优化
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), self.args.grad_clip)

                self.optimizer.step()

                # 记录损失
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                total_losses.update(total_loss.item(), boards.size(0))

                # 更新进度条
                t.set_postfix({
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'pi_loss': pi_losses.avg,
                    'v_loss': v_losses.avg,
                    'total_loss': total_losses.avg
                })

            # 打印训练信息
            log.info(f'Epoch {epoch + 1}/{self.args.epochs} - '
                     f'Policy Loss: {pi_losses.avg:.4f}, '
                     f'Value Loss: {v_losses.avg:.4f}, '
                     f'Total Loss: {total_losses.avg:.4f}, '
                     f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

        return total_losses.avg

    def predict(self, board):
        """
        预测棋盘状态的策略和价值
        Args:
            board: 棋盘状态（带有历史记录）

        Returns:
            pi: 策略向量
            v: 价值标量
        """
        # 设置为评估模式
        self.nnet.eval()

        # 准备输入数据 - 检查board类型并获取历史平面
        if hasattr(board, 'get_history_planes'):
            # 如果board是Board对象，直接获取历史平面
            history_planes = board.get_history_planes()
            board_tensor = torch.FloatTensor(history_planes.astype(np.float32)).to(self.device)
        else:
            # 如果board是numpy数组，创建21个平面(当前+20个空历史)
            history_planes = np.zeros((21, self.board_x, self.board_y), dtype=np.float32)
            history_planes[0] = board  # 当前状态放在第一个平面
            board_tensor = torch.FloatTensor(history_planes).to(self.device)

        # 关闭梯度计算
        with torch.no_grad():
            # 前向传播
            pi, v = self.nnet(board_tensor.unsqueeze(0))  # 添加批次维度

        # 转换为numpy数组
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0][0]

    def loss_pi(self, targets, outputs):
        """策略损失（交叉熵）"""
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """价值损失（均方误差）"""
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='model.pt'):
        """保存模型"""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        checkpoint = {
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            'args': self.args
        }

        torch.save(checkpoint, filepath)
        log.info(f"Model saved to {filepath}")

    def load_checkpoint(self, folder='checkpoint', filename='model.pt'):
        """加载模型"""
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            log.warning(f"No model found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.nnet.load_state_dict(checkpoint['state_dict'])

        # 加载优化器和调度器状态
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
        except KeyError:
            log.warning("Optimizer or scheduler state not found in checkpoint")

        log.info(f"Model loaded from {filepath}")
        return True