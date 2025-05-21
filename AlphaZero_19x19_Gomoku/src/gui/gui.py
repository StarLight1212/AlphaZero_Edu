import pygame
import sys
import numpy as np
import time
import logging
from pygame import gfxdraw
import argparse
import os
import threading

# Assume these modules exist in your project structure
# Make sure the paths (src.core, src.utils) are correct relative to where you run this script
try:
    from src.core.game import GomokuGame
    from src.core.neural_network import NNetWrapper
    from src.core.mcts import MCTS # AIPlayer needs this
    from src.utils.config_utils import load_config, print_config # main needs this
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the 'src' directory is in your Python path or the script is run from the correct root directory.")
    sys.exit(1)


# --- Logging Setup ---
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("play_gui.log", mode='a') # Renamed log file for clarity
    ]
)


# --- GomokuGUI Class (User Interface) ---
class GomokuGUI:
    """五子棋GUI类 (Corrected)"""

    def __init__(self, game, args):
        """
        初始化GUI

        Args:
            game: 游戏实例 (GomokuGame)
            args: 配置参数 (e.g., from config or argparse)
        """
        self.game = game
        # Ensure args is an object allowing attribute access (like argparse.Namespace)
        # If args is a dict, convert or access with args.get('key', default)
        class ArgsPlaceholder: # Simple placeholder if args is not provided
            pass
        if not hasattr(args, '__dict__'): # Basic check if it supports attribute access
             log.warning("Args object doesn't support attribute access, using defaults.")
             args = ArgsPlaceholder()

        self.args = args # Store the config object

        # 棋盘大小
        self.board_size = self.game.n # Use n from game instance

        # GUI配置 - Use attributes from the args object directly
        self.cell_size = getattr(args, 'cell_size', 40) # Default 40 if not in config
        self.margin = getattr(args, 'margin', 50)      # Default 50
        self.bottom_margin = getattr(args, 'bottom_margin', 100) # Default 100

        # 计算窗口大小
        board_pixel_width = (self.board_size - 1) * self.cell_size
        board_pixel_height = (self.board_size - 1) * self.cell_size
        self.window_width = board_pixel_width + 2 * self.margin
        self.window_height = board_pixel_height + self.margin + self.bottom_margin

        # Optionally update args, though recalculating might be safer if needed elsewhere
        # self.args.window_width = self.window_width
        # self.args.window_height = self.window_height

        log.info(f"Initialized GUI: Board Size={self.board_size}x{self.board_size}, Cell Size={self.cell_size}, Window=({self.window_width}x{self.window_height})")

        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BROWN = (205, 170, 125) # Board color
        self.RED = (255, 0, 0)
        self.GREEN = (0, 128, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200) # Bottom area background
        self.BUTTON_COLOR = (220, 220, 220)
        self.BUTTON_BORDER = (100, 100, 100)
        self.TEXT_COLOR = (50, 50, 50)

        # 按钮配置
        self.button_height = 40
        self.button_width = 120
        self.button_margin = 20 # Margin between buttons and from window edge

        # 初始化pygame
        pygame.init()
        pygame.display.set_caption("AlphaGomoku - Human vs AI")

        # 创建窗口
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))

        # 加载字体
        try:
            # Try common Chinese fonts first
            self.font = pygame.font.SysFont('simhei', 24) # SimHei is common
        except pygame.error:
            try:
                 self.font = pygame.font.SysFont('microsoftyahei', 24) # Microsoft YaHei
            except pygame.error:
                 log.warning("Chinese fonts (SimHei, MicrosoftYaHei) not found, using default.")
                 self.font = pygame.font.Font(None, 30) # Fallback default font
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 72) # For game over message

        # --- Calculate Button Positions ---
        button_area_y = self.window_height - self.bottom_margin + (self.bottom_margin - self.button_height) // 2
        self.resign_button = pygame.Rect(
            self.window_width - self.button_width - self.button_margin,
            button_area_y, self.button_width, self.button_height
        )
        self.new_game_button = pygame.Rect(
            self.resign_button.left - self.button_width - self.button_margin,
            button_area_y, self.button_width, self.button_height
        )

        # --- Game State ---
        self.board = None
        self.player = 1 # 1=黑棋(先手)，-1=白棋. Start with black.
        self.game_over = False
        self.winner = None # 1 for black, -1 for white, 0 for draw
        self.last_move = None # Store (row, col) of the last move
        self.winning_path = [] # Store list of (row, col) for the winning line

        # AI Status - Managed by main loop
        self.ai_thinking = False # Flag to indicate if AI thread is running
        self.ai_move = None      # Stores the action calculated by AI thread (set by thread, consumed by main)
        self.ai_thread = None    # Holds the AI calculation thread object

        # FPS控制
        self.clock = pygame.time.Clock()
        self.fps = getattr(args, 'fps', 30) # Default 30 fps if not in config

        # --- Calculate Board Drawing Offset ---
        board_grid_width = (self.board_size - 1) * self.cell_size
        board_grid_height = (self.board_size - 1) * self.cell_size
        # Center the grid horizontally
        self.board_offset_x = (self.window_width - board_grid_width) // 2
        # Place the grid considering the top margin
        self.board_offset_y = self.margin # Use top margin directly

        self.reset_game() # Initialize the board state

    def get_coord_from_mouse(self, pos):
        """将鼠标坐标转换为棋盘坐标 (row, col)"""
        x, y = pos
        half_cell = self.cell_size / 2
        tolerance_factor = 0.8 # Click must be within 80% of the half-cell distance to snap

        # Calculate potential row/col based on nearest intersection
        # Adjust for the offset and cell size
        col_float = (x - self.board_offset_x) / self.cell_size
        row_float = (y - self.board_offset_y) / self.cell_size

        # Round to nearest intersection index
        col = round(col_float)
        row = round(row_float)

        # Check if rounded indices are within board bounds
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            # Calculate the exact center of the target intersection
            center_x = self.board_offset_x + col * self.cell_size
            center_y = self.board_offset_y + row * self.cell_size

            # Check if the click is close enough to this intersection center
            if abs(x - center_x) < half_cell * tolerance_factor and \
               abs(y - center_y) < half_cell * tolerance_factor:
                return (row, col)

        return None # Clicked too far from an intersection or outside board bounds

    def get_action_from_coord(self, coord):
        """将棋盘坐标(row, col)转换为动作索引"""
        return self.game.get_action_from_coord(coord)

    def get_coord_from_action(self, action):
        """将动作索引转换为棋盘坐标(row, col)"""
        if action < 0 or action >= self.board_size * self.board_size:
            log.error(f"Invalid action index received: {action}")
            return None
        return self.game.get_coord_from_action(action)

    def draw_board(self):
        """绘制棋盘、棋子、按钮和状态"""
        # 1. Fill background (board area and bottom area)
        self.screen.fill(self.BROWN) # Board area background
        pygame.draw.rect(self.screen, self.GRAY, (0, self.window_height - self.bottom_margin, self.window_width, self.bottom_margin)) # Bottom area

        # 2. Draw grid lines
        board_pixel_width = (self.board_size - 1) * self.cell_size
        board_pixel_height = (self.board_size - 1) * self.cell_size
        for i in range(self.board_size):
            # Horizontal lines
            start_pos_h = (self.board_offset_x, self.board_offset_y + i * self.cell_size)
            end_pos_h = (self.board_offset_x + board_pixel_width, self.board_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.BLACK, start_pos_h, end_pos_h, 1)
            # Vertical lines
            start_pos_v = (self.board_offset_x + i * self.cell_size, self.board_offset_y)
            end_pos_v = (self.board_offset_x + i * self.cell_size, self.board_offset_y + board_pixel_height)
            pygame.draw.line(self.screen, self.BLACK, start_pos_v, end_pos_v, 1)

        # Optional: Draw thicker outer border for the grid
        # pygame.draw.rect(self.screen, self.BLACK,
        #                   (self.board_offset_x, self.board_offset_y, board_pixel_width, board_pixel_height), 2)

        # 3. Draw star points (Hoshi)
        if self.board_size >= 9:
            star_points_indices = []
            if self.board_size == 15: # Standard 15x15
                points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)] # Center (Tengen) + 4 corners
                star_points_indices.extend(points)
            elif self.board_size == 19: # Standard 19x19
                points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
                star_points_indices.extend(points)
            elif self.board_size == 9: # Standard 9x9
                 points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
                 star_points_indices.extend(points)
            # Add more cases or a general calculation if needed for other sizes

            for row, col in star_points_indices:
                if 0 <= row < self.board_size and 0 <= col < self.board_size:
                    center_x = self.board_offset_x + col * self.cell_size
                    center_y = self.board_offset_y + row * self.cell_size
                    pygame.draw.circle(self.screen, self.BLACK, (center_x, center_y), 5) # Small solid circle

        # 4. Draw stones (pieces)
        if self.board is not None:
            radius = int(self.cell_size * 0.45) # Stone radius
            for row in range(self.board_size):
                for col in range(self.board_size):
                    player_at_pos = self.board.pieces[row, col]
                    if player_at_pos != 0:
                        center_x = self.board_offset_x + col * self.cell_size
                        center_y = self.board_offset_y + row * self.cell_size
                        color = self.BLACK if player_at_pos == 1 else self.WHITE

                        # Draw anti-aliased filled circle for smooth look
                        gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                        gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                        # Optional: Add black border to white stones for visibility
                        # if player_at_pos == -1:
                        #     gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.BLACK)


        # 5. Highlight the last move
        if self.last_move is not None and not self.game_over: # Don't highlight if game just ended
            row, col = self.last_move
            center_x = self.board_offset_x + col * self.cell_size
            center_y = self.board_offset_y + row * self.cell_size
            # Draw a small rectangle marker
            marker_size = 10
            pygame.draw.rect(
                self.screen, self.RED,
                (center_x - marker_size // 2, center_y - marker_size // 2, marker_size, marker_size),
                2 # Line thickness
            )

        # 6. Highlight the winning path (if game is over and there's a winner)
        if self.game_over and self.winner != 0 and self.winning_path:
            highlight_color = self.GREEN # Use Green or Red
            for row, col in self.winning_path:
                 center_x = self.board_offset_x + col * self.cell_size
                 center_y = self.board_offset_y + row * self.cell_size
                 radius = int(self.cell_size * 0.2) # Small inner circle
                 pygame.draw.circle(self.screen, highlight_color, (center_x, center_y), radius, 3) # Draw thicker circle inside winning stones

        # 7. Draw buttons
        self.draw_buttons()

        # 8. Draw game status text
        self.draw_game_state()

        # 9. Draw game over overlay and message (drawn last, on top)
        if self.game_over:
            self.draw_game_over()

    def draw_buttons(self):
        """绘制按钮"""
        # New Game Button
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, self.new_game_button, border_radius=5)
        pygame.draw.rect(self.screen, self.BUTTON_BORDER, self.new_game_button, 2, border_radius=5)
        new_game_text = self.font.render("新游戏", True, self.TEXT_COLOR)
        text_rect = new_game_text.get_rect(center=self.new_game_button.center)
        self.screen.blit(new_game_text, text_rect)

        # Resign Button
        resign_button_color = self.BUTTON_COLOR if not self.game_over else self.GRAY # Dim if game over
        pygame.draw.rect(self.screen, resign_button_color, self.resign_button, border_radius=5)
        pygame.draw.rect(self.screen, self.BUTTON_BORDER, self.resign_button, 2, border_radius=5)
        resign_text = self.font.render("认输", True, self.TEXT_COLOR)
        text_rect = resign_text.get_rect(center=self.resign_button.center)
        self.screen.blit(resign_text, text_rect)

    def draw_game_state(self):
        """绘制游戏状态信息"""
        # Display current player turn or game over status
        if self.game_over:
            status_text = "游戏结束"
        else:
            player_str = '黑棋 (你)' if self.player == 1 else '白棋 (AI)'
            status_text = f"轮到: {player_str}"

        player_surface = self.font.render(status_text, True, self.TEXT_COLOR)
        # Position text in the bottom area, left-aligned
        status_rect = player_surface.get_rect(midleft=(self.button_margin, self.new_game_button.centery))
        self.screen.blit(player_surface, status_rect)

        # Display AI thinking status
        if self.ai_thinking:
            thinking_text = "AI正在思考..."
            thinking_surface = self.font.render(thinking_text, True, self.BLUE)
            # Position next to the status text
            thinking_rect = thinking_surface.get_rect(midleft=(status_rect.right + 30, status_rect.centery))
            self.screen.blit(thinking_surface, thinking_rect)

    def draw_game_over(self):
        """绘制游戏结束时的遮罩和结果文本"""
        # 1. Semi-transparent overlay for the board area
        overlay_height = self.window_height - self.bottom_margin
        overlay = pygame.Surface((self.window_width, overlay_height))
        overlay.set_alpha(180) # Transparency (0-255)
        overlay.fill(self.GRAY) # Use a neutral color
        self.screen.blit(overlay, (0, 0))

        # 2. Determine result text and color
        if self.winner == 0:
            result_text = "游戏平局！"
            text_color = self.BLUE
        elif self.winner == 1: # Human wins (assuming human is Black=1)
            result_text = "恭喜你，获胜！"
            text_color = self.BLACK # Black text for contrast
        elif self.winner == -1: # AI wins (assuming AI is White=-1)
            result_text = "AI 获胜！"
            text_color = self.BLACK # Use black text even for white win for visibility on gray
        else: # Should not happen if winner is set properly
            result_text = "游戏结束"
            text_color = self.RED

        # 3. Render the text using the large font
        text_surface = self.large_font.render(result_text, True, text_color)
        # Center on the overlay area
        text_rect = text_surface.get_rect(center=(self.window_width // 2, overlay_height // 2))

        # 4. Optional: Background box for text
        bg_rect = text_rect.inflate(40, 20) # Padding
        try: # Use try-except for drawing rounded rect in older pygame versions
             pygame.draw.rect(self.screen, self.BROWN, bg_rect, border_radius=10)
             pygame.draw.rect(self.screen, self.BLACK, bg_rect, 3, border_radius=10) # Border
        except TypeError: # Fallback for older Pygame without border_radius in draw.rect
             pygame.draw.rect(self.screen, self.BROWN, bg_rect)
             pygame.draw.rect(self.screen, self.BLACK, bg_rect, 3)


        # 5. Blit the text onto the screen
        self.screen.blit(text_surface, text_rect)

    def handle_click(self, pos, human_player_id):
        """处理鼠标点击事件"""
        # 检查是否点击了棋盘
        coord = self.get_coord_from_mouse(pos)
        if coord and not self.game_over:
            row, col = coord
            
            # 检查落子位置是否为空
            if self.board.pieces[row, col] == 0:
                # 执行落子
                self.last_move = coord
                # 使用Board对象的execute_move
                self.board.execute_move(coord, self.player)
                
                # 检查是否获胜
                result = self.game.get_game_ended(self.board.pieces, self.player)
                if result is not None:
                    self.game_over = True
                    self.winner = self.player if result == 1 else (-self.player if result == -1 else 0)
                    # 如果有获胜路径，计算它
                    if self.winner != 0:  # 不是平局
                        self.winning_path = self.game.get_winning_path(self.board.pieces, self.winner)
                    return True
                
                # 切换玩家
                self.player = -self.player
                return True
        
        # 检查是否点击了按钮
        if self.new_game_button.collidepoint(pos):
            self.reset_game()
            return True
        if self.resign_button.collidepoint(pos):
            self.game_over = True
            self.winner = -human_player_id  # 投降者失败
            return True
            
        return False

    def get_ai_move_threaded(self, ai_player_instance):
        """Target function for the AI thread. Calculates and sets self.ai_move."""
        log.debug(f"AI Thread (Player {ai_player_instance.player_id}) started.")
        try:
            # Pass a copy of the board to prevent modification races
            action = ai_player_instance.play(self.board.pieces.copy())
            self.ai_move = action # Store the result for the main thread
            log.debug(f"AI Thread finished calculation. Action: {self.ai_move}")
        except Exception as e:
            log.error(f"Error during AI calculation thread: {e}", exc_info=True)
            self.ai_move = -1 # Indicate error or invalid action

        # --- DO NOT MODIFY self.ai_thinking here ---
        # The main loop manages the ai_thinking flag based on thread status.

    def execute_ai_move(self, action):
        """执行AI的动作"""
        # 解析动作
        row, col = self.game.get_coord_from_action(action)
        coord = (row, col)
        
        # 检查是否有效
        if self.board.pieces[row, col] != 0:
            log.error(f"AI tried to play an invalid move at {coord}")
            return False
            
        # 执行动作
        self.last_move = coord
        # 使用Board对象的execute_move方法
        self.board.execute_move(coord, self.player)
        
        # 检查游戏状态
        result = self.game.get_game_ended(self.board.pieces, self.player)
        if result is not None:
            self.game_over = True
            self.winner = self.player if result == 1 else (-self.player if result == -1 else 0)
            # 如果有获胜路径，计算它
            if self.winner != 0:  # 不是平局
                self.winning_path = self.game.get_winning_path(self.board.pieces, self.winner)
        else:
            # 切换玩家
            self.player = -self.player
            
        return True

    def reset_game(self):
        """重置游戏状态"""
        # 初始化棋盘为Board对象而不是numpy数组
        initial_board = self.game.get_init_board()
        self.board = self.game.game_to_board(initial_board)
        self.player = 1  # 黑棋先手
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.winning_path = []
        # Reset AI state flags - crucial!
        self.ai_thinking = False
        self.ai_move = None
        self.ai_thread = None # Clear any reference to an old thread

    def run_game(self, human_player_instance, ai_player_instance):
        """
        运行人机对弈游戏主循环 (Corrected Threading)

        Args:
            human_player_instance: HumanGUIPlayer instance (mainly for player ID)
            ai_player_instance: AIPlayer instance
        """
        self.reset_game() # Ensure clean start
        human_player_id = human_player_instance.player_id
        ai_player_id = ai_player_instance.player_id
        log.info(f"Starting game loop. Human: {human_player_id}, AI: {ai_player_id}, First Turn: {self.player}")

        running = True
        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    log.info("Quit event received. Exiting game loop.")
                    running = False
                    # Attempt cleanup - signal AI thread to stop if possible (depends on AI impl.)
                    # For now, just exit; thread will terminate as daemon=True
                    if self.ai_thread and self.ai_thread.is_alive():
                         log.info("AI thread may still be running upon exit.")

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left mouse button
                         # handle_click now checks if it's human's turn etc.
                        self.handle_click(event.pos, human_player_id)

            # --- AI Turn Logic ---
            # Check if it's AI's turn, game is not over, and AI is not already thinking
            if not self.game_over and self.player == ai_player_id and not self.ai_thinking:
                 # Check if thread exists and is finished OR if no thread is running
                 if self.ai_thread is None or not self.ai_thread.is_alive():
                    # Start AI move calculation in a separate thread
                    log.debug(f"Starting AI thinking thread for player {ai_player_id}")
                    self.ai_thinking = True # Set flag: AI calculation is in progress
                    self.ai_move = None     # Clear previous result before starting new calc
                    self.ai_thread = threading.Thread(
                        target=self.get_ai_move_threaded,
                        args=(ai_player_instance,),
                        daemon=True # Allows main program to exit even if thread is stuck
                    )
                    self.ai_thread.start()

            # --- Check AI Thread Completion and Execute Move ---
            # Check if an AI thread exists, has finished, and we haven't processed its move yet
            if self.ai_thread is not None and not self.ai_thread.is_alive() and self.ai_thinking:
                 # Thread has finished, reset the thinking flag
                 log.debug("AI thread finished.")
                 self.ai_thinking = False

                 # Now, check if the thread produced a move (self.ai_move is set)
                 if self.ai_move is not None:
                     action_to_execute = self.ai_move
                     self.ai_move = None # Consume the move immediately after reading
                     # Ensure it's still AI's turn (or game just ended by AI) before executing
                     # This check might be redundant if state changes only happen here, but safe
                     if not self.game_over and self.player == ai_player_id:
                         self.execute_ai_move(action_to_execute)
                     elif self.game_over and self.winner == ai_player_id:
                         # AI just won, no need to execute further, state is already updated
                         log.debug("AI move execution skipped as AI just won.")
                     else:
                          # State changed unexpectedly (e.g., human resigned while AI was thinking)
                          log.warning(f"AI move calculation finished, but state changed (Player: {self.player}, Game Over: {self.game_over}). Discarding AI move.")
                 else:
                     # This case means the thread finished but self.ai_move wasn't set (e.g., error)
                     log.error("AI thread finished but no move was found in self.ai_move.")
                     # Handle as error/forfeit if needed (execute_ai_move might handle None/negative)
                     if not self.game_over and self.player == ai_player_id:
                          self.execute_ai_move(-1) # Trigger forfeit in execute method
                 # Clear the thread reference once processed
                 self.ai_thread = None

            # --- Drawing ---
            self.draw_board() # Draw all elements based on current state

            # --- Update Display ---
            pygame.display.flip() # Update the full screen

            # --- Frame Rate Control ---
            self.clock.tick(self.fps)

        # --- Cleanup ---
        log.info("Exiting Pygame.")
        pygame.quit()


class Player:
    """玩家基类"""

    def __init__(self, game, player_id=1):
        self.game = game
        self.player_id = player_id

    def play(self, board):
        """
        根据当前棋盘状态返回动作
        Args:
            board: 当前棋盘状态

        Returns:
            action: 动作索引（0到n^2-1）
        """
        raise NotImplementedError


class RandomPlayer(Player):
    """随机玩家"""

    def play(self, board):
        valids = self.game.get_valid_moves(board)
        valid_indices = np.where(valids == 1)[0]
        if len(valid_indices) == 0:
            return -1  # 无法走子
        return np.random.choice(valid_indices)


class GreedyGomokuPlayer(Player):
    """贪心玩家"""
    def play(self, board):
        valids = self.game.get_valid_moves(board)
        valid_indices = np.where(valids == 1)[0]
        if len(valid_indices) == 0:
            return -1  # 无法走子

        # 评估每个可行位置的得分
        candidates = []
        for move in valid_indices:
            next_board, _ = self.game.get_next_state(board, self.player_id, move)
            score = self.game.get_score(next_board, self.player_id)
            candidates.append((-score, move))  # 用负分便于排序

        candidates.sort()  # 按分数升序排序（实际是取最高分）
        return candidates[0][1]  # 返回得分最高的动作


# --- AIPlayer Class ---
class AIPlayer:
    """AI玩家类"""
    def __init__(self, game, nnet, args, player_id):
        """
        初始化AI玩家
        
        Args:
            game: 游戏规则实例
            nnet: 神经网络实例
            args: 配置参数
            player_id: 玩家ID (1=黑棋, -1=白棋)
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.player_id = player_id
        self.mcts = MCTS(self.game, self.nnet, self.args)
        
    def play(self, board):
        """
        AI玩家做出决策
        
        Args:
            board: 当前棋盘状态 (Board对象或numpy数组)
            
        Returns:
            action: 选择的动作索引 (0 to n^2-1)
        """
        # 把numpy数组转换为Board对象
        if isinstance(board, np.ndarray):
            board_obj = self.game.game_to_board(board)
        else:
            board_obj = board
            
        # 获取标准形式的棋盘
        canonical_board = self.game.get_canonical_form(board_obj.pieces, self.player_id)
        
        # 创建一个带有历史记录的标准形式Board对象
        canonical_board_obj = self.game.game_to_board(canonical_board)
        if self.player_id == 1:
            canonical_board_obj.history = [np.copy(hist) for hist in board_obj.history]
        else:
            # 如果是第二个玩家，需要翻转历史记录
            canonical_board_obj.history = [np.copy(hist) * -1 for hist in board_obj.history]
        
        # 重置MCTS树
        self.mcts.reset()
        
        # 获取动作概率
        pi = self.mcts.get_action_prob(canonical_board_obj, temp=0)
        
        # 选择最可能的动作
        action = np.argmax(pi)
        return action
        
    def reset(self):
        """重置AI状态"""
        self.mcts.reset()


# --- HumanGUIPlayer Class ---
class HumanGUIPlayer(Player):
    """代表通过GUI交互的人类玩家"""

    def __init__(self, game, player_id):
        """
        初始化人类玩家

        Args:
            game: 游戏实例
            player_id: 玩家ID（1=黑棋，-1=白棋）
        """
        super(HumanGUIPlayer, self).__init__(game, player_id)
        log.info(f"Human Player initialized. ID: {self.player_id}")

    def play(self, board):
        """
        人类玩家的动作通过GUI事件处理获取，此方法不直接用于决策。
        It might be called by generic game loops, so return an indicator.
        """
        # This method shouldn't be called in the GUI-driven loop for the human player.
        # Return None or -1 to indicate the move comes from elsewhere.
        log.debug("HumanGUIPlayer.play() called (should be inactive in GUI mode)")
        return None # Or raise an error