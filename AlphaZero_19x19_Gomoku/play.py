import pygame
import sys
import logging
from pygame import gfxdraw
import argparse
import os
import threading
from src.core.game import GomokuGame
from src.core.neural_network import NNetWrapper
from src.gui.gui import GomokuGUI, AIPlayer, HumanGUIPlayer
# MCTS is no longer needed by AIPlayer in this version
# from src.core.mcts import MCTS
from src.utils.config_utils import load_config, print_config # main needs this


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("play_gui_direct.log", mode='a') # Changed log file name
    ]
)

log = logging.getLogger(__name__)

# --- Main Execution ---
# [NO CHANGES TO main function - Keep the previous version]
# ... (main function code remains exactly the same as in the previous response) ...
def main():
    """运行人机对弈的主函数"""
    parser = argparse.ArgumentParser(description='与AlphaZero五子棋AI对弈 (GUI - Direct Policy)') # Updated description
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径 (e.g., models/best.pth.tar). Overrides config.')
    # --human_first flag is now ignored, human is always first (black)
    parser.add_argument('--human_first', action='store_true', help='人类玩家先手（此版本中无效，人类总是先手）')
    parser.add_argument('--mcts_sims', type=int, default=None, help='MCTS模拟次数 (Ignored in Direct Policy mode).') # Added note
    cli_args = parser.parse_args()

    # --- Load Config ---
    config = load_config(cli_args.config)

    # --- Override Config with CLI Args ---
    # Note: mcts_sims is loaded but not used by the modified AIPlayer
    if cli_args.mcts_sims is not None:
        log.info(f"Overriding config 'num_sims' ({config.num_sims}) with CLI argument: {cli_args.mcts_sims} (Note: Not used in Direct Policy mode)")
        config.num_sims = cli_args.mcts_sims
        # Also update MCTS args if separate struct exists in config, e.g., config.mcts.num_simulations

    # Handle model path override
    if cli_args.model is not None:
        if os.path.exists(cli_args.model):
            model_path = cli_args.model
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path)
            log.info(f"Overriding config model load path with CLI argument: {model_path}")
            config.load_folder_file = [model_dir, model_name]
            config.load_model = True # Ensure loading is enabled
        else:
            log.error(f"Specified model file does not exist: {cli_args.model}. Exiting.")
            sys.exit(1)
    elif config.load_model:
         # Use model path from config if load_model is true and no CLI override
         model_path = os.path.join(config.load_folder_file[0], config.load_folder_file[1])
         log.info(f"Using model path from config: {model_path}")
         if not os.path.exists(model_path):
              log.warning(f"Model file specified in config not found: {model_path}. AI will use an untrained network.")
              # Optionally disable loading: config.load_model = False
    else:
         log.warning("No model specified via CLI and 'load_model' is false or path not set in config. AI will use an untrained network.")

    # --- Print Final Config ---
    log.info("--- Effective Configuration ---")
    # Use the utility function if it prints nicely, otherwise log key parts
    try:
        print_config(config) # Assumes this function logs or prints
    except NameError: # Fallback if print_config wasn't imported
        log.info(f"Board Size: {config.board_size}")
        log.info(f"Win Length: {config.win_length}")
        log.info(f"MCTS Sims: {config.num_sims} (Ignored)")
        log.info(f"Load Model: {config.load_model}")
        if config.load_model:
             log.info(f"Model Path: {os.path.join(config.load_folder_file[0], config.load_folder_file[1])}")
    log.info("-----------------------------")

    # --- Initialize Game and Network ---
    game = GomokuGame(board_size=config.board_size, win_length=config.win_length)
    nnet = NNetWrapper(game, config) # Pass the config object to NNetWrapper

    # --- Load Network Model ---
    if config.load_model:
        model_file_path = os.path.join(config.load_folder_file[0], config.load_folder_file[1])
        if os.path.exists(model_file_path):
            log.info(f"Loading model checkpoint from: {model_file_path}")
            try:
                 nnet.load_checkpoint(config.load_folder_file[0], config.load_folder_file[1])
            except Exception as e:
                 log.error(f"Failed to load model checkpoint: {e}", exc_info=True)
                 log.warning("Continuing with untrained network.")
        # else: warning already logged above

    # --- Set Players (Human = Black = 1, AI = White = -1) ---
    human_player_id = 1
    ai_player_id = -1
    log.info("--- Player Assignment ---")
    log.info("Human plays as Black (先手), Player ID: 1")
    log.info("AI plays as White (后手), Player ID: -1 (Using Direct Policy)") # Added note
    log.info("-------------------------")

    # --- Create Player Instances ---
    # Pass the config object (args) to AIPlayer
    ai_player = AIPlayer(game, nnet, config, player_id=ai_player_id) # Uses the modified AIPlayer
    human_player = HumanGUIPlayer(game, player_id=human_player_id)

    # --- Create and Run GUI ---
    # Pass the config object (args) to GomokuGUI
    gui = GomokuGUI(game, config)
    gui.run_game(human_player, ai_player) # Pass player instances


if __name__ == "__main__":
    main()
