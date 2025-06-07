import pygame
import yaml
import argparse
import os

from tetris_game import TetrisGame
from dqn_agent import DQNAgent # Needed for AI mode
from train import train as run_training_mode # Import the training function
from operation_module import generate_move_sequence # << 导入新模块

import yaml

def load_config(config_path='config.yaml'):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The configuration loaded from the YAML file.
    """
    with open(config_path, 'r', encoding='utf-8') as f: # <--- 修改点：添加 encoding='utf-8'
        return yaml.safe_load(f)


def manual_mode(config):
    game = TetrisGame(config, render_mode=True)
    game.reset()
    
    running = True
    fall_time = 0
    fall_speed = 500  # milliseconds per fall step (adjust for difficulty)
    clock = pygame.time.Clock()

    while running:
        game.render() # Render first
        dt = clock.tick(config['fps']) # Get delta time for smooth falling

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game.game_over:
                    if event.key == pygame.K_r: # Restart game
                        game.reset()
                    continue

                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # Let step_manual handle locking if move results in no space below
                _rew, _done, _lines = game.step_manual(event.key)
                if _done: # Game over might be set by step_manual
                    game.game_over = True # Ensure it's set for rendering game over message
                    # game.render() # Render one last time to show game over
                
        if not game.game_over:
            fall_time += dt
            if fall_time >= fall_speed:
                fall_time = 0
                # Simulate a soft drop tick if no other input
                _rew, _done, _lines = game.step_manual(pygame.K_DOWN) # Using K_DOWN for the fall logic
                if _done:
                    game.game_over = True
                    # game.render()
        
        if game.game_over:
            # Display game over message handled by game.render()
            pass # Await 'r' to restart or ESC to quit

    pygame.quit()


def ai_play_mode(config, model_path):
    if not pygame.get_init():
        pygame.init()

    game = TetrisGame(config, render_mode=True)
    agent = DQNAgent(
        input_dims=config['input_dims'],
        hidden_dims=config.get('hidden_dims', config.get('nn_hidden_dims')),
        output_dims=config['output_dims'],
        lr=0, gamma=0, epsilon_start=0, epsilon_end=0, epsilon_decay_frames=0,
        memory_size=1, batch_size=1, target_update_freq=9999999,
        weights_dir=config['weights_dir']
    )
    
    load_success, _ = agent.load_weights(model_path)
    if not load_success:
        print(f"无法从 {model_path} 加载模型。退出AI模式。")
        if pygame.get_init(): pygame.quit()
        return

    # --- 从config加载回放参数 ---
    playback_config = config.get('ai_playback', {})
    playback_enabled = playback_config.get('enabled', False)
    move_delay = playback_config.get('move_delay_ms', 50)
    fall_interval = playback_config.get('auto_fall_interval_ms', 200)

    # --- 游戏主循环 ---
    game.reset()
    running = True
    
    op_sequence = [] # 存储当前方块要执行的操作序列
    fall_timer = 0
    clock = pygame.time.Clock()

    while running:
        dt = clock.tick(config.get('fps', 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if game.game_over and event.key == pygame.K_r:
                    game.reset()
                    op_sequence = []

        if not game.game_over:
            # 如果当前没有操作序列，说明需要AI为新方块决策
            if not op_sequence:
                possible_moves = game.get_all_possible_next_states_and_features()
                if possible_moves:
                    _, chosen_move_data = agent.select_action(possible_moves, is_eval_mode=True)
                    if chosen_move_data:
                        if playback_enabled:
                            op_sequence = generate_move_sequence(game.current_piece, chosen_move_data[0])
                        else: # 瞬移模式
                            game.apply_ai_chosen_action(chosen_move_data[0])
                            pygame.time.wait(fall_interval) 
                else:
                    game.game_over = True
            
            # 如果有操作序列（仅在回放模式下），则按序列执行
            if op_sequence:
                next_op = op_sequence.pop(0)
                # << 核心修改点：调用新的专用方法 >>
                game.execute_atomic_action(next_op) 
                
                if next_op != 'hard_drop':
                    pygame.time.wait(move_delay)
            
            # 处理自动下落（仅在回放模式下）
            if playback_enabled:
                fall_timer += dt
                if fall_timer >= fall_interval:
                    fall_timer = 0
                    # << 核心修改点：调用新的专用方法 >>
                    game.execute_atomic_action('soft_drop')

        # 每一帧都渲染
        game.render() 

    if pygame.get_init():
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Tetris Game")
    parser.add_argument('--mode', type=str, default='manual', choices=['manual', 'ai', 'train'],
                        help="游戏模式: 'manual', 'ai', 或 'train'")
    parser.add_argument('--model_path', type=str, default=None, # Changed from model_load_path to model_path for consistency
                        help="为 'ai' 或 'train' 模式指定预训练模型的路径以加载。")
    
    args = parser.parse_args()
    game_config = load_config()

    if args.mode == 'manual':
        print("启动手动模式...")
        manual_mode(game_config)
    elif args.mode == 'ai':
        model_to_use_for_ai = args.model_path
        if not model_to_use_for_ai: # Auto-load latest if not specified for AI mode
            weights_dir = game_config['weights_dir']
            if os.path.exists(weights_dir) and os.listdir(weights_dir):
                try:
                    weight_files = [f for f in os.listdir(weights_dir) if f.startswith("dqn_tetris_episode_") and f.endswith(".pth")]
                    if weight_files:
                        def get_episode_num_from_filename(filename_str):
                            try: return int(filename_str.split("episode_")[1].split(".pth")[0])
                            except: return -1
                        latest_model_filename = max(weight_files, key=get_episode_num_from_filename)
                        if get_episode_num_from_filename(latest_model_filename) != -1:
                             model_to_use_for_ai = os.path.join(weights_dir, latest_model_filename)
                             print(f"未指定模型路径，AI模式将使用最新模型: {model_to_use_for_ai}")
                        else:
                            print("发现权重文件但无法解析。请为AI模式提供 --model_path。")
                            exit()
                    else:
                        print("权重目录中没有合适的模型文件。请为AI模式提供 --model_path。")
                        exit()
                except ValueError: # No files found after filtering
                     print("权重目录中没有合适的模型文件。请为AI模式提供 --model_path。")
                     exit()
            else:
                print("未指定模型路径，且权重目录为空或不存在。请为AI模式提供 --model_path。")
                exit()
        
        print(f"启动AI模式，使用模型: {model_to_use_for_ai}...")
        ai_play_mode(game_config, model_to_use_for_ai)

    elif args.mode == 'train':
        print("启动训练模式...")
        # 将 args.model_path (可能为 None) 传递给训练函数
        run_training_mode(model_to_load=args.model_path) 
    else:
        print(f"未知模式: {args.mode}")