# main.py

import pygame
import yaml
import argparse
import os
import sys
import torch # << 新增导入，analyze_mode中需要用到
import imageio
# 导入我们项目中的其他模块
from tetris_game import TetrisGame
from dqn_agent import DQNAgent 
from train import train as run_training_mode
from operation_module import generate_move_sequence
from tetrominoes import TETROMINOES, Piece # << 新增导入，解决 "TETROMINOES is not defined"
from datetime import datetime


def load_config(config_path='config.yaml'):
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def manual_mode(config):
    """[最终修正版] 运行手动游戏/调试模式，具有稳健的输入处理。"""
    if not pygame.get_init():
        pygame.init()
    
    game = TetrisGame(config, render_mode=True)
    game.reset()
    
    print("\n--- 进入手动/调试模式 ---")
    print("方向键:移动 | 上/X:顺时针 | Z:逆时针 | 空格:硬降 | C/Shift:暂存 | N:下一块 | R:重开 | ESC:退出")

    running = True
    fall_timer = 0
    fall_speed = 50000  # ms
    clock = pygame.time.Clock()

    # --- 完整的、统一的按键到动作的映射 ---
    key_to_action_map = {
        pygame.K_LEFT: 'left',
        pygame.K_RIGHT: 'right',
        pygame.K_DOWN: 'soft_drop',
        pygame.K_SPACE: 'hard_drop',
        pygame.K_UP: 'rotate_cw',
        pygame.K_x: 'rotate_cw',
        pygame.K_z: 'rotate_ccw',
        pygame.K_c: 'hold',
        pygame.K_LSHIFT: 'hold',
        pygame.K_RSHIFT: 'hold'
    }

    while running:
        dt = clock.tick(config.get('fps', 30))

        # --- 稳健的事件处理循环 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                
                if game.game_over:
                    if event.key == pygame.K_r:
                        game.reset()
                        fall_timer = 0
                    continue # 游戏结束时，只响应R键，忽略所有其他按键

                # 游戏进行中的按键处理
                if event.key == pygame.K_n:
                    game._spawn_new_piece() # 调试用
                elif event.key in key_to_action_map:
                    action = key_to_action_map[event.key]
                    game.execute_atomic_action(action)
                # 对于其他所有未在map中定义的按键，将在此被安全地忽略

        # 处理重力自动下落
        if not game.game_over:
            fall_timer += dt
            if fall_timer >= fall_speed:
                fall_timer = 0
                game.execute_atomic_action('soft_drop')
        
        game.render()

    if pygame.get_init():
        pygame.quit()

def ai_play_mode(config, model_path):
    """
    [最终重构版] 运行AI自动游戏模式，实现带Hold决策的分支评估逻辑。
    支持可视化回放和即时放置两种模式。
    """
    if not pygame.get_init():
        pygame.init()

    # --- 初始化游戏和AI代理 ---
    game = TetrisGame(config, render_mode=True)
    # agent = DQNAgent(
    #     input_dims=config['input_dims'],
    #     hidden_dims=config.get('hidden_dims', config.get('nn_hidden_dims')),
    #     output_dims=config['output_dims'],
    #     lr=0, gamma=0, epsilon_start=0, epsilon_end=0, epsilon_decay_frames=0,
    #     memory_size=1, batch_size=1, target_update_freq=9999999,
    #     weights_dir=config['weights_dir']
    # )
    agent = DQNAgent(config)
    
    load_success, loaded_at_episode = agent.load_weights(model_path)
    if not load_success:
        if pygame.get_init(): pygame.quit()
        return

    # --- 加载配置参数 ---
    playback_config = config.get('ai_playback', {})
    playback_enabled = playback_config.get('enabled', True)
    move_delay = playback_config.get('move_delay_ms', 30)
    fall_interval = playback_config.get('auto_fall_interval_ms', 200)

    record_config = playback_config.get('record_game', {})
    record_enabled = record_config.get('enabled', False)
    record_duration_seconds = record_config.get('record_duration_seconds', 0)
    output_fps = record_config.get('output_fps', 15)
    max_frames_to_capture = 0
    if record_enabled and record_duration_seconds > 0:
        max_frames_to_capture = record_duration_seconds * output_fps

    # --- 游戏主循环初始化 ---
    game.reset()
    running = True
    op_sequence = [] 
    clock = pygame.time.Clock()
    frames_for_recording = []
    frame_counter = 0

    print("AI回放模式已启动 (支持Hold决策)。按ESC键退出。")
    if record_enabled:
        print(f"游戏录制已启用。格式: {record_config.get('output_format', 'gif').upper()}")
        if record_duration_seconds > 0:
            print(f"录制时长将被限制为: {record_duration_seconds} 秒。")

    while running:
        dt = clock.tick(config.get('fps', 30))
        
        # --- 事件处理循环 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if game.game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()
                op_sequence = []
                frames_for_recording = []
                frame_counter = 0

        # --- 录制结束逻辑 ---
        if record_enabled and max_frames_to_capture > 0 and len(frames_for_recording) >= max_frames_to_capture:
            print(f"已捕获足够帧数以生成 {record_duration_seconds} 秒的视频，正在结束游戏...")
            game.game_over = True
            running = False

        if not game.game_over:
            # 当上一组操作序列执行完毕后，或在即时模式下，进行新一轮决策
            if not op_sequence:
                
                # ====================================================================
                # ============== 核心决策逻辑: 两分支评估 (Hold vs No-Hold) ==============
                # ====================================================================
                
                decision_is_to_hold = False
                best_action_info_no_hold = None

                # --- 步骤 1: 检查是否可以执行Hold ---
                if not game.can_hold:
                    # Case 1: Cannot hold, no choice to make. The decision is forced to "No-Hold".
                    decision_is_to_hold = False
                    # We still need to find the best move for the current piece
                    moves_no_hold = game.get_all_possible_next_states_and_features(for_piece=game.current_piece)
                    best_action_no_hold, v_no_hold = agent.get_best_action_and_value(moves_no_hold)
                    if best_action_no_hold:
                        best_action_info_no_hold = best_action_no_hold[0]
                else:
                    # Case 2: Can hold, perform the full two-branch evaluation.
                    
                    # --- 分支 A: 评估 "No-Hold" (使用当前方块) ---
                    moves_no_hold = game.get_all_possible_next_states_and_features(for_piece=game.current_piece)
                    best_action_no_hold, v_no_hold = agent.get_best_action_and_value(moves_no_hold)
                    
                    if best_action_no_hold:
                         best_action_info_no_hold = best_action_no_hold[0] # Extract the (type, rot, x, y) tuple

                    # --- 分支 B: 评估 "Hold" (使用 next_piece 或 held_piece) ---
                    piece_for_hold_branch = game.next_piece if game.held_piece is None else game.held_piece
                    moves_hold = game.get_all_possible_next_states_and_features(for_piece=piece_for_hold_branch)
                    _best_action_hold, v_hold_branch = agent.get_best_action_and_value(moves_hold)

                    # --- 最终决策对比 ---
                    if v_hold_branch > v_no_hold:
                        decision_is_to_hold = True
                
                # ====================================================================
                # ==================== 决策结束，准备执行或生成序列 =====================
                # ====================================================================
                
                if playback_enabled:
                    # 模式 1: 为可视化回放生成操作指令序列 ("菜谱")
                    if decision_is_to_hold:
                        op_sequence = ['hold']
                    elif best_action_info_no_hold:
                                            # Pass the whole game object to the new pathfinding function
                        op_sequence = generate_move_sequence(game, best_action_info_no_hold)
                        # If no path is found, fall back to instant placement for this move
                        if op_sequence is None:
                            game.apply_ai_chosen_action(best_action_info_no_hold)

                    else: # No possible moves found
                        game.game_over = True
                else:
                    # 模式 2: 即时应用决策 (无动画)
                    if decision_is_to_hold:
                        game.execute_atomic_action('hold')
                    elif best_action_info_no_hold:
                        game.apply_ai_chosen_action(best_action_info_no_hold)
                    else: # No possible moves found
                        game.game_over = True
                    # 在即时模式下增加一个延迟，以便观察
                    pygame.time.wait(fall_interval) 

            # --- 可视化回放的序列执行 (如果启用) ---
            if playback_enabled and op_sequence:
                next_op = op_sequence.pop(0)
                game.execute_atomic_action(next_op) 
                if next_op not in ['hard_drop', 'hold']: 
                    pygame.time.wait(move_delay)

        game.render() 

        # --- 录制帧画面 ---
        if record_enabled and not game.game_over:
            capture_interval = record_config.get('capture_interval', 2) # Adjusted for smoother recording
            frame_counter += 1
            if frame_counter % capture_interval == 0:
                frame_data = pygame.surfarray.array3d(game.screen)
                frames_for_recording.append(frame_data.transpose([1, 0, 2]))
                
                if max_frames_to_capture > 0 and len(frames_for_recording) >= max_frames_to_capture:
                    print(f"已捕获足够帧数以生成 {record_duration_seconds} 秒的视频，正在结束游戏...")
                    running = False 

    # --- 游戏循环结束后，保存回放文件 ---
    if record_enabled and frames_for_recording:
        print(f"\n正在保存游戏回放为 {record_config.get('output_format', 'gif').upper()}...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_extension = ".mp4" if record_config.get('output_format', 'gif') == "mp4" else ".gif"
        filename = f"ai_playback_ep{loaded_at_episode-1}_{timestamp}{file_extension}"
        
        save_dir = "replays"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        try:
            if record_config.get('output_format', 'gif') == 'mp4':
                imageio.mimsave(save_path, frames_for_recording, fps=output_fps, macro_block_size=1, quality=8)
            else:
                imageio.mimsave(save_path, frames_for_recording, fps=output_fps)
            print(f"游戏回放已成功保存至: {save_path}")
        except Exception as e:
            print(f"保存回放时发生错误: {e}")

    if pygame.get_init():
        pygame.quit()

def analyze_mode(config, model_path):
    """
    交互式分析模式：
    - 使用上下箭头滚动并预览所有可能的落点。
    - 使用 Enter 键执行当前高亮的落点。
    - 使用 Space 键执行AI认为最优的落点。
    """
    if not pygame.get_init():
        pygame.init()

    game = TetrisGame(config, render_mode=True)
    agent = DQNAgent(
        input_dims=config['input_dims'],
        hidden_dims=config.get('hidden_dims', config.get('nn_hidden_dims')),
        output_dims=config['output_dims'],
        lr=0, gamma=0, epsilon_start=0, epsilon_end=0, epsilon_decay_frames=0,
        memory_size=1, batch_size=1, target_update_freq=9999999
    )
    
    load_success, _ = agent.load_weights(model_path)
    if not load_success:
        if pygame.get_init(): pygame.quit()
        return

    game.reset()
    running = True
    
    evaluated_moves = []
    needs_re_evaluation = True # 控制是否需要AI重新进行一轮决策
    scroll_offset = 0 # 控制列表的滚动
    
    font = pygame.font.Font(None, 20)
    highlight_font = pygame.font.Font(None, 22) # 用于高亮选项

    while running:
        # --- 步骤1: 如果需要，进行AI决策 ---
        if needs_re_evaluation:
            if game.game_over:
                print("游戏结束。按R键重开，按ESC键退出。")
                # 保持循环以响应按键，但停止AI决策
            else:
                possible_moves = game.get_all_possible_next_states_and_features()
                if not possible_moves:
                    game.game_over = True
                    print("没有可行的移动, 游戏结束。")
                else:
                    evaluated_moves = []
                    with torch.no_grad():
                        for move_info, features in possible_moves:
                            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(agent.device)
                            value = agent.policy_net(features_tensor).item()
                            evaluated_moves.append({'info': move_info, 'features': features, 'value': value})
                    evaluated_moves.sort(key=lambda x: x['value'], reverse=True)
                    scroll_offset = 0 # 每次重新决策后，滚动条归零
            
            needs_re_evaluation = False

        # --- 步骤2: 处理用户输入 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if game.game_over and event.key == pygame.K_r: # 游戏结束后按R重置
                    game.reset()
                    needs_re_evaluation = True
                    continue # 开始下一次循环以重新决策
                
                # --- 新的交互逻辑 ---
                if not game.game_over:
                    if event.key == pygame.K_DOWN:
                        scroll_offset = min(len(evaluated_moves) - 1, scroll_offset + 1)
                    elif event.key == pygame.K_UP:
                        scroll_offset = max(0, scroll_offset - 1)
                    elif event.key == pygame.K_RETURN: # Enter键：执行当前高亮的动作
                        if evaluated_moves:
                            chosen_move = evaluated_moves[scroll_offset]
                            game.apply_ai_chosen_action(chosen_move['info'])
                            needs_re_evaluation = True # 标记需要为下一个方块做决策
                    elif event.key == pygame.K_SPACE: # Space键：执行最优动作
                        if evaluated_moves:
                            best_move = evaluated_moves[0]
                            game.apply_ai_chosen_action(best_move['info'])
                            needs_re_evaluation = True # 标记需要为下一个方块做决策
        
        # --- 步骤3: 绘制界面 ---
        game.screen.fill((20, 20, 20))
        game.draw_board()
        game.draw_gridlines()
        game.draw_piece(game.current_piece) # 绘制当前在顶部的方块

        if not game.game_over and evaluated_moves:
            # --- 预览高亮的落点 ---
            selected_move = evaluated_moves[scroll_offset]
            _, rot, x, y = selected_move['info']
            # 创建一个临时的Piece对象用于绘制幽灵预览
            preview_ghost_piece = Piece(x, y, game.current_piece.type, rot)
            game.draw_piece(preview_ghost_piece, is_ghost=True)

            # --- 在右侧绘制可滚动的决策列表 ---
            panel_x = game.board_width * game.block_size + 10
            current_y = 10
            line_height = 20
            
            # 为了让高亮选项始终可见，我们调整列表的起始显示位置
            list_display_start_index = max(0, scroll_offset - 5) # 让高亮选项尽量在屏幕中间

            for i, move in enumerate(evaluated_moves[list_display_start_index:]):
                if current_y > game.screen_height - 30: break
                
                actual_index = i + list_display_start_index
                is_selected = (actual_index == scroll_offset)
                
                text_color = (255, 255, 0) if is_selected else ((0, 255, 0) if actual_index == 0 else (220, 220, 220))
                current_font = highlight_font if is_selected else font
                
                info_text = f"#{actual_index + 1} V: {move['value']:.2f}"
                text_surf = current_font.render(info_text, True, text_color)
                game.screen.blit(text_surf, (panel_x, current_y))
                current_y += line_height

        if game.game_over:
            # 游戏结束后也保持界面，以便响应重开或退出
            font_large = pygame.font.Font(None, 50)
            text_surf = font_large.render("GAME OVER", True, (255,0,0))
            text_rect = text_surf.get_rect(center=(game.screen_width/2, game.screen_height/2))
            game.screen.blit(text_surf, text_rect)

        pygame.display.flip()

    if pygame.get_init():
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Tetris Game")
    
    # 修正后的参数定义
    parser.add_argument('--mode', 
                        type=str, 
                        default='manual', 
                        choices=['manual', 'ai', 'train', 'analyze'],
                        help="游戏模式: 'manual', 'ai', 'train', 或 'analyze'")
    
    parser.add_argument('--model_path', 
                        type=str, 
                        default=None,
                        help="为 'ai', 'train', 或 'analyze' 模式指定预训练模型的路径。")
    
    args = parser.parse_args()
    game_config = load_config()

    if args.mode == 'manual':
        print("启动手动模式...")
        manual_mode(game_config)
    
    elif args.mode == 'ai':
        model_to_use_for_ai = args.model_path
        if not model_to_use_for_ai:
            weights_dir = game_config.get('weights_dir', 'weights/')
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
                            sys.exit()
                    else:
                        print("权重目录中没有合适的模型文件。请为AI模式提供 --model_path。")
                        sys.exit()
                except (ValueError, FileNotFoundError):
                     print("权重目录中没有合适的模型文件或目录不存在。请为AI模式提供 --model_path。")
                     sys.exit()
            else:
                print("未指定模型路径，且权重目录为空或不存在。请为AI模式提供 --model_path。")
                sys.exit()
        
        print(f"启动AI模式，使用模型: {model_to_use_for_ai}...")
        ai_play_mode(game_config, model_to_use_for_ai)

    elif args.mode == 'train':
        print("启动训练模式...")
        run_training_mode(model_to_load=args.model_path) 
    
    elif args.mode == 'analyze':
        model_to_use_for_analysis = args.model_path
        if not model_to_use_for_analysis:
            # 您可以复用上面ai_play_mode中的自动加载逻辑
            print("分析模式需要一个模型，请使用 --model_path 指定。")
            sys.exit()
        
        print(f"启动分析模式，使用模型: {model_to_use_for_analysis}...")
        analyze_mode(game_config, model_to_use_for_analysis)

    else:
        print(f"未知模式: {args.mode}")