# train.py
# (版本：支持继续训练、详细日志记录、以及为截图传递评估信息)

import pygame
import yaml
import os
import numpy as np
import time
from datetime import datetime
from collections import deque, defaultdict

from tetris_game import TetrisGame
from dqn_agent import DQNAgent

def load_config(config_path='config.yaml'):
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    return config_data

def train(model_to_load=None):
    """
    执行AI智能体的训练过程。

    Args:
        model_to_load (str, optional): 要加载的预训练模型的路径。
                                     如果为 None，则根据配置尝试自动加载或从头开始。
    """
    config = load_config()
    
    # 创建必要的目录
    os.makedirs(config.get('weights_dir', 'weights/'), exist_ok=True)
    os.makedirs(config.get('screenshots_dir', 'screenshots/'), exist_ok=True)
    os.makedirs(config.get('logs_dir', 'logs/'), exist_ok=True)

    # 初始化游戏环境 (训练时通常不渲染以提高速度)
    game = TetrisGame(config, render_mode=False) 
    
    # 初始化 DQNAgent
    agent = DQNAgent(
        input_dims=config['input_dims'],
        hidden_dims=config.get('hidden_dims', config.get('nn_hidden_dims')),
        output_dims=config['output_dims'],
        lr=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay_frames=config['epsilon_decay_frames'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq'],
        weights_dir=config['weights_dir']
    )

    start_absolute_episode = 1 # 默认从第一个episode开始
    
    path_to_attempt_load = model_to_load

    if not path_to_attempt_load and config.get('auto_load_latest_weights', True):
        weights_dir = config['weights_dir']
        if os.path.exists(weights_dir):
            weight_files = [f for f in os.listdir(weights_dir) if f.startswith("dqn_tetris_episode_") and f.endswith(".pth")]
            if weight_files:
                def get_episode_num_from_filename(filename_str):
                    try:
                        return int(filename_str.split("episode_")[1].split(".pth")[0])
                    except (IndexError, ValueError):
                        return -1
                
                latest_model_filename = max(weight_files, key=get_episode_num_from_filename)
                if get_episode_num_from_filename(latest_model_filename) != -1:
                    path_to_attempt_load = os.path.join(weights_dir, latest_model_filename)
                    print(f"未提供模型路径，尝试自动加载最新模型: {path_to_attempt_load}")
                else:
                    print("发现权重文件但无法解析episode编号。将开始新训练。")
            else:
                print(f"权重目录 '{weights_dir}' 为空或无匹配文件。将开始新训练。")
        else:
            print(f"权重目录 '{weights_dir}' 未找到。将开始新训练。")

    if path_to_attempt_load:
        load_success, next_episode_to_start_from = agent.load_weights(path_to_attempt_load)
        if load_success:
            start_absolute_episode = next_episode_to_start_from
            print(f"将从绝对Episode {start_absolute_episode} 继续训练。")
        else:
            print(f"无法从 {path_to_attempt_load} 加载权重。将从Episode 1开始新训练。")
    else:
        print("未指定或未找到可加载的模型。将从Episode 1开始新训练。")

    # --- 初始化日志和追踪变量 ---
    reward_component_keys = [
        "score_change", "height_penalty", "hole_penalty",
        "bumpiness_penalty", "lines_reward", "combo_reward",
        "well_occupancy_penalty", # << 新增
        "game_over_penalty", "piece_drop_reward"
    ]
# ...
    episode_total_rewards_deque = deque(maxlen=100)
    episode_lines_deque = deque(maxlen=100)
    episode_step_avg_reward_components_deques = defaultdict(lambda: deque(maxlen=100))
    
    start_time = time.time()

    log_file_path = os.path.join(config['logs_dir'], f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    header_parts = ["Episode", "TotalSteps", "Epsilon", "AvgTotalReward100", "AvgLines100", "Loss", "TimeElapsed"]
    for key in reward_component_keys:
        col_name = f"Avg_{key.replace('_penalty','P').replace('_reward','R').replace('_change','Chg').capitalize()[:7]}_100"
        header_parts.append(col_name)
    csv_header = ",".join(header_parts) + "\n"
    
    with open(log_file_path, 'w') as log_f:
        log_f.write(csv_header)

    num_episodes_this_session = config.get('num_episodes', 50000)

    for i in range(num_episodes_this_session): 
        current_absolute_episode = start_absolute_episode + i
        
        game.reset() 
        current_episode_total_reward = 0.0
        current_episode_lines = 0
        current_episode_steps = 0
        current_episode_sum_reward_components = defaultdict(float)
        
        possible_moves_with_features = game.get_all_possible_next_states_and_features()
        loss = None

        for step_in_episode in range(config.get('max_steps_per_episode', 3000)):
            if game.game_over or not possible_moves_with_features:
                break

            agent.frames_done += 1 
            agent._decay_epsilon() 
            current_episode_steps += 1

            chosen_action_idx, chosen_move_data = agent.select_action(possible_moves_with_features)
            
            if chosen_move_data is None: 
                break 
                
            chosen_action_info = chosen_move_data[0]
            features_of_chosen_s_prime = chosen_move_data[1]

            reward_info_dict, game_over_after_action, lines_cleared_count = game.apply_ai_chosen_action(chosen_action_info)
            total_reward_this_step = reward_info_dict["final_reward"]

            current_episode_total_reward += total_reward_this_step
            current_episode_lines += lines_cleared_count
            for key in reward_component_keys:
                current_episode_sum_reward_components[key] += reward_info_dict.get(key, 0.0)

            next_possible_moves_with_features = []
            if not game_over_after_action:
                next_possible_moves_with_features = game.get_all_possible_next_states_and_features()

            next_best_q_value = 0.0
            if not game_over_after_action and next_possible_moves_with_features:
                 next_best_q_value = agent.get_max_q_value_for_next_states(next_possible_moves_with_features)
            
            agent.remember(features_of_chosen_s_prime, chosen_action_idx, total_reward_this_step, next_best_q_value, game_over_after_action)
            loss = agent.learn()
            
            possible_moves_with_features = next_possible_moves_with_features

        # --- Episode 结束 ---
        episode_total_rewards_deque.append(current_episode_total_reward)
        episode_lines_deque.append(current_episode_lines)
        if current_episode_steps > 0:
            for key in reward_component_keys:
                avg_component_val = current_episode_sum_reward_components[key] / current_episode_steps
                episode_step_avg_reward_components_deques[key].append(avg_component_val)
        else:
            for key in reward_component_keys:
                episode_step_avg_reward_components_deques[key].append(0.0)

        # --- 定期日志输出 ---
        if current_absolute_episode % config['log_freq'] == 0:
            elapsed_time = time.time() - start_time
            avg_total_reward_100 = np.mean(episode_total_rewards_deque) if episode_total_rewards_deque else 0.0
            avg_lines_100 = np.mean(episode_lines_deque) if episode_lines_deque else 0.0
            
            console_log_parts = [
                f"Ep: {current_absolute_episode}", f"TotalSteps: {agent.frames_done}", 
                f"Eps: {agent.epsilon:.4f}", f"AvgRew100: {avg_total_reward_100:.3f}", 
                f"AvgLines100: {avg_lines_100:.3f}", f"Loss: {loss if loss is not None else 0.0:.4f}"
            ]
            print(", ".join(console_log_parts))

            console_detail_parts = ["  AvgStepComps100:"]
            csv_log_values = [
                str(current_absolute_episode), str(agent.frames_done), f"{agent.epsilon:.4f}",
                f"{avg_total_reward_100:.4f}", f"{avg_lines_100:.4f}", 
                f"{loss if loss is not None else 0.0:.4f}", f"{elapsed_time:.2f}"
            ]
            for key in reward_component_keys:
                avg_val_100_steps = np.mean(episode_step_avg_reward_components_deques[key]) if episode_step_avg_reward_components_deques[key] else 0.0
                short_key = key.replace('_penalty','P').replace('_reward','R').replace('_change','Chg').capitalize()[:7]
                console_detail_parts.append(f"{short_key}: {avg_val_100_steps:.2f}")
                csv_log_values.append(f"{avg_val_100_steps:.4f}")
            print(" | ".join(console_detail_parts))
            
            with open(log_file_path, 'a') as log_f:
                log_f.write(",".join(csv_log_values) + "\n")

        # --- 定期保存模型 ---
        if current_absolute_episode > 0 and current_absolute_episode % config['save_model_freq'] == 0:
            agent.save_weights(current_absolute_episode)

        # --- 定期评估 ---
        if current_absolute_episode > 0 and current_absolute_episode % config['eval_freq'] == 0:
            print(f"--- 正在为 Episode {current_absolute_episode} 运行评估 ---")
            reward_params_to_display = {
                k: config.get(k, 'N/A') for k in [
                    "height_penalty_factor", "hole_penalty_factor", "bumpiness_penalty_factor",
                    "lines_cleared_reward_factor", "game_over_penalty", "piece_drop_reward", "combo_base_reward"
                ]
            }

            eval_game = TetrisGame(config, render_mode=True) 
            eval_game.reset() 
            eval_episode_total_reward = 0.0
            eval_episode_lines = 0
            eval_possible_moves = eval_game.get_all_possible_next_states_and_features()
            
            for eval_step in range(config.get('max_steps_per_episode', 3000)):
                if eval_game.game_over or not eval_possible_moves:
                    eval_game.render(episode_num_for_eval=current_absolute_episode, reward_params_for_eval=reward_params_to_display)
                    pygame.time.wait(config.get('eval_step_wait_ms', 100))
                    break 
                
                _idx, eval_chosen_move_data = agent.select_action(eval_possible_moves, is_eval_mode=True)
                if eval_chosen_move_data is None: 
                    eval_game.render(episode_num_for_eval=current_absolute_episode, reward_params_for_eval=reward_params_to_display)
                    pygame.time.wait(config.get('eval_step_wait_ms', 100))
                    break
                
                eval_action_info = eval_chosen_move_data[0]
                eval_reward_info_dict, eval_done, eval_l_cleared = eval_game.apply_ai_chosen_action(eval_action_info)
                
                eval_episode_total_reward += eval_reward_info_dict["final_reward"]
                eval_episode_lines += eval_l_cleared
                
                if not eval_done:
                    eval_possible_moves = eval_game.get_all_possible_next_states_and_features()
                
                eval_game.render(episode_num_for_eval=current_absolute_episode, reward_params_for_eval=reward_params_to_display)
                pygame.time.wait(config.get('eval_step_wait_ms', 50))

            screenshot_path = os.path.join(config['screenshots_dir'], f"eval_episode_{current_absolute_episode}_final.png")
            eval_game.get_screenshot(screenshot_path) 
            print(f"--- 评估结束. 得分: {eval_game.score}, 消行数: {eval_episode_lines}, 本局总奖励: {eval_episode_total_reward:.2f} ---")
            del eval_game

    # --- 训练会话结束 ---
    if num_episodes_this_session > 0:
        final_absolute_episode_trained = start_absolute_episode + num_episodes_this_session - 1
        print(f"训练会话结束。最后一个训练的绝对Episode: {final_absolute_episode_trained}")
        agent.save_weights(final_absolute_episode_trained)
    else:
        print("未运行新的训练episodes。")

if __name__ == '__main__':
    # 如果直接运行 train.py，可以测试加载功能
    # 例如：train(model_to_load="weights/dqn_tetris_episode_SOME_NUMBER.pth")
    # 或不带参数以自动加载或全新开始：
    train()