# train.py
# (版本：最终修复版，集成所有功能，无代码冲突)

import pygame
import yaml
import os
import numpy as np
import torch
import time
from datetime import datetime
from collections import deque, defaultdict
import mlflow
import mlflow.pytorch

from tetris_game import TetrisGame
from dqn_agent import DQNAgent

def load_config(config_path='config.yaml'):
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    return config_data

def train(model_to_load=None):
    """
    [最终修复版] 执行AI智能体的训练过程，并使用MLflow进行监控。
    """
    config = load_config()
    
    # --- 1. 初始化环境和目录 ---
    os.makedirs(config.get('weights_dir', 'weights/'), exist_ok=True)
    os.makedirs(config.get('screenshots_dir', 'screenshots/'), exist_ok=True)
    os.makedirs(config.get('logs_dir', 'logs/'), exist_ok=True)
    
    game = TetrisGame(config, render_mode=False) 
    agent = DQNAgent(config)

    # --- 2. 加载模型逻辑 ---
    start_absolute_episode = 1
    path_to_attempt_load = model_to_load
    if not path_to_attempt_load and config.get('auto_load_latest_weights', True):
        weights_dir = config['weights_dir']
        if os.path.exists(weights_dir):
            weight_files = [f for f in os.listdir(weights_dir) if f.startswith("dqn_tetris_episode_") and f.endswith(".pth")]
            if weight_files:
                def get_episode_num_from_filename(filename_str):
                    try: return int(filename_str.split("episode_")[1].split(".pth")[0])
                    except (IndexError, ValueError): return -1
                latest_model_filename = max(weight_files, key=get_episode_num_from_filename)
                if get_episode_num_from_filename(latest_model_filename) != -1:
                    path_to_attempt_load = os.path.join(weights_dir, latest_model_filename)
                    print(f"未提供模型路径，尝试自动加载最新模型: {path_to_attempt_load}")
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

    # --- 3. MLflow 和日志初始化 ---
    mlflow.set_experiment("Tetris CNN T-Spin Training")

    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run-ID: {run_id}")

        mlflow.log_params({
            'learning_rate': config['learning_rate'], 'gamma': config['gamma'],
            'batch_size': config['batch_size'], 'epsilon_start': config['epsilon_start'],
            'epsilon_end': config['epsilon_end'], 'epsilon_decay_frames': config['epsilon_decay_frames'],
            'network_type': config.get('network_type', 'cnn'), 'target_update_freq': config['target_update_freq'],
            'per_alpha': config.get('per_alpha'), 'per_beta_start': config.get('per_beta_start'),
            'per_beta_frames': config.get('per_beta_frames'),
            'aux_loss_weight_lines': config.get('aux_loss_weight_lines'),
            'aux_loss_weight_holes': config.get('aux_loss_weight_holes'),
            'aux_loss_weight_height': config.get('aux_loss_weight_height')
        })
        mlflow.log_artifact("config.yaml")

        reward_component_keys = [
            "tspin_reward", "tspin_mini_reward", "tspin_single_reward",
            "tspin_double_reward", "tspin_triple_reward", "line_clear_reward",
            "tetris_reward_bonus", "game_over_penalty"
        ]
        episode_total_rewards_deque = deque(maxlen=100)
        episode_lines_deque = deque(maxlen=100)
        episode_trigger_counts_deques = defaultdict(lambda: deque(maxlen=100))
        start_time = time.time()

        # --- 4. 训练主循环 ---
        num_episodes_this_session = config.get('num_episodes', 50000)
        for i in range(num_episodes_this_session): 
            current_absolute_episode = start_absolute_episode + i
            game.reset() 
            current_episode_total_reward = 0.0
            current_episode_lines = 0
            
            for step_in_episode in range(config.get('max_steps_per_episode', 3000)):
                if game.game_over: break

                possible_moves = game.get_all_possible_next_states_and_features()
                if not possible_moves: break

                agent.frames_done += 1 
                agent._decay_epsilon() 

                chosen_action_idx, chosen_move_data = agent.select_action(possible_moves)
                if chosen_move_data is None: break 
                
                action_info, s_prime_tensor, aux_labels = chosen_move_data
                
                reward_info_dict, game_over_after_action, lines_cleared = game.apply_ai_chosen_action(action_info)
                total_reward_this_step = reward_info_dict.get("final_reward", 0.0)

                current_episode_total_reward += total_reward_this_step
                current_episode_lines += lines_cleared
                
                next_best_q_value = 0.0
                if not game_over_after_action:
                    next_possible_moves = game.get_all_possible_next_states_and_features()
                    if next_possible_moves:
                         next_best_q_value = agent.get_max_q_value_for_next_states(next_possible_moves)
                
                agent.remember(s_prime_tensor, total_reward_this_step, next_best_q_value, game_over_after_action, aux_labels)
                agent.learn()
            
            # --- Episode End: Update stats ---
            episode_total_rewards_deque.append(current_episode_total_reward)
            episode_lines_deque.append(current_episode_lines)

            # --- Logging Period ---
            if current_absolute_episode % config['log_freq'] == 0:
                avg_total_reward_100 = np.mean(episode_total_rewards_deque) if episode_total_rewards_deque else 0.0
                avg_lines_100 = np.mean(episode_lines_deque) if episode_lines_deque else 0.0
                
                print(f"Ep: {current_absolute_episode}, Steps: {agent.frames_done}, Eps: {agent.epsilon:.4f}, "
                      f"AvgRew100: {avg_total_reward_100:.2f}, AvgLines100: {avg_lines_100:.2f}")

                mlflow.log_metrics({
                    'avg_reward_100eps': avg_total_reward_100,
                    'avg_lines_100eps': avg_lines_100,
                    'epsilon': agent.epsilon,
                    'per_beta': agent.memory.beta
                }, step=current_absolute_episode)

            # --- Periodic Model Save ---
            if current_absolute_episode > 0 and current_absolute_episode % config['save_model_freq'] == 0:
                model_path = agent.save_weights(current_absolute_episode)
                if model_path: mlflow.log_artifact(model_path, artifact_path="checkpoints")

            # --- Periodic Evaluation ---
            if current_absolute_episode > 0 and current_absolute_episode % config['eval_freq'] == 0:
                print(f"--- Running evaluation for Episode {current_absolute_episode} ---")
                eval_game = TetrisGame(config, render_mode=True) 
                eval_game.reset() 
                for eval_step in range(config.get('max_steps_per_episode', 3000)):
                    if eval_game.game_over: break 
                    eval_possible_moves = eval_game.get_all_possible_next_states_and_features()
                    if not eval_possible_moves: break
                    _idx, eval_chosen_move_data = agent.select_action(eval_possible_moves, is_eval_mode=True)
                    if eval_chosen_move_data is None: break
                    eval_game.apply_ai_chosen_action(eval_chosen_move_data[0])
                    eval_game.render()
                    pygame.time.wait(config.get('eval_step_wait_ms', 1))

                screenshot_path = os.path.join(config['screenshots_dir'], f"eval_episode_{current_absolute_episode}_final.png")
                eval_game.get_screenshot(screenshot_path) 
                mlflow.log_artifact(screenshot_path, artifact_path="evaluation_screenshots")
                mlflow.log_metric("eval_lines_cleared", eval_game.lines_cleared_total, step=current_absolute_episode)
                print(f"--- Evaluation finished. Score: {eval_game.score}, Lines: {eval_game.lines_cleared_total} ---")
                if eval_game.render_mode: pygame.display.quit()
                del eval_game

        # --- End of Training Session ---
        if num_episodes_this_session > 0:
            final_absolute_episode_trained = start_absolute_episode + num_episodes_this_session - 1
            print(f"Training session finished. Last trained absolute episode: {final_absolute_episode_trained}")
            final_model_path = agent.save_weights(final_absolute_episode_trained)
            if final_model_path: mlflow.log_artifact(final_model_path, artifact_path="final_model")

if __name__ == '__main__':
    train()