# train.py
# (版本：最终修复版，包含T-Spin日志和所有功能)

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
    
    # --- 1. 初始化 ---
    os.makedirs(config.get('weights_dir', 'weights/'), exist_ok=True)
    os.makedirs(config.get('screenshots_dir', 'screenshots/'), exist_ok=True)
    os.makedirs(config.get('logs_dir', 'logs/'), exist_ok=True)
    
    game = TetrisGame(config, render_mode=False) 
    agent = DQNAgent(config)

    # --- 2. 加载模型 ---
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
    
    if path_to_attempt_load:
        load_success, next_episode_to_start_from = agent.load_weights(path_to_attempt_load)
        if load_success:
            start_absolute_episode = next_episode_to_start_from
            print(f"将从绝对Episode {start_absolute_episode} 继续训练。")
    else:
        print("未指定或未找到可加载的模型。将从Episode 1开始新训练。")

    # --- 3. MLflow 和日志初始化 ---
    mlflow.set_experiment("Tetris CNN T-Spin Training")

    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run-ID: {run_id}")

        mlflow.log_params(config)
        mlflow.log_artifact("config.yaml")

        # --- 使用T-Spin的正确reward_keys ---
        reward_component_keys = [
            "tspin_reward", "tspin_mini_reward", "tspin_single_reward",
            "tspin_double_reward", "tspin_triple_reward", "line_clear_reward",
            "tetris_reward_bonus", "game_over_penalty"
        ]

        episode_total_rewards_deque = deque(maxlen=100)
        episode_lines_deque = deque(maxlen=100)
        episode_steps_deque = deque(maxlen=100) # Add a deque to store steps per episode
        episode_trigger_counts_deques = defaultdict(lambda: deque(maxlen=100))
        start_time = time.time()

        # --- 4. 训练主循环 ---
        num_episodes_this_session = config.get('num_episodes', 50000)
        for i in range(num_episodes_this_session): 
            current_absolute_episode = start_absolute_episode + i
            game.reset() 
            current_episode_total_reward = 0.0
            current_episode_lines = 0
            current_episode_steps = 0 # Initialize step counter for the new episode
            current_episode_trigger_counts = defaultdict(int)
            total_loss, value_loss, lines_loss, holes_loss, height_loss = None, None, None, None, None

            for step_in_episode in range(config.get('max_steps_per_episode', 3000)):
                if game.game_over: break

                possible_moves = game.get_all_possible_next_states_and_features()
                if not possible_moves: break

                agent.frames_done += 1 
                agent._decay_epsilon() 
                current_episode_steps += 1 # Increment the step counter for the current episode

                chosen_action_idx, chosen_move_data = agent.select_action(possible_moves)
                if chosen_move_data is None: break 
                
                action_info, s_prime_tensor, aux_labels = chosen_move_data
                
                reward_info_dict, game_over_after_action, lines_cleared = game.apply_ai_chosen_action(action_info)
                total_reward_this_step = reward_info_dict.get("final_reward", 0.0)

                current_episode_total_reward += total_reward_this_step
                current_episode_lines += lines_cleared
                for key in reward_component_keys:
                    if reward_info_dict.get(key, 0.0) != 0:
                        current_episode_trigger_counts[key] += 1
                
                next_best_q_value = 0.0
                if not game_over_after_action:
                    next_possible_moves = game.get_all_possible_next_states_and_features()
                    if next_possible_moves:
                         next_best_q_value = agent.get_max_q_value_for_next_states(next_possible_moves)
                
                agent.remember(s_prime_tensor, total_reward_this_step, next_best_q_value, game_over_after_action, aux_labels)
                
                learn_result = agent.learn()
                if learn_result:
                    total_loss, value_loss, lines_loss, holes_loss, height_loss = learn_result


            # --- Episode End: Update stats deques ---
            episode_total_rewards_deque.append(current_episode_total_reward)
            episode_lines_deque.append(current_episode_lines)
            episode_steps_deque.append(current_episode_steps) # Add current episode's steps to the deque
            for key in reward_component_keys:
                episode_trigger_counts_deques[key].append(current_episode_trigger_counts[key])


            for key in reward_component_keys:
                episode_trigger_counts_deques[key].append(current_episode_trigger_counts[key])


            # --- Logging Period ---
            if current_absolute_episode > 0 and current_absolute_episode % config['log_freq'] == 0:
                avg_total_reward_100 = np.mean(episode_total_rewards_deque) if episode_total_rewards_deque else 0.0
                avg_lines_100 = np.mean(episode_lines_deque) if episode_lines_deque else 0.0
                avg_steps_100 = np.mean(episode_steps_deque) if episode_steps_deque else 0.0 # Calculate average steps
                
                # --- Console Log ---
                tsd_count = sum(episode_trigger_counts_deques['tspin_double_reward'])
                tst_count = sum(episode_trigger_counts_deques['tspin_triple_reward'])
                print(f"Ep: {current_absolute_episode}, Steps: {agent.frames_done}, Eps: {agent.epsilon:.4f}, "
                      f"AvgRew100: {avg_total_reward_100:.2f}, AvgLines100: {avg_lines_100:.2f}, AvgSteps100: {avg_steps_100:.2f}, " # Add to console log
                      f"Loss: {total_loss or 0.0:.4f}, TSD_Count100: {tsd_count}, TST_Count100: {tst_count}")

                # --- MLflow Log ---
                mlflow_metrics = {
                    'avg_reward_100eps': avg_total_reward_100,
                    'avg_lines_100eps': avg_lines_100,
                    'avg_steps_100eps': avg_steps_100, # Add to mlflow metrics
                    'epsilon': agent.epsilon,
                    'per_beta': agent.memory.beta,
                    'loss_total': total_loss,
                    'loss_value': value_loss,
                    'loss_aux_lines': lines_loss,
                    'loss_aux_holes': holes_loss,
                    'loss_aux_height': height_loss
                }
                for key in reward_component_keys:
                     mlflow_metrics[f'count_{key}_100eps'] = sum(episode_trigger_counts_deques[key])
                mlflow.log_metrics({k: v for k, v in mlflow_metrics.items() if v is not None}, step=current_absolute_episode)
                
                # CSV logging is getting complex, focusing on console and mlflow for now.
                # You can add the new metric to the CSV logic if needed.

            if current_absolute_episode > 0 and current_absolute_episode % config['save_model_freq'] == 0:
                model_path = agent.save_weights(current_absolute_episode)
                if model_path: mlflow.log_artifact(model_path, artifact_path="checkpoints")

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

        if num_episodes_this_session > 0:
            final_episode = start_absolute_episode + num_episodes_this_session - 1
            print(f"Training session finished. Last trained episode: {final_episode}")
            final_model_path = agent.save_weights(final_episode)
            if final_model_path: mlflow.log_artifact(final_model_path, artifact_path="final_model")

if __name__ == '__main__':
    train()