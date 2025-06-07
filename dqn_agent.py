# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class QNetwork(nn.Module):
    # ... (QNetwork class definition remains the same) ...
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dims
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dims))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    # ... (__init__, _decay_epsilon, select_action, remember, learn, get_max_q_value_for_next_states methods remain largely the same) ...
    def __init__(self, input_dims, hidden_dims, output_dims, lr, gamma, epsilon_start, epsilon_end, epsilon_decay_frames, memory_size, batch_size, target_update_freq, weights_dir="weights/", device=None):
        self.input_dims = input_dims
        self.output_dims = output_dims 
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start # This will be overwritten if loading weights
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.policy_net = QNetwork(input_dims, hidden_dims, output_dims).to(self.device)
        self.target_net = QNetwork(input_dims, hidden_dims, output_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = deque(maxlen=memory_size)
        self.frames_done = 0 # This will be overwritten if loading weights

    def _decay_epsilon(self):
        if self.frames_done < self.epsilon_decay_frames:
             self.epsilon = self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** (self.frames_done / self.epsilon_decay_frames)
        else:
            self.epsilon = self.epsilon_end
            
    def select_action(self, possible_moves_features_list, is_eval_mode=False):
        # No change to frames_done increment here, it's an overall step counter
        # self.frames_done += 1 # This should be incremented per environment step, not per action selection call
        if not is_eval_mode: # Epsilon decay happens based on actual environment steps
            pass # Epsilon decay is handled by agent step or training loop

        if not possible_moves_features_list: 
            return None, None 

        if not is_eval_mode and random.random() < self.epsilon:
            chosen_action_index = random.randrange(len(possible_moves_features_list))
            return chosen_action_index, possible_moves_features_list[chosen_action_index]
        else:
            with torch.no_grad():
                feature_tensors = [torch.FloatTensor(move_data[1]).unsqueeze(0).to(self.device) for move_data in possible_moves_features_list]
                q_values = [self.policy_net(tensor).item() for tensor in feature_tensors]
                chosen_action_index = np.argmax(q_values)
                return chosen_action_index, possible_moves_features_list[chosen_action_index]

    def remember(self, state_features, action_index, reward, next_best_q_value, done):
        self.memory.append((state_features, reward, next_best_q_value, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None 

        batch = random.sample(self.memory, self.batch_size)
        states_features, rewards, next_states_best_q, dones = zip(*batch)

        states_features_tensor = torch.FloatTensor(np.array(states_features)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_best_q_tensor = torch.FloatTensor(next_states_best_q).unsqueeze(1).to(self.device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(states_features_tensor)
        target_q_values = rewards_tensor + (self.gamma * next_states_best_q_tensor * (~dones_tensor))

        loss = F.mse_loss(current_q_values, target_q_values.detach()) 

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()

        if self.frames_done % self.target_update_freq == 0: # frames_done should be updated in training loop
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def get_max_q_value_for_next_states(self, possible_next_moves_features_list):
        if not possible_next_moves_features_list: 
            return 0.0 
        with torch.no_grad():
            feature_tensors = [torch.FloatTensor(move_data[1]).unsqueeze(0).to(self.device) for move_data in possible_next_moves_features_list]
            q_values = [self.target_net(tensor).item() for tensor in feature_tensors]
            return np.max(q_values) if q_values else 0.0

    def save_weights(self, absolute_episode_num): # Renamed parameter for clarity
        """Saves model weights and training state."""
        path = os.path.join(self.weights_dir, f"dqn_tetris_episode_{absolute_episode_num}.pth")
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'frames_done': self.frames_done,
            'episode_num': absolute_episode_num  # Save the absolute episode number checkpointed
        }, path)
        print(f"Saved weights for absolute episode {absolute_episode_num} to {path}")

    def load_weights(self, path):
        """
        Loads model weights and training state from a file.
        Returns:
            tuple: (bool: success, int: next_episode_to_start)
        """
        if not os.path.exists(path):
            print(f"警告: 权重文件 {path} 未找到。")
            return False, 1 # success_flag, default_next_episode

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint: # For backward compatibility
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("警告: 检查点中未找到优化器状态。优化器未加载。")

            self.epsilon = checkpoint.get('epsilon', self.epsilon_start) # Use current if not saved
            self.frames_done = checkpoint.get('frames_done', 0)
            # Get the episode number at which this checkpoint was saved
            loaded_episode_num_checkpointed = checkpoint.get('episode_num', 0) 
            
            self.policy_net.to(self.device) # Ensure model is on the correct device
            self.target_net.to(self.device)
            self.target_net.eval()
            
            print(f"成功从 {path} 加载权重。")
            print(f"  当前 Epsilon: {self.epsilon:.4f}, 已完成帧数 (Frames_done): {self.frames_done}")
            print(f"  权重保存于 Episode: {loaded_episode_num_checkpointed}")
            return True, loaded_episode_num_checkpointed + 1 # Return success and NEXT episode to start training
        except Exception as e:
            print(f"加载权重时发生错误 {path}: {e}")
            return False, 1