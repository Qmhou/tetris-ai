# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from replay_buffer import PERBuffer


class QNetwork(nn.Module):
    """
    A Convolutional Neural Network to evaluate the value of Tetris board states.
    It takes a multi-channel 2D representation of the board as input.
    """
    """
    A multi-headed CNN. It predicts the main V-value and several auxiliary targets.
    """
    def __init__(self, input_channels):
        super(QNetwork, self).__init__()
        
        # --- Shared Convolutional Body ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        flattened_size = 64 * 20 * 10
        
        # --- Multiple Output Heads ---
        # 1. Main Value Head (predicts V-value)
        self.value_head = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # 2. Auxiliary Head for Completed Lines (Classification: 0, 1, 2, 3, or 4 lines)
        self.lines_head = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 5) # 5 classes for 0, 1, 2, 3, 4 lines
        )
        
        # 3. Auxiliary Head for Hole Count (Regression)
        self.holes_head = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Predict a single value for hole count
        )

        # 4. Auxiliary Head for Aggregate Height (Regression)
        self.height_head = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Predict a single value for aggregate height
        )

    def forward(self, state):
        # Pass through the shared convolutional body
        shared_features = self.conv_layers(state)
        shared_features = shared_features.view(shared_features.size(0), -1) # Flatten

        # Get outputs from all heads
        value = self.value_head(shared_features)
        lines = self.lines_head(shared_features)
        holes = self.holes_head(shared_features)
        height = self.height_head(shared_features)
        
        return value, lines, holes, height
    
class DQNAgent:
    def __init__(self, config, device=None):
        self.config = config
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.weights_dir = config['weights_dir']
        os.makedirs(self.weights_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Initialize networks based on config
        input_channels = config['cnn_input_channels']
        self.policy_net = QNetwork(input_channels).to(self.device)
        self.target_net = QNetwork(input_channels).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.memory = PERBuffer(
            capacity=config['memory_size'],
            per_epsilon=config['per_epsilon'],
            per_alpha=config['per_alpha'],
            per_beta_start=config['per_beta_start'],
            per_beta_frames=config['per_beta_frames']
        )
        
        self.frames_done = 0
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay_frames = config['epsilon_decay_frames']
        self.epsilon = self.epsilon_start

    def _decay_epsilon(self):
        """Linearly decays epsilon based on the number of frames done."""
        if self.frames_done < self.epsilon_decay_frames:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.frames_done / self.epsilon_decay_frames)
        else:
            self.epsilon = self.epsilon_end
            
    def select_action(self, possible_moves_list, is_eval_mode=False):
        """
        Selects an action using an epsilon-greedy policy.
        Correctly handles the multi-headed network output.
        """
        if not possible_moves_list: 
            return None, None 

        if not is_eval_mode and random.random() < self.epsilon:
            chosen_action_index = random.randrange(len(possible_moves_list))
            return chosen_action_index, possible_moves_list[chosen_action_index]
        else:
            with torch.no_grad():
                state_tensors = [move[1] for move in possible_moves_list]
                batch_tensor = torch.cat(state_tensors, dim=0).to(self.device)
                
                # Unpack the tuple from the network, we only need the value predictions here.
                value_predictions, _, _, _ = self.policy_net(batch_tensor)
                q_values = value_predictions.squeeze().cpu().numpy()
                
                if q_values.ndim == 0:
                    chosen_action_index = 0
                else:
                    chosen_action_index = np.argmax(q_values)
                
                return chosen_action_index, possible_moves_list[chosen_action_index]


    def remember(self, state_tensor, reward, next_best_q_value, done, aux_labels):
        """Stores an experience, now including auxiliary task labels."""
        experience = (state_tensor, reward, next_best_q_value, done, aux_labels)
        self.memory.add(experience)

    def learn(self):
        """
        Samples a batch, calculates a composite loss (Value + Auxiliaries),
        and updates the network.
        """
        if len(self.memory) < self.batch_size:
            return None 

        tree_indices, batch_data, is_weights = self.memory.sample(self.batch_size)
        
        # Unpack batch data, now including auxiliary labels
        states, rewards, next_qs, dones, aux_labels_list = zip(*batch_data)

        # --- Prepare Tensors ---
        states_tensor = torch.cat(states, dim=0).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_qs_tensor = torch.FloatTensor(next_qs).unsqueeze(1).to(self.device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        is_weights_tensor = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
        
        # Prepare auxiliary label tensors
        true_lines = torch.LongTensor([l['lines'] for l in aux_labels_list]).to(self.device)
        true_holes = torch.FloatTensor([l['holes'] for l in aux_labels_list]).unsqueeze(1).to(self.device)
        true_height = torch.FloatTensor([l['height'] for l in aux_labels_list]).unsqueeze(1).to(self.device)
        
        # --- Forward Pass ---
        pred_value, pred_lines, pred_holes, pred_height = self.policy_net(states_tensor)

        # --- Calculate Composite Loss ---
        # 1. Main Value Loss (MSE, weighted by PER)
        target_value = rewards_tensor + (self.gamma * next_qs_tensor * (~dones_tensor))
        td_errors = (target_value - pred_value).detach()
        value_loss = (is_weights_tensor * F.mse_loss(pred_value, target_value.detach(), reduction='none')).mean()
        
        # 2. Auxiliary Losses
        lines_loss = F.cross_entropy(pred_lines, true_lines)
        holes_loss = F.mse_loss(pred_holes, true_holes)
        height_loss = F.mse_loss(pred_height, true_height)

        # 3. Total Loss
        total_loss = value_loss + \
                     self.config['aux_loss_weight_lines'] * lines_loss + \
                     self.config['aux_loss_weight_holes'] * holes_loss + \
                     self.config['aux_loss_weight_height'] * height_loss

        # --- Optimization & Priority Update ---
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()
        
        self.memory.update_priorities(tree_indices, td_errors.squeeze().cpu().numpy())

        if self.frames_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return total_loss.item(), value_loss.item(), lines_loss.item(), holes_loss.item(), height_loss.item()

    def get_max_q_value_for_next_states(self, possible_next_moves_list):
        """[Multi-head fix] Evaluates all possible next states and returns the maximum V-value."""
        if not possible_next_moves_list:
            return 0.0

        with torch.no_grad():
            next_state_tensors = [move[1] for move in possible_next_moves_list]
            batch_tensor = torch.cat(next_state_tensors, dim=0).to(self.device)
            
            # Unpack the tuple from the target network, we only need the value predictions.
            value_predictions, _, _, _ = self.target_net(batch_tensor)
            
            # Find the max value from the value_predictions tensor.
            max_q_value = torch.max(value_predictions).item()
            return max_q_value

    def get_best_action_and_value(self, possible_moves_list):
        """[Multi-head fix] Evaluates all moves and returns the best action and its V-value."""
        if not possible_moves_list:
            return None, -float('inf') 
            
        with torch.no_grad():
            state_tensors = [move[1] for move in possible_moves_list]
            batch_tensor = torch.cat(state_tensors, dim=0).to(self.device)

            # Unpack the tuple from the policy network.
            value_predictions, _, _, _ = self.policy_net(batch_tensor)
            q_values = value_predictions.squeeze().cpu().numpy()

            if q_values.ndim == 0:
                best_idx = 0
            else:
                best_idx = np.argmax(q_values)
            
            q_values_array = np.atleast_1d(q_values)
            
            return possible_moves_list[best_idx], q_values_array[best_idx]

    def save_weights(self, absolute_episode_num):
        """Saves model weights and training state, and returns the path."""
        path = os.path.join(self.weights_dir, f"dqn_tetris_episode_{absolute_episode_num}.pth")
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'frames_done': self.frames_done,
            'episode_num': absolute_episode_num
        }, path)
        print(f"Saved weights for absolute episode {absolute_episode_num} to {path}")
        return path # Return the path of the saved file


    def load_weights(self, path):
        """Loads model weights and training state from a file."""
        if not os.path.exists(path):
            print(f"Warning: Weight file {path} not found.")
            return False, 1

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            self.frames_done = checkpoint.get('frames_done', 0)
            loaded_episode_num = checkpoint.get('episode_num', 0) 
            
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            self.target_net.eval()
            
            print(f"Successfully loaded weights from {path}.")
            print(f"  - Epsilon set to: {self.epsilon:.4f}")
            print(f"  - Frames done: {self.frames_done}")
            print(f"  - Checkpoint saved at episode: {loaded_episode_num}")
            return True, loaded_episode_num + 1 # Return success and the NEXT episode to start training
        except Exception as e:
            print(f"Error loading weights from {path}: {e}")
            return False, 1