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
    """
    A Convolutional Neural Network to evaluate the value of Tetris board states.
    It takes a multi-channel 2D representation of the board as input.
    """
    def __init__(self, input_channels):
        super(QNetwork, self).__init__()
        
        # Convolutional layers to process the board "image"
        self.conv_layers = nn.Sequential(
            # Input shape: [N, input_channels, 20, 10]
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Flattened size calculation assumes input HxW is 20x10.
        # Conv layers with padding=1 and stride=1 preserve the dimensions.
        flattened_size = 64 * 20 * 10
        
        # Fully connected layers to produce the final state value (V-value)
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Output a single value V(s')
        )

    def forward(self, state):
        # The state is expected to be a tensor of shape [N, C, H, W]
        x = self.conv_layers(state)
        x = x.view(x.size(0), -1) # Flatten the tensor
        q_value = self.fc_layers(x)
        return q_value

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
        self.memory = deque(maxlen=config['memory_size'])
        
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
        In eval mode, it's purely greedy. For CNN, it uses batch prediction for efficiency.
        """
        if not possible_moves_list: 
            return None, None 

        if not is_eval_mode and random.random() < self.epsilon:
            # Exploration: choose a random action
            chosen_action_index = random.randrange(len(possible_moves_list))
            return chosen_action_index, possible_moves_list[chosen_action_index]
        else:
            # Exploitation: choose the best action
            with torch.no_grad():
                # Extract tensors, batch them, and evaluate
                state_tensors = [move[1] for move in possible_moves_list]
                batch_tensor = torch.cat(state_tensors, dim=0).to(self.device)
                q_values = self.policy_net(batch_tensor).squeeze().cpu().numpy()
                
                # If there's only one move, q_values might be a float, not an array
                if q_values.ndim == 0:
                    chosen_action_index = 0
                else:
                    chosen_action_index = np.argmax(q_values)
                
                return chosen_action_index, possible_moves_list[chosen_action_index]

    def remember(self, state_tensor, reward, next_best_q_value, done):
        """Stores an experience in the replay buffer. The state is now a CNN input tensor."""
        self.memory.append((state_tensor, reward, next_best_q_value, done))

    def learn(self):
        """
        Samples a batch from the replay buffer and performs a Q-learning update.
        """
        if len(self.memory) < self.batch_size:
            return None 

        batch = random.sample(self.memory, self.batch_size)
        # Unzip the batch. 'states' will be a tuple of tensors.
        states, rewards, next_states_best_q, dones = zip(*batch)

        # Batch states, which are now tensors, by concatenating them
        states_tensor = torch.cat(states, dim=0).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_best_q_tensor = torch.FloatTensor(next_states_best_q).unsqueeze(1).to(self.device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q-values for the states in the batch
        current_q_values = self.policy_net(states_tensor)
        
        # Calculate the target Q-values using the Bellman equation
        target_q_values = rewards_tensor + (self.gamma * next_states_best_q_tensor * (~dones_tensor))

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values.detach()) 

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()

        # Update the target network periodically
        if self.frames_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def get_max_q_value_for_next_states(self, possible_next_moves_list):
        """[CNN Version] Evaluates all possible next states in a batch and returns the maximum Q-value."""
        if not possible_next_moves_list:
            return 0.0

        with torch.no_grad():
            next_state_tensors = [move[1] for move in possible_next_moves_list]
            batch_tensor = torch.cat(next_state_tensors, dim=0).to(self.device)
            q_values = self.target_net(batch_tensor)
            max_q_value = torch.max(q_values).item()
            return max_q_value

    def get_best_action_and_value(self, possible_moves_list):
        """[CNN Version] Evaluates all possible moves and returns the best action tuple and its V-value."""
        if not possible_moves_list:
            return None, -float('inf') 
            
        with torch.no_grad():
            state_tensors = [move[1] for move in possible_moves_list]
            batch_tensor = torch.cat(state_tensors, dim=0).to(self.device)
            q_values = self.policy_net(batch_tensor).squeeze().cpu().numpy()

            if q_values.ndim == 0:
                best_idx = 0
            else:
                best_idx = np.argmax(q_values)
            
            # Ensure q_values is treated as an array for consistent indexing
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