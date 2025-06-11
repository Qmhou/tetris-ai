# replay_buffer.py
import numpy as np
import random

class SumTree:
    """
    A SumTree data structure for efficient priority-based sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx, change):
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def get(self, s):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class PERBuffer:
    """
    Prioritized Experience Replay Buffer that uses a SumTree.
    """
    def __init__(self, capacity, per_epsilon, per_alpha, per_beta_start, per_beta_frames):
        self.tree = SumTree(capacity)
        self.epsilon = per_epsilon  # Small value to ensure no experience has zero priority
        self.alpha = per_alpha      # Controls how much prioritization is used (0: random, 1: full priority)
        self.beta = per_beta_start  # Importance-sampling exponent, anneals to 1
        self.beta_increment_per_sampling = (1.0 - per_beta_start) / per_beta_frames
        self.max_priority = 1.0

    def add(self, experience):
        # Store new experience with max priority to ensure it gets sampled
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        batch_indices = np.empty(batch_size, dtype=np.int32)
        batch_data = np.empty(batch_size, dtype=object)
        is_weights = np.empty(batch_size, dtype=np.float32)

        segment = self.tree.total_priority / batch_size

        # Anneal beta
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, p, data = self.tree.get(s)

            sampling_prob = p / self.tree.total_priority
            is_weights[i] = np.power(self.tree.n_entries * sampling_prob, -self.beta)

            batch_indices[i] = idx
            batch_data[i] = data

        # Normalize weights for stability
        is_weights /= is_weights.max()

        return batch_indices, batch_data, is_weights

    def update_priorities(self, tree_indices, td_errors):
        priorities = np.abs(td_errors) + self.epsilon
        clipped_priorities = np.minimum(priorities, self.max_priority)

        powered_priorities = np.power(clipped_priorities, self.alpha)

        for ti, p in zip(tree_indices, powered_priorities):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.n_entries