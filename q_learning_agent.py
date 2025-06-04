import random
import numpy as np
from collections import deque
import pickle

# Hyperparameters
MAX_MEM_SIZE = 100_000 # For experience replay (though simpler Q-learning might not need it for very small states)
BATCH_SIZE = 1000 # For batch training from memory (more relevant for DQN, but useful here for conceptual similarity)
LR = 0.1 # Learning rate for Q-learning
GAMMA = 0.9 # Discount factor

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0 # Initial exploration rate (starts high, decays)
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate for epsilon
        self.gamma = GAMMA # Discount factor
        self.q_table = {} # Our Q-table: {state_str: {action_idx: q_value}}
        self.memory = deque(maxlen=MAX_MEM_SIZE) # For experience replay (optional for basic tabular Q-learning but good practice)

        # Actions mapping: 0=straight, 1=right, 2=left
        self.action_space_size = 3

    # Helper to convert state array to a hashable string for Q-table key
    def _state_to_key(self, state):
        return str(state.tolist())

    def get_state(self, game):
        head = game.snake_position
        cell_size = game.cell_size

        # Points to check for danger relative to the head
        point_r = [head[0] + cell_size, head[1]]
        point_l = [head[0] - cell_size, head[1]]
        point_u = [head[0], head[1] - cell_size]
        point_d = [head[0], head[1] + cell_size]

        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right (relative to current direction)
            (dir_u and game.is_collision(point_r)) or # If moving UP, right turn leads to checking point_r
            (dir_d and game.is_collision(point_l)) or # If moving DOWN, right turn leads to checking point_l
            (dir_l and game.is_collision(point_u)) or # If moving LEFT, right turn leads to checking point_u
            (dir_r and game.is_collision(point_d)), # If moving RIGHT, right turn leads to checking point_d

            # Danger left (relative to current direction)
            (dir_u and game.is_collision(point_l)) or # If moving UP, left turn leads to checking point_l
            (dir_d and game.is_collision(point_r)) or # If moving DOWN, left turn leads to checking point_r
            (dir_r and game.is_collision(point_u)) or # If moving RIGHT, left turn leads to checking point_u
            (dir_l and game.is_collision(point_d)), # If moving LEFT, left turn leads to checking point_d

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location relative to snake head
            game.fruit_position[0] < head[0],  # food left
            game.fruit_position[0] > head[0],  # food right
            game.fruit_position[1] < head[1],  # food up
            game.fruit_position[1] > head[1]   # food down
        ]
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self._update_q_value(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self._update_q_value(state, action, reward, next_state, done)

    def _update_q_value(self, state, action, reward, next_state, done):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Initialize Q-values for new states if not already in table
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(self.action_space_size)}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {i: 0.0 for i in range(self.action_space_size)}

        action_idx = np.argmax(action) # Convert one-hot action to index

        old_q_value = self.q_table[state_key][action_idx]

        if done:
            new_q_value = reward # No future reward if game is over
        else:
            max_future_q = max(self.q_table[next_state_key].values())
            new_q_value = reward + self.gamma * max_future_q

        # Q-learning update rule
        self.q_table[state_key][action_idx] = old_q_value + self.epsilon * (new_q_value - old_q_value)
        # Note: Some Q-learning implementations use LR (alpha) here, others use epsilon for decay.
        # For simplicity and common practice, we'll use epsilon for exploration and learning rate for update.
        # Let's adjust the update rule to use LR:
        self.q_table[state_key][action_idx] = old_q_value + LR * (new_q_value - old_q_value)


    def get_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # Epsilon decay

        final_move = [0, 0, 0] # [straight, right, left]
        if random.random() < self.epsilon:
            # Exploration: Choose a random action
            move = random.randint(0, self.action_space_size - 1)
            final_move[move] = 1
        else:
            # Exploitation: Choose the best action from the Q-table
            state_key = self._state_to_key(state)
            if state_key not in self.q_table:
                # If state not seen before, initialize with zeros and pick random
                self.q_table[state_key] = {i: 0.0 for i in range(self.action_space_size)}
                move = random.randint(0, self.action_space_size - 1) # Fallback to random
            else:
                q_values_for_state = self.q_table[state_key]
                # Find the action with the highest Q-value
                move = max(q_values_for_state, key=q_values_for_state.get)

            final_move[move] = 1

        return final_move

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print(f"No Q-table found at {filename}. Starting with an empty table.")
            self.q_table = {}