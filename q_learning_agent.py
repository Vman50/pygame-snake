# q_learning_agent.py
import random
import numpy as np
from collections import deque
import pickle
import os # For checking file existence

# Import constants from the environment for state calculation
from snake_environment import CELL_SIZE, WINDOW_SIZE

# --- Hyperparameters ---
MAX_MEM_SIZE = 100_000 # Size of the experience replay buffer
BATCH_SIZE = 1000      # Number of experiences to sample for long-term training
LEARNING_RATE = 0.1    # Alpha (α) in Q-learning formula
GAMMA = 0.9            # Discount factor (γ) in Q-learning formula

class Agent:
    # Reverted __init__ to use a local q_table
    def __init__(self):
        self.n_games = 0 # Number of games played
        self.epsilon = 1.0 # Initial exploration rate (starts high)
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate for epsilon
        self.gamma = GAMMA # Discount factor

        self.q_table = {} # Our Q-table: {state_str: {action_idx: q_value}}
        self.memory = deque(maxlen=MAX_MEM_SIZE) # Experience replay buffer

        self.action_space_size = 3 # [straight, right, left]

    def _state_to_key(self, state_array):
        """Converts a NumPy state array to a hashable string for Q-table key."""
        return str(state_array.tolist())

    def get_state(self, game):
        """
        Extracts a simplified state representation from the game environment.
        This is crucial for keeping the Q-table manageable.
        """
        head = game.snake_position
        cell_size = game.cell_size
        body = game.snake_body  # Store snake body for easy access

        point_r = [head[0] + cell_size, head[1]]
        point_l = [head[0] - cell_size, head[1]]
        point_u = [head[0], head[1] - cell_size]
        point_d = [head[0], head[1] + cell_size]

        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        # Helper function to get just the boolean collision status
        def get_collision_status(point):
            collision, _ = game.is_collision(point) # Unpack, take only the boolean part
            return collision

        # Helper function to count nearby body segments
        def count_nearby_body_segments(point):
            count = 0
            if point in body[1:]:  # Exclude head
                count = 1  # Simple presence/absence
            return count

        state = [
            # Danger straight (is there a collision in the current forward direction?)
            (dir_r and get_collision_status(point_r)) or
            (dir_l and get_collision_status(point_l)) or
            (dir_u and get_collision_status(point_u)) or
            (dir_d and get_collision_status(point_d)),

            # Danger right (if I turn right, is there a collision?)
            (dir_u and get_collision_status(point_r)) or
            (dir_d and get_collision_status(point_l)) or
            (dir_l and get_collision_status(point_u)) or
            (dir_r and get_collision_status(point_d)),

            # Danger left (if I turn left, is there a collision?)
            (dir_u and get_collision_status(point_l)) or
            (dir_d and get_collision_status(point_r)) or
            (dir_r and get_collision_status(point_u)) or
            (dir_l and get_collision_status(point_d)),

            # Count nearby body segments in each direction
            count_nearby_body_segments(point_r if dir_r else (point_l if dir_l else (point_u if dir_u else point_d))),  # Straight
            count_nearby_body_segments(point_u if dir_r else (point_d if dir_l else (point_l if dir_u else point_r))),  # Right
            count_nearby_body_segments(point_l if dir_r else (point_r if dir_l else (point_d if dir_u else point_u))),  # Left

            # Current movement direction (one-hot encoded)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location relative to snake head
            game.fruit_position[0] < head[0],  # food left
            game.fruit_position[0] > head[0],  # food right
            game.fruit_position[1] < head[1],  # food up
            game.fruit_position[1] > head[0]   # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience (state, action, reward, next_state, done) in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """Performs a single Q-value update based on the most recent experience."""
        self._update_q_value(state, action, reward, next_state, done)

    def train_long_memory(self):
        """
        Performs Q-value updates on a batch of experiences sampled from memory.
        This is for experience replay, improving stability.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Randomly sample a batch
        else:
            mini_sample = self.memory # Use all memory if less than batch size

        for state, action, reward, next_state, done in mini_sample:
            self._update_q_value(state, action, reward, next_state, done)

    def _update_q_value(self, state, action, reward, next_state, done):
        """Applies the Q-learning update rule."""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Initialize Q-values for new states if not already in table
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(self.action_space_size)}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {i: 0.0 for i in range(self.action_space_size)}

        action_idx = np.argmax(action) # Convert one-hot action to index (0, 1, or 2)

        old_q_value = self.q_table[state_key][action_idx]

        if done:
            # If game is over, there are no future rewards from this state
            new_q_value = reward
        else:
            # Bellman equation: r + gamma * max(Q(s', a'))
            max_future_q = max(self.q_table[next_state_key].values())
            new_q_value = reward + self.gamma * max_future_q

        # Q-learning update: Q(s,a) = Q(s,a) + alpha * (new_estimate - old_estimate)
        self.q_table[state_key][action_idx] = old_q_value + LEARNING_RATE * (new_q_value - old_q_value)

    def get_action(self, state):
        """
        Decides the next action using an epsilon-greedy strategy.
        Explores randomly initially, then exploits learned Q-values.
        """
        # Epsilon decay: Decrease exploration rate over games
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        final_move = [0, 0, 0] # [straight, right, left]

        if random.random() < self.epsilon:
            # Exploration: Choose a random action
            move_idx = random.randint(0, self.action_space_size - 1)
            final_move[move_idx] = 1
        else:
            # Exploitation: Choose the best action from the Q-table
            state_key = self._state_to_key(state)
            if state_key not in self.q_table:
                # If this state has never been seen, initialize Q-values and pick random
                self.q_table[state_key] = {i: 0.0 for i in range(self.action_space_size)}
                move_idx = random.randint(0, self.action_space_size - 1) # Fallback to random
            else:
                q_values_for_state = self.q_table[state_key]
                # Find the action with the highest Q-value
                move_idx = max(q_values_for_state, key=q_values_for_state.get)

            final_move[move_idx] = 1

        return final_move

    def save_q_table(self, filename="q_table.pkl"):
        """Saves the Q-table to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.pkl"):
        """Loads the Q-table from a file."""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Q-table loaded from {filename}. Size: {len(self.q_table)}")
            except Exception as e:
                print(f"Error loading Q-table from {filename}: {e}. Starting with an empty table.")
                self.q_table = {}
        else:
            print(f"No Q-table found at {filename}. Starting with an empty table.")
            self.q_table = {}