import random
import numpy as np
from collections import deque
import torch
import torch.optim as optim
import torch.nn as nn
from dqn_model import DQN

MAX_MEM_SIZE = 100_000
BATCH_SIZE = 512
LEARNING_RATE = 0.0005
GAMMA = 0.99

class Agent:
    def __init__(self, state_size, action_size):
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.gamma = GAMMA

        self.memory = deque(maxlen=MAX_MEM_SIZE)
        self.state_size = state_size
        self.action_size = action_size

        self.model = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def get_state(self, game):
        grid = np.zeros((game.board_cells, game.board_cells), dtype=np.float32)
        for segment in game.snake_body:
            x, y = segment[0] // game.cell_size, segment[1] // game.cell_size
            if 0 <= x < game.board_cells and 0 <= y < game.board_cells:
                grid[y, x] = 1.0  # Mark snake body
        fx, fy = game.fruit_position[0] // game.cell_size, game.fruit_position[1] // game.cell_size
        if 0 <= fx < game.board_cells and 0 <= fy < game.board_cells:
            grid[fy, fx] = 2.0  # Mark fruit

        flat_grid = grid.flatten()

        head_x, head_y = game.snake_position[0], game.snake_position[1]
        fruit_dir = [
            int(game.fruit_position[0] < head_x),
            int(game.fruit_position[0] > head_x),
            int(game.fruit_position[1] < head_y),
            int(game.fruit_position[1] > head_y),
        ]
        fruit_coords = [
            game.fruit_position[0] / game.w,
            game.fruit_position[1] / game.h
        ]

        # Distance to walls (normalized)
        dist_wall_left = head_x / (game.w - game.cell_size)
        dist_wall_right = (game.w - game.cell_size - head_x) / (game.w - game.cell_size)
        dist_wall_up = head_y / (game.h - game.cell_size)
        dist_wall_down = (game.h - game.cell_size - head_y) / (game.h - game.cell_size)

        # Wall presence (one-hot)
        at_left_wall = int(head_x == 0)
        at_right_wall = int(head_x == game.w - game.cell_size)
        at_top_wall = int(head_y == 0)
        at_bottom_wall = int(head_y == game.h - game.cell_size)

        wall_info = [
            dist_wall_left, dist_wall_right, dist_wall_up, dist_wall_down,
            at_left_wall, at_right_wall, at_top_wall, at_bottom_wall
        ]

        # --- Add direction as one-hot ---
        direction = [
            int(game.direction == 'LEFT'),
            int(game.direction == 'RIGHT'),
            int(game.direction == 'UP'),
            int(game.direction == 'DOWN')
        ]

        # --- Add normalized snake length ---
        max_length = game.board_cells * game.board_cells
        norm_length = len(game.snake_body) / max_length

        # --- Combine all state info ---
        state = np.concatenate([flat_grid, fruit_dir, fruit_coords, wall_info, direction, [norm_length]])
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            mini_sample = random.sample(self.memory, len(self.memory))
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self._train_step([state], [action], [reward], [next_state], [done])

    def _train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor([np.argmax(a) for a in actions], dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        pred = self.model(states)
        target = pred.clone().detach()

        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            target[idx][actions[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            move = random.randint(0, self.action_size - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        final_move = [0] * self.action_size
        final_move[move] = 1
        return final_move

    def save_model(self, filename="dqn_model.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="dqn_model.pth"):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        print(f"Model loaded from {filename}")