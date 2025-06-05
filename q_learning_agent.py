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
        head = game.snake_position
        cell_size = game.cell_size
        body = game.snake_body

        point_r = [head[0] + cell_size, head[1]]
        point_l = [head[0] - cell_size, head[1]]
        point_u = [head[0], head[1] - cell_size]
        point_d = [head[0], head[1] + cell_size]

        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        def get_collision_status(point):
            collision, _ = game.is_collision(point)
            return collision

        def count_nearby_body_segments(point):
            return int(point in body[1:])  # Exclude head

        dist_wall_left = head[0] / (game.w - cell_size)
        dist_wall_right = (game.w - cell_size - head[0]) / (game.w - cell_size)
        dist_wall_up = head[1] / (game.h - cell_size)
        dist_wall_down = (game.h - cell_size - head[1]) / (game.h - cell_size)

        def min_body_dist(dx, dy):
            min_dist = 1.0
            for segment in body[1:]:
                if dx != 0 and segment[1] == head[1]:
                    dist = abs(segment[0] - head[0])
                    if (dx > 0 and segment[0] > head[0]) or (dx < 0 and segment[0] < head[0]):
                        min_dist = min(min_dist, dist / (game.w - cell_size))
                if dy != 0 and segment[0] == head[0]:
                    dist = abs(segment[1] - head[1])
                    if (dy > 0 and segment[1] > head[1]) or (dy < 0 and segment[1] < head[1]):
                        min_dist = min(min_dist, dist / (game.h - cell_size))
            return min_dist

        dist_body_right = min_body_dist(1, 0)
        dist_body_left = min_body_dist(-1, 0)
        dist_body_up = min_body_dist(0, -1)
        dist_body_down = min_body_dist(0, 1)

        state = [
            (dir_r and get_collision_status(point_r)) or
            (dir_l and get_collision_status(point_l)) or
            (dir_u and get_collision_status(point_u)) or
            (dir_d and get_collision_status(point_d)),

            (dir_u and get_collision_status(point_r)) or
            (dir_d and get_collision_status(point_l)) or
            (dir_l and get_collision_status(point_u)) or
            (dir_r and get_collision_status(point_d)),

            (dir_u and get_collision_status(point_l)) or
            (dir_d and get_collision_status(point_r)) or
            (dir_r and get_collision_status(point_u)) or
            (dir_l and get_collision_status(point_d)),

            count_nearby_body_segments(point_r if dir_r else (point_l if dir_l else (point_u if dir_u else point_d))),
            count_nearby_body_segments(point_u if dir_r else (point_d if dir_l else (point_l if dir_u else point_r))),
            count_nearby_body_segments(point_l if dir_r else (point_r if dir_l else (point_d if dir_u else point_u))),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.fruit_position[0] < head[0],
            game.fruit_position[0] > head[0],
            game.fruit_position[1] < head[1],
            game.fruit_position[1] > head[1],

            dist_wall_left,
            dist_wall_right,
            dist_wall_up,
            dist_wall_down,

            dist_body_left,
            dist_body_right,
            dist_body_up,
            dist_body_down,
        ]
        return np.array(state, dtype=float)

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