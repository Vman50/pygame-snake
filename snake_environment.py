# snake_environment.py
import pygame
import random
import numpy as np
import time # Import the time module

# --- Game Constants ---
BOARD_CELLS = 10 # Assuming you updated this to 10
WINDOW_SIZE = 800 - (800 % BOARD_CELLS)
CELL_SIZE = WINDOW_SIZE // BOARD_CELLS

# Colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
LIGHT_GREEN = pygame.Color(170, 215, 81)
DARK_GREEN = pygame.Color(162, 209, 73)

class SnakeGameAI:
    def __init__(self, w=WINDOW_SIZE, h=WINDOW_SIZE, speed=200, render=True):
        self.w = w
        self.h = h
        self.cell_size = CELL_SIZE
        self.board_cells = BOARD_CELLS
        self.speed = speed
        self.render = render

        pygame.init()
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI Environment')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None
        self.reset()

    def reset(self):
        self.snake_position = [5 * self.cell_size, 5 * self.cell_size]
        self.snake_body = [
            list(self.snake_position),
            [4 * self.cell_size, 5 * self.cell_size],
            [3 * self.cell_size, 5 * self.cell_size],
            [2 * self.cell_size, 5 * self.cell_size]
        ]
        self.snake_body_set = set(tuple(pos) for pos in self.snake_body)
        self.fruit_position = self._place_fruit()
        while tuple(self.fruit_position) in self.snake_body_set:
            self.fruit_position = self._place_fruit()

        self.direction = 'RIGHT'
        self.score = 0
        self.frame_iteration = 0
        self.game_over_flag = False
        self.start_time = time.time() # Record the start time of the game

    def _place_fruit(self):
        x = random.randrange(0, self.board_cells) * self.cell_size
        y = random.randrange(0, self.board_cells) * self.cell_size
        return [x, y]

    def _update_ui(self):
        if not self.render:
            return
        for row in range(self.board_cells):
            for col in range(self.board_cells):
                color = LIGHT_GREEN if (row + col) % 2 == 0 else DARK_GREEN
                pygame.draw.rect(
                    self.display,
                    color,
                    pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                )
        for pos in self.snake_body:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pos[0], pos[1], self.cell_size, self.cell_size))
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.fruit_position[0], self.fruit_position[1], self.cell_size, self.cell_size))
        self.show_score(1, WHITE, 'times new roman', 20)
        pygame.display.flip()

    def show_score(self, choice, color, font, size):
        if not self.render:
            return
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        self.display.blit(score_surface, score_rect)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake_position
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True, "wall"
        if tuple(pt) in self.snake_body_set and pt != self.snake_body[0]:
            return True, "self"
        return False, None

    def play_step(self, action):
        self.frame_iteration += 1

        reward = 0

        prev_distance = np.linalg.norm(np.array(self.snake_position) - np.array(self.fruit_position))

        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1 + 4) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x_change, y_change = 0, 0
        if self.direction == 'UP':
            y_change = -self.cell_size
        elif self.direction == 'DOWN':
            y_change = self.cell_size
        elif self.direction == 'LEFT':
            x_change = -self.cell_size
        elif self.direction == 'RIGHT':
            x_change = self.cell_size

        new_head_pos = [self.snake_position[0] + x_change, self.snake_position[1] + y_change]
        self.snake_position = new_head_pos
        self.snake_body.insert(0, list(self.snake_position))
        self.snake_body_set = set(tuple(pos) for pos in self.snake_body)

        self.game_over_flag = False
        cause_of_death = None

        new_distance = np.linalg.norm(np.array(self.snake_position) - np.array(self.fruit_position))
        if new_distance < prev_distance:
            reward += 0.5
        elif new_distance > prev_distance:
            reward -= 0.5

        collision_detected, collision_type = self.is_collision()
        if collision_detected or self.frame_iteration > 100 * len(self.snake_body):
            self.game_over_flag = True
            if collision_detected:
                if collision_type == "self":
                    reward = -50
                else:
                    reward = -50
                cause_of_death = collision_type
            else:
                reward = -10
                cause_of_death = "timeout"
            return reward, self.game_over_flag, self.score, cause_of_death

        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 1
            reward = 10
            self.fruit_position = self._place_fruit()
            while tuple(self.fruit_position) in self.snake_body_set:
                self.fruit_position = self._place_fruit()
            self.frame_iteration = 0
        else:
            tail = self.snake_body.pop()
            self.snake_body_set.remove(tuple(tail))

        for segment in self.snake_body[1:]:
            distance = np.linalg.norm(np.array(self.snake_position) - np.array(segment))
            if distance < 2 * self.cell_size:
                reward -= 0.3

        min_dist_to_body = float('inf')
        if len(self.snake_body) > 1:
            min_dist_to_body = min(
                np.linalg.norm(np.array(self.snake_position) - np.array(segment))
                for segment in self.snake_body[1:]
            )
        max_board_dist = np.linalg.norm(np.array([self.w, self.h]))
        if min_dist_to_body != float('inf'):
            reward += 0.05 * (min_dist_to_body / max_board_dist)

        head_x, head_y = self.snake_position
        min_dist_to_wall = min(
            head_x, self.w - self.cell_size - head_x,
            head_y, self.h - self.cell_size - head_y
        )
        reward += 0.01 * (min_dist_to_wall / (self.w // 2))

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, self.game_over_flag, self.score, cause_of_death