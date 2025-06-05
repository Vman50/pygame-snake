# snake_environment.py
import pygame
import random
import numpy as np

# --- Game Constants ---
BOARD_CELLS = 12
WINDOW_SIZE = 800 - (800 % BOARD_CELLS) # Make sure window size is a multiple of board_cells
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
        self.display = pygame.display.set_mode((self.w, self.h)) if self.render else None
        if self.render:
            pygame.display.set_caption('Snake AI Environment')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game state for a new episode."""
        self.snake_position = [5 * self.cell_size, 5 * self.cell_size]
        self.snake_body = [
            list(self.snake_position),
            [4 * self.cell_size, 5 * self.cell_size],
            [3 * self.cell_size, 5 * self.cell_size],
            [2 * self.cell_size, 5 * self.cell_size]
        ]
        self.fruit_position = self._place_fruit()
        while self.fruit_position in self.snake_body: # Ensure fruit doesn't spawn on snake
            self.fruit_position = self._place_fruit()

        self.direction = 'RIGHT' # Initial direction
        self.score = 0
        self.frame_iteration = 0 # To track how long it's alive without eating
        self.game_over_flag = False

    def _place_fruit(self):
        """Places the fruit at a random valid position."""
        x = random.randrange(0, self.board_cells) * self.cell_size
        y = random.randrange(0, self.board_cells) * self.cell_size
        return [x, y]

    def _update_ui(self):
        if not self.render:
            return
        """Draws the game state to the Pygame window."""
        # Draw checkerboard background
        for row in range(self.board_cells):
            for col in range(self.board_cells):
                color = LIGHT_GREEN if (row + col) % 2 == 0 else DARK_GREEN
                pygame.draw.rect(
                    self.display,
                    color,
                    pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                )

        # Draw snake
        for pos in self.snake_body:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pos[0], pos[1], self.cell_size, self.cell_size))

        # Draw fruit
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.fruit_position[0], self.fruit_position[1], self.cell_size, self.cell_size))

        # Draw score
        self.show_score(1, WHITE, 'times new roman', 20)
        pygame.display.flip() # Update the full display Surface to the screen

    def show_score(self, choice, color, font, size):
        """Helper to display the current score."""
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        self.display.blit(score_surface, score_rect)

    def is_collision(self, pt=None):
        """
        Checks for collision.
        Returns: (bool, str or None) -> (True if collision, "wall" or "self" if collision, None otherwise)
        """
        if pt is None:
            pt = self.snake_position
        # Hits boundary
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True, "wall"
        # Hits itself (check if pt is in body, excluding head)
        if pt in self.snake_body[1:]:
            return True, "self"
        return False, None

    def play_step(self, action):
        """
        Performs one step of the game given an action.
        Returns reward, game_over_flag, score, cause_of_death.
        """
        self.frame_iteration += 1

        # Determine the new direction based on the action [straight, right, left]
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # Straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]): # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1] Left turn
            next_idx = (idx - 1 + 4) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        # Move snake
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

        reward = 0
        self.game_over_flag = False
        cause_of_death = None # Initialize cause of death

        # Check for collision
        collision_detected, collision_type = self.is_collision()
        # Penalty if it collides or gets stuck (timeout)
        if collision_detected or self.frame_iteration > 100 * len(self.snake_body):
            self.game_over_flag = True
            if collision_detected:
                if collision_type == "self":
                    reward = -50  # MUCH larger penalty for self-collision
                else:
                    reward = -10  # Standard wall collision penalty
                cause_of_death = collision_type # Set to "wall" or "self"
            else:
                reward = -10 # Large penalty for dying
                cause_of_death = "timeout" # Set to "timeout" if no collision but frame limit hit
            return reward, self.game_over_flag, self.score, cause_of_death # Return cause_of_death

        # Check if fruit eaten
        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 1
            reward = 10 # Reward for eating fruit
            self.fruit_position = self._place_fruit()
            while self.fruit_position in self.snake_body: # Ensure fruit doesn't spawn on snake
                self.fruit_position = self._place_fruit()
            self.frame_iteration = 0 # Reset frame iteration on eating
        else:
            self.snake_body.pop() # Remove tail if no fruit eaten

        # Proximity penalty (encourage staying away from body)
        for segment in self.snake_body[1:]:
            distance = np.linalg.norm(np.array(self.snake_position) - np.array(segment))
            if distance < 2 * self.cell_size:  # Adjust threshold as needed
                reward -= 0.3  # Small negative reward for proximity

        # Small negative reward for each step to encourage faster fruit consumption
        reward += -0.1

        # Update UI and clock
        if self.render:
            self._update_ui()
            self.clock.tick(self.speed) # Control game speed for visualization

        return reward, self.game_over_flag, self.score, cause_of_death # Always return cause_of_death (will be None if game continues)