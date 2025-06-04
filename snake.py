#snake game
import pygame
import random
import time
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle

# Default board settings
board_cells = 12
window_x = 800 - (800 % board_cells)
window_y = 800 - (800 % board_cells)
cell_size = window_x // board_cells

# Colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)
light_green = pygame.Color(170, 215, 81)
dark_green = pygame.Color(162, 209, 73)

snake_speed = 10  # You can adjust this as needed

def run_game():
    pygame.init()
    pygame.display.set_caption('Snake')
    game_window = pygame.display.set_mode((window_x, window_y))
    fps = pygame.time.Clock()

    snake_position = [5 * cell_size, 5 * cell_size]
    snake_body = [
        [5 * cell_size, 5 * cell_size],
        [4 * cell_size, 5 * cell_size],
        [3 * cell_size, 5 * cell_size],
        [2 * cell_size, 5 * cell_size]
    ]
    fruit_position = [
        random.randrange(0, board_cells) * cell_size,
        random.randrange(0, board_cells) * cell_size
    ]
    fruit_spawn = True
    direction = 'RIGHT'
    change_to = direction
    score = 0

    def show_score(choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(score), True, color)
        score_rect = score_surface.get_rect()
        game_window.blit(score_surface, score_rect)

    def game_over():
        my_font = pygame.font.SysFont('times new roman', 50)
        game_over_surface = my_font.render('Your Score is : ' + str(score), True, red)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (window_x / 2, window_y / 4)
        game_window.fill(black)
        game_window.blit(game_over_surface, game_over_rect)
        pygame.display.flip()
        time.sleep(2)
        pygame.quit()
        quit()
    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    change_to = 'UP'
                if event.key == pygame.K_DOWN:
                    change_to = 'DOWN'
                if event.key == pygame.K_LEFT:
                    change_to = 'LEFT'
                if event.key == pygame.K_RIGHT:
                    change_to = 'RIGHT'

        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        if direction == 'UP':
            snake_position[1] -= cell_size
        if direction == 'DOWN':
            snake_position[1] += cell_size
        if direction == 'LEFT':
            snake_position[0] -= cell_size
        if direction == 'RIGHT':
            snake_position[0] += cell_size

        snake_body.insert(0, list(snake_position))
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 1
            fruit_spawn = False
        else:
            snake_body.pop()

        if not fruit_spawn:
            fruit_position = [
                random.randrange(0, board_cells) * cell_size,
                random.randrange(0, board_cells) * cell_size
            ]
        fruit_spawn = True

        # Draw checkerboard background
        for row in range(board_cells):
            for col in range(board_cells):
                color = light_green if (row + col) % 2 == 0 else dark_green
                pygame.draw.rect(
                    game_window,
                    color,
                    pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                )

        for pos in snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], cell_size, cell_size))

        pygame.draw.rect(game_window, white, pygame.Rect(fruit_position[0], fruit_position[1], cell_size, cell_size))

        if snake_position[0] < 0 or snake_position[0] >= window_x:
            game_over()
        if snake_position[1] < 0 or snake_position[1] >= window_y:
            game_over()

        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                game_over()

        show_score(1, white, 'times new roman', 20)
        pygame.display.update()
        fps.tick(snake_speed)

def start_game():
    class SnakeMenuApp(App):
        def build(self):
            # Main layout with background
            root = BoxLayout(orientation='vertical', padding=20, spacing=20)
            # Add rounded background
            with root.canvas.before:
                Color(0.4, 0.8, 1, 0.95)
                self.bg = RoundedRectangle(radius=[20], pos=root.pos, size=root.size)
            root.bind(pos=self.update_bg, size=self.update_bg)

            # Title
            root.add_widget(Label(text="Snake Game", font_size=32, size_hint=(1, 0.2)))

            # Board size row
            board_row = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.2))
            board_row.add_widget(Label(text="Board Size:", font_size=20, size_hint=(0.5, 1)))

            small_btn = Button(text="Small\n8x8", font_size=18, background_color=(0.7, 0.9, 1, 1))
            small_btn.bind(on_release=lambda instance: self.set_cell_size(8))
            board_row.add_widget(small_btn)

            medium_btn = Button(text="Medium\n12x12", font_size=18, background_color=(1, 0.6, 0, 1))
            medium_btn.bind(on_release=lambda instance: self.set_cell_size(12))
            board_row.add_widget(medium_btn)

            large_btn = Button(text="Large\n16x16", font_size=18, background_color=(1, 0, 0, 1), color=(1, 1, 1, 1))
            large_btn.bind(on_release=lambda instance: self.set_cell_size(16))
            board_row.add_widget(large_btn)

            root.add_widget(board_row)

            # Spacer
            root.add_widget(Widget(size_hint=(1, 0.2)))

            # Start game button
            start_btn = Button(text="Start Game", background_color=(0, 1, 0, 1), font_size=24, size_hint=(1, 0.2))
            start_btn.bind(on_release=self.start_game)
            root.add_widget(start_btn)

            Window.size = (400, 350)
            return root

        def update_bg(self, *args):
            self.bg.pos = self.root.pos
            self.bg.size = self.root.size

        def set_cell_size(self, size):
            global board_cells, cell_size, window_x, window_y
            board_cells = size
            window_x = 800 - (800 % board_cells)
            window_y = 800 - (800 % board_cells)
            cell_size = window_x // board_cells

        def start_game(self, instance):
            self.stop()
            run_game()

    SnakeMenuApp().run()

start_game()