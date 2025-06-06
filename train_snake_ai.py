# train_snake_ai.py
import pygame
import matplotlib.pyplot as plt
from IPython import display
import os
from collections import defaultdict
import time
from snake_environment import SnakeGameAI
from q_learning_agent import Agent

# --- Configuration ---
MODEL_FILENAME = "dqn_model.pth"
TOTAL_GAMES_TO_TRAIN = 250000
PLOT_EVERY_N_GAMES = 100
USE_AI_TO_PLAY = False
GAME_SPEED = 0
RENDER_GAME = USE_AI_TO_PLAY

plt.ion()

def init_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax1, ax2

def update_plot(fig, ax1, ax2, scores, mean_scores, record_score, death_causes_counts):
    ax1.cla()
    ax2.cla()
    ax1.set_title('Training Progress (DQN)')
    ax1.set_ylabel('Score')
    ax1.plot(scores, label='Score')
    ax1.plot(mean_scores, label='Mean Score')
    ax1.axhline(y=record_score, color='r', linestyle='--', label=f'Record: {record_score}')
    ax1.set_ylim(ymin=0)
    if len(scores) > 0:
        ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        ax1.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    ax1.legend(loc='upper left')
    ax1.grid(True)
    labels = []
    sizes = []
    colors = []
    color_map = {"wall": "red", "self": "blue", "timeout": "orange"}
    for cause in ["wall", "self", "timeout"]:
        count = death_causes_counts.get(cause, 0)
        if count > 0:
            labels.append(cause)
            sizes.append(count)
            colors.append(color_map[cause])
    if sizes:
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Death Causes Distribution')
    else:
        ax2.set_title('Death Causes Distribution (No Data Yet)')
        ax2.axis('off')
    fig.canvas.draw()
    fig.canvas.flush_events()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    death_causes_counts = defaultdict(int)
    game = SnakeGameAI(speed=GAME_SPEED, render=RENDER_GAME)
    dummy_state = Agent(0,0).get_state(game)
    agent = Agent(state_size=len(dummy_state), action_size=3)

    if os.path.exists(MODEL_FILENAME):
        agent.load_model(MODEL_FILENAME)

    print("Starting DQN Training...")

    fig, ax1, ax2 = init_plot()

    total_game_time = 0.0 # Initialize total time

    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score, cause_of_death = game.play_step(final_move)
            state_new = agent.get_state(game)

            if not USE_AI_TO_PLAY:
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game_duration = time.time() - game.start_time # Calculate game duration
                total_game_time += game_duration # Add to total

                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.save_model(MODEL_FILENAME)

                if cause_of_death:
                    death_causes_counts[cause_of_death] += 1
                    print(f"Cause of death: {cause_of_death}")

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                avg_game_time = total_game_time / agent.n_games if agent.n_games > 0 else 0 # Calculate average

                print(f'Game: {agent.n_games}, Score: {score}, Mean: {mean_score:.2f}, Record: {record}, Epsilon: {agent.epsilon:.4f}, Avg Time: {avg_game_time:.4f}s') # Display avg time

                print("Total Death Causes:")
                for cause, count in death_causes_counts.items():
                    print(f"  {cause}: {count}")

                if agent.n_games % PLOT_EVERY_N_GAMES == 0:
                    update_plot(fig, ax1, ax2, plot_scores, plot_mean_scores, record, death_causes_counts)

                if agent.n_games >= TOTAL_GAMES_TO_TRAIN:
                    print(f"Training complete after {agent.n_games} games.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        time.sleep(5)
        agent.save_model(MODEL_FILENAME)
        print("Training finished and model saved.")
        pygame.quit()
        plt.close(fig)

if __name__ == '__main__':
    train()