# train_snake_ai.py
import pygame
import matplotlib.pyplot as plt
from IPython import display # Still useful for clearing output in environments like Jupyter
import os
from collections import defaultdict

# Import the game environment and the Q-learning agent
from snake_environment import SnakeGameAI
from q_learning_agent import Agent

# --- Configuration ---
Q_TABLE_FILENAME = "q_table.pkl"
TOTAL_GAMES_TO_TRAIN = 5000
PLOT_EVERY_N_GAMES = 10 # How often to update the plot (adjust for desired plot frequency)

# Ensure matplotlib is in interactive mode for live plotting
plt.ion() # Turn on interactive mode

def init_plot():
    """Initializes and returns the figure and axes for live plotting."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show(block=False) # Display the plot without blocking
    return fig, ax1, ax2

def update_plot(fig, ax1, ax2, scores, mean_scores, record_score, death_causes_counts):
    """Updates the content of the existing plot axes."""
    # Clear previous content on axes
    ax1.cla() # Clear axis 1
    ax2.cla() # Clear axis 2

    # Plot 1: Scores
    ax1.set_title('Training Progress (Q-Learning)')
    ax1.set_ylabel('Score')
    ax1.plot(scores, label='Score')
    ax1.plot(mean_scores, label='Mean Score')
    ax1.axhline(y=record_score, color='r', linestyle='--', label=f'Record: {record_score}')
    ax1.set_ylim(ymin=0)
    # Add text labels for the last score and mean score
    if len(scores) > 0:
        ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        ax1.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot 2: Death Causes
    # Ensure consistent order for plotting categories
    ordered_labels = ["wall", "self", "timeout"]
    ordered_values = [death_causes_counts.get(label, 0) for label in ordered_labels]

    # Filter out labels that have 0 count if you don't want them in the plot.
    filtered_labels = [label for i, label in enumerate(ordered_labels) if ordered_values[i] > 0]
    filtered_values = [value for value in ordered_values if value > 0]

    if filtered_labels: # Only plot if there are actually any deaths to show
        ax2.bar(filtered_labels, filtered_values, color=['red', 'blue', 'orange'])
    ax2.set_title('Cumulative Death Causes')
    ax2.set_xlabel('Death Cause')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y')

    # Redraw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events() # Process any GUI events

def train():
    """Main training loop for the Q-learning Snake AI."""
    pygame.init()

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    death_causes_counts = defaultdict(int)

    agent = Agent()
    game = SnakeGameAI()

    agent.load_q_table(Q_TABLE_FILENAME)

    game.speed = 200 # Visual game speed

    print("Starting Q-Learning training...")

    # Initialize the plot outside the loop
    fig, ax1, ax2 = init_plot()

    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score, cause_of_death = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.save_q_table(Q_TABLE_FILENAME)

                if cause_of_death:
                    death_causes_counts[cause_of_death] += 1

                print(f'Game: {agent.n_games}, Score: {score}, Record: {record}, Epsilon: {agent.epsilon:.4f}, Q-table size: {len(agent.q_table)}')

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                # Update the existing plot instead of creating a new one
                if agent.n_games % PLOT_EVERY_N_GAMES == 0:
                    update_plot(fig, ax1, ax2, plot_scores, plot_mean_scores, record, death_causes_counts)

                if agent.n_games >= TOTAL_GAMES_TO_TRAIN:
                    print(f"Training complete after {agent.n_games} games.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        agent.save_q_table(Q_TABLE_FILENAME)
        pygame.quit()
        plt.close(fig) # Close the matplotlib figure when done

if __name__ == '__main__':
    train()