# train_snake_ai.py
import pygame
import matplotlib.pyplot as plt
from IPython import display # For live plotting in notebooks
import os # To check for saved Q-table file

# Import the game environment and the Q-learning agent
from snake_environment import SnakeGameAI, WINDOW_SIZE, WINDOW_SIZE # Import constants as well if needed
from q_learning_agent import Agent

# Ensure matplotlib is in interactive mode for live plotting
plt.ion()

def plot(scores, mean_scores, record_score):
    """Plots the scores and mean scores in real-time."""
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress (Q-Learning)')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.axhline(y=record_score, color='r', linestyle='--', label=f'Record: {record_score}')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(.1)

def train():
    """Main training loop for the Q-learning Snake AI."""
    pygame.init() # Initialize Pygame for the game display

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 # Highest score achieved so far

    agent = Agent() # Instantiate the Q-learning agent
    game = SnakeGameAI() # Instantiate the game environment

    # Optional: Load a pre-trained Q-table if it exists
    # This is useful to continue training or evaluate a previously trained agent
    #agent.load_q_table("q_table.pkl")

    # Set game speed for training visualization.
    # Higher speed = faster training, less visible frames.
    # Set to 0 to run "headless" (no visual updates) for maximum speed.
    game.speed = 1000 # Adjust this value (e.g., 50 for visible, 1000 for fast, 0 for headless)

    print("Starting Q-Learning training...")

    while True: # Training loop runs indefinitely until manually stopped
        # Get the current state of the game
        state_old = agent.get_state(game)

        # Agent chooses an action based on the current state (epsilon-greedy)
        final_move = agent.get_action(state_old)

        # Perform the chosen action in the game environment
        reward, done, score = game.play_step(final_move)

        # Get the new state after performing the action
        state_new = agent.get_state(game)

        # Train the agent's short memory (update Q-value for the current step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the experience for long-term training (experience replay)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game is over:
            game.reset() # Reset the environment for the next game
            agent.n_games += 1 # Increment game count

            # Train the agent's long memory using a batch of past experiences
            agent.train_long_memory()

            # Update the record score and save Q-table if a new record is set
            if score > record:
                record = score
                agent.save_q_table() # Save the Q-table when a new record is achieved

            # Print training progress to console
            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}, Epsilon: {agent.epsilon:.4f}, Q-table size: {len(agent.q_table)}')

            # Store scores for plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Update the live plot
            plot(plot_scores, plot_mean_scores, record)

            # Optional: Add a condition to stop training after a certain number of games
            if agent.n_games > 5000:
                print("Training complete after 5000 games.")
                break # Exit the training loop

if __name__ == '__main__':
    train() # Start the training process
    pygame.quit() # Quit Pygame once training loop ends (e.g., by breaking or closing window)