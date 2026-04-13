# Experiment Log

## Baseline (commit 49cd781)

**What**: Ran the original DQN implementation without any modifications, to establish a baseline score.

**Architecture**: Standard DQN with 3 convolutional layers (32→64→64 filters) + 2 fully connected layers (3136→512→n_actions). Input: 4 stacked grayscale 84x84 frames.

**Algorithm**: DQN with epsilon-greedy exploration (ε: 1.0→0.02 over 10k steps), experience replay buffer (10k), target network updated every 1000 steps.

**Reward shaping**: Distance-based reward (+2× x_pos delta per step, -0.1 time penalty, +1000 flag bonus).

**Hyperparameters**: LR=1e-4, γ=0.99, batch=128, memory=10k, frame skip=3.

**Results**: See results.tsv.
