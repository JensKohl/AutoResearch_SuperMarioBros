# AutoResearch Super Mario

This repository is an experiment applying Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) methodology to reinforcement learning (RL) on the use case of the famous Super Mario Bros game.

## Overview
An AI agent works autonomously to build a RL model via the `train.py` script so to reach the end of the level or as far as possible, in minimal time and a high score (in this order). To achieve this, the AI agent can modify the `train.py` script, the RL model algorithm, architecture, hyperparameters, reward function, etc.

### Rules
- `train.py` is fully mutable by the AI research agent. The agent can change the algorithm, model architecture, hyperparameters such as learning rate, the reward function, etc.
- `evaluate.py` is read-only in which the the results of the training are evaluated.
- `constants.py` contains static constraints like the training time budget, button configuration, etc.
- The training loops run for exactly 10 minutes.
- Afterwards, the agent evalutes the results and either keeps the changes for the next iteration or discards the changes.

## Running Experiments

To kick off the autonomous loop:
1. Read the `AGENTS.md` rules.
2. The autonomous researcher will handle the git branching, experimental modifications, and recording results to `results.tsv`.
