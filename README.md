# AutoResearch Super Mario

This repository is an experiment applying Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) to reinforcement learning (RL) on the use case of the famous Super Mario Bros game. 

## Overview
An AI agent works autonomously to build a RL model via the `train.py` script so to reach the end of the level or as far as possible, in minimal time and a high score (in this order). To achieve this, the AI agent can modify the `train.py` script, the RL model algorithm, architecture, hyperparameters, reward function, etc.

### Rules
- `train.py` is fully mutable by the AI research agent. The agent can change the algorithm, model architecture, hyperparameters such as learning rate, the reward function, etc.
- `evaluate.py` is read-only in which the the results of the training are evaluated.
- `constants.py` contains static constraints like the training time budget, button configuration, etc.
- The training loops run for exactly 5 minutes.
- Afterwards, the agent evalutes the results and either keeps the changes for the next iteration or discards the changes.

## Running Experiments

To kick off the autonomous loop:
1. Ensure the `ROMS` folder has the necessary files.
2. Read the `AGENTS.md` rules.
3. The autonomous researcher will handle the git branching, experimental modifications, and recording results to `results.tsv`.

---

## Experiment Summary (203 experiments, branch `autoresearch/Apr13_2226`)

This summary covers 203 autonomous experiments run on an NVIDIA RTX 2060 (6GB VRAM). Each experiment ran for ~10 minutes of training time followed by a greedy evaluation. The primary optimization target is `total_reward = score + max_x_dist` (or `10000 + time_bonus + score + x_dist` if the level is completed).

### Best Results

| Rank | Experiment | Algorithm | total_reward | max_x_dist | score | Notes |
|------|-----------|-----------|-------------|------------|-------|-------|
| 1 | Exp 91 | DQN (Dueling) | **2695** | 2195 | 500 | All-time best: first past x=2023 barrier |
| 2 | Exp 113 | DQN (Dueling) | 2559 | 1959 | 600 | eps=1.0 random exploration breakthrough |
| 3 | Exp 80 | DQN (Dueling) | 2423 | 2023 | 400 | Pure greedy self-training x=1137→2023 |
| 4 | Exp 192 | PPO | **2221** | 1521 | 700 | PPO best: T=0.5+LR=3e-6 finally-block save |
| 5 | Exp 70 | DQN (Dueling) | 2034 | 1434 | 600 | Targeted fine-tuning x>1200 |

The level was **never completed** in the greedy evaluation. However, PPO stochastic workers completed the level multiple times during training (train_best_reward ~17000-21000), confirming it is achievable in principle.

### Reward Progression Over Time

```
Exp 1  (DQN baseline):     total=434,  x=434
Exp 15 (RMSprop):          total=1818, x=1518  ← RMSprop optimizer breakthrough
Exp 36 (warm start):       total=2117, x=1517
Exp 64 (ApeX diverse ε):   total=1427, x=927   ← diversity enables warm-start chain
Exp 65 (pure greedy):      total=1935, x=1435
Exp 70 (x>1200 filter):    total=2034, x=1434
Exp 80 (pure greedy):      total=2423, x=2023  ← all-time DQN breakthrough
Exp 91 (barrier replay):   total=2695, x=2195  ← all-time best (DQN)
Exp 113 (eps=1.0):         total=2559, x=1959  ← fresh approach after 91-series stall
Exp 185 (PPO T=0.3):       total=1719, x=1519  ← PPO finally past x=899
Exp 192 (PPO T=0.5):       total=2221, x=1521  ← PPO best
```

### Algorithms Tried

**Deep Q-Network (DQN) variants** — experiments 1–120:
- Standard DQN, Double DQN, Dueling DQN (best)
- Optimizers: Adam, RMSprop (best), SGD
- Loss functions: MSE, Huber
- Replay buffers: uniform, prioritized (PER), biased, selective barrier
- Exploration: ε-greedy, ApeX diverse epsilons, state-conditional exploration, pure greedy
- Training techniques: warm starting, pure greedy self-training, targeted fine-tuning, barrier bonuses

**PPO (Proximal Policy Optimization)** — experiments 121–203:
- Actor-Critic with GAE, clipped surrogate objective, entropy regularization
- Behavioral Cloning (BC) from successful stochastic episodes
- KL-anchored PPO, temperature-scaled sampling, advantage masking
- Key hyperparameter: LR=3e-6, SAMPLE_TEMP=0.3-0.5, finally-block save

### Key Discoveries

**1. Pure greedy self-training (DQN) compounds well:**  
Running all workers with ε=0 and very low LR (1e-5) concentrates experience on the current best trajectory, enabling jumps: x=723→1439 (exp74), x=1137→2023 (exp80). Limitation: requires exploration to discover new paths first.

**2. Selective barrier replay solves catastrophic forgetting:**  
Standard exploration at hard barriers (x=2023) fills the replay buffer with "death at the barrier" transitions, corrupting Q-values for the entire preceding route. Solution (exp91): only add barrier-crossing transitions to the buffer if they *successfully cross* the barrier. This enabled the all-time best of x=2195 (total=2695).

**3. The x=899 DQN barrier was a shared-weights problem:**  
For over 30 experiments, the policy was stuck at x=899. Any gradient update for x>899 states also modified the shared final linear layer (policy[-1]), corrupting decisions at x=303 and x=722. Solution: switch to PPO with frozen conv+FC features and very low LR (3e-6), allowing tiny weight changes that accumulate improvements without crossing the corruption threshold. This pushed x=723→722→898→1519 (exp179–185).

**4. The "finally-block save" mechanism enables late-training captures:**  
Many improvements were captured by the training's final evaluation step, not by in-training greedy checks. Training often oscillates/degrades during the run but the final state can be better than the starting checkpoint. This mechanism was responsible for the exp185 (x=1519), exp190 (x=1523), and exp192 (score=700) breakthroughs.

**5. PPO stochastic workers can complete the level; greedy policy cannot (yet):**  
PPO workers with temperature sampling reliably completed Mario 1-1 within 10 minutes of training (train_best_reward ~20000). The bottleneck is the gap between stochastic (T=0.5-0.7) and greedy (T=0) behavior — the greedy argmax policy is much harder to train since it requires every single critical state to have the correct action as the argmax.

**6. Each new checkpoint requires retuning the safe LR:**  
The maximum LR that avoids corruption is checkpoint-specific. At x=899: LR=1e-6 safe, LR=3e-6 corrupts. At x=1519: LR=3e-6 sometimes works. At x=1521 (score=700): LR=3e-6 corrupts, LR=2e-6 also corrupts. Each improvement raises the bar.

### What Didn't Work

- **Double DQN**: unstable with current setup, never improved over plain DQN
- **Prioritized Experience Replay (PER)**: training diverged
- **Barrier exploration without selective replay**: any failed barrier attempt corrupts Q-values
- **PPO Behavioral Cloning**: flag_get detection timing bugs, BC gradient too small, or destabilized training
- **State-conditional exploration** (reach barrier, then explore): degradation from failed attempts overwhelmed any crossings
- **KL-anchored PPO**: too anchored to learn past barrier; too loose to prevent degradation
- **k=8 frame stack**: incompatible with evaluate.py (fixed k=4)
- **Large BARRIER_BONUS (10000+)**: corrupted earlier-game Q-values via gradient overflow
- **LR ≥ 3e-5 with warm start**: always caused catastrophic forgetting

### Future Work

1. **Level completion**: PPO stochastic workers complete the level. The gap to greedy completion needs bridging. Ideas:
   - Reduce temperature slowly over training (annealing from T=0.5 to T=0.0)
   - Train exclusively on level-completion episodes to specialize the greedy policy
   - Population-based training: maintain a pool of policies at different temperatures

2. **Better architecture for multi-scale features**:
   - Current CNN: 3 conv layers (16→32→64 filters) with a single 3136→512 FC layer
   - ResNet-style skip connections may help preserve early-game features
   - LSTM/GRU for temporal memory — currently using 4-frame stack only

3. **Improved barrier handling**:
   - Instead of one-time bonus, use shaped reward: increasing reward as agent approaches barrier from below
   - Curriculum: train a separate "barrier specialist" policy on only the ~100 frames before the barrier
   - Option-based RL: learn sub-goals where one option targets each known barrier

4. **Smarter checkpoint management**:
   - Always keep the global all-time best model separately from the training checkpoint
   - Implement checkpoint ranking to prevent exp177-style overwrites
   - Multi-seed evaluation (3-5 greedy runs) to reduce variance in checkpoint decisions

5. **Score maximization**:
   - Current score=700 out of a possible ~3200 on 1-1 (coins, Goombas, mushrooms)
   - A dedicated score-farming phase after reaching x=1521 could significantly improve total_reward
   - Intrinsic motivation reward for enemy kills and coin collection

6. **Algorithm alternatives**:
   - **SAC (Soft Actor-Critic)**: entropy-regularized, naturally handles exploration-exploitation trade-off
   - **DreamerV3**: world-model-based; could plan past barriers without needing to experience them directly
   - **R2D2**: recurrent DQN with better multi-episode memory
   - **MuZero**: planning with learned model, showed strong results on Atari

### File Map

| File | Description |
|------|-------------|
| `src/train.py` | Main training script (all algorithms tried here) |
| `src/model.py` | PolicyModel class — CNN + policy + value heads |
| `src/evaluate.py` | Read-only evaluation harness |
| `src/constants.py` | Training constants (time budget, game config) |
| `results.tsv` | All 203 experiment results (untracked) |
| `CHANGES.MD` | Detailed per-experiment log (untracked) |
