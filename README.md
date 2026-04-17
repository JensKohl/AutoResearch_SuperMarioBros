# AutoResearch Super Mario

This repository is an experiment applying Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) to reinforcement learning (RL) on the use case of the famous Super Mario Bros game. 

## Overview
An AI agent works autonomously to build a RL model via the `train.py` script so to reach the end of the level or as far as possible, in minimal time and a high score (in this order). To achieve this, the AI agent can modify the `train.py` script, the RL model algorithm, architecture, hyperparameters, reward function, etc.

## Rules
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

## Summary

203 autonomous experiments were run on branch `autoresearch/Apr13_2226` on an NVIDIA RTX 2060 (6 GB VRAM). Each experiment ran for ~10 minutes of wall-clock training, followed by a deterministic greedy evaluation. The optimisation target is:

```
total_reward = score + max_x_dist                          (level not completed)
total_reward = 10000 + (400 − seconds_to_finish) + score + max_x_dist   (level completed)
```

### Results at a glance

| Rank | Exp | Algorithm | total_reward | max_x_dist | score | What happened |
|------|-----|-----------|-------------|------------|-------|---------------|
| 1 | 91 | Dueling DQN | **2695** | 2195 | 500 | First ever crossing of x=2023 via selective barrier replay |
| 2 | 113 | Dueling DQN | 2559 | 1959 | 600 | ε=1.0 random exploration on clean reward |
| 3 | 80 | Dueling DQN | 2423 | 2023 | 400 | Pure-greedy self-training: x=1137→2023 |
| 4 | 192 | PPO | **2221** | 1521 | 700 | T=0.5 + LR=3e-6, captured by finally-block save |
| 5 | 70 | Dueling DQN | 2034 | 1434 | 600 | Targeted fine-tuning on x>1200 transitions only |

The level was **never completed in the greedy evaluation**. However, PPO stochastic workers (temperature-sampled) completed the level multiple times during training (train_best_reward ≈ 17 000–21 000), which proves the level is solvable by the policy class — the remaining challenge is getting the *greedy* argmax to agree.

### How total_reward evolved

```
Exp   1  (DQN, baseline)          total =  434   x =  434
Exp  15  (RMSprop optimizer)       total = 1818   x = 1518  ← first big jump
Exp  36  (warm start v4)           total = 2117   x = 1517
Exp  64  (ApeX diverse ε)          total = 1427   x =  927  ← enables warm-start chain
Exp  65  (pure greedy warm start)  total = 1935   x = 1435
Exp  70  (x>1200 fine-tune)        total = 2034   x = 1434
Exp  80  (pure greedy x=1137)      total = 2423   x = 2023  ← DQN breakthrough
Exp  91  (selective barrier replay) total = 2695  x = 2195  ← all-time best (DQN)
Exp 113  (eps=1.0 clean reward)    total = 2559   x = 1959
Exp 185  (PPO T=0.3 LR=3e-6)      total = 1719   x = 1519  ← PPO past x=899
Exp 192  (PPO T=0.5 LR=3e-6)      total = 2221   x = 1521  ← PPO best
```

### Algorithms tried

**Dueling DQN (exp 1–120)** — the workhorse of the first half:
- Architecture: 3-layer CNN shared trunk, separate advantage + value streams
- Optimisers compared: Adam, RMSprop (best for fresh starts), SGD
- Loss functions: MSE and Huber (similar performance)
- Replay buffers: uniform, prioritised (PER — diverged), biased oversampling, selective barrier replay (key innovation)
- Exploration: ε-greedy, ApeX diverse epsilons, state-conditional ε, pure greedy (ε=0)
- Special techniques: warm starting from prior checkpoints, pure-greedy self-training, targeted fine-tuning on x>threshold transitions, barrier crossing bonuses

**PPO (exp 121–203)** — adopted after DQN stalled at x=2023:
- Standard Actor-Critic with GAE (γ=0.99, λ=0.95), clipped surrogate objective, entropy regularisation
- Shared CNN trunk warm-started from the DQN checkpoint; FC heads re-initialised
- Key hyperparameter discoveries: SAMPLE_TEMP=0.3–0.5, LR=3e-6, frozen conv+policy[:2], finally-block save
- Techniques tried: Behavioural Cloning from flag-get episodes, KL-anchored updates, advantage masking, frontier filtering, temperature annealing

### Key discoveries

**1. Pure-greedy self-training compounds (DQN)**  
Setting all workers to ε=0 and LR=1e-5 focuses every gradient step on the best-known path. Used twice for large jumps: x=723→1439 (exp 74) and x=1137→2023 (exp 80). The limitation is that it cannot discover new paths — it needs a separate exploration phase first.

**2. Selective barrier replay prevents catastrophic forgetting**  
Exploring near a hard barrier (e.g. x=2023) fills the replay buffer with "death at the barrier" transitions. These train Q-values to avoid that position, corrupting the policy for the entire preceding route. The fix (exp 91): buffer a worker's episode transitions only if the episode *successfully crosses* the barrier; discard failed attempts. This was the only technique that ever pushed past x=2023, achieving the all-time best of 2695.

**3. Shared final-layer weights cause invisible corruption**  
The single linear output layer (policy[-1]) maps 512D features → action logits for all states simultaneously. Any gradient update for x>899 states modifies the same weights that determine actions at x=303 and x=722. This was the root cause of a ~30-experiment plateau at x=899. The solution was PPO with those layers frozen and LR so small (1e-6–3e-6) that the per-step weight change never exceeds the corruption threshold.

**4. The "finally-block save" captures late-training improvements**  
Training often degrades during a run — in-training greedy checks show x going down — but the *end-of-training* model state can be better than the starting checkpoint. The finally block evaluates the final model and saves it only if it strictly improves on the run's own baseline. This mechanism was responsible for the exp 185 (x=899→1519), exp 190 (x=1520→1523), and exp 192 (score 400→700) improvements.

**5. PPO stochastic workers complete the level; greedy policy lags behind**  
PPO with temperature sampling (T=0.5–0.7) reliably solves the level within 10 minutes of training. The gap to the greedy policy arises because temperature sampling allows ~21–45 % non-greedy actions per step — enough "luck" to navigate difficult obstacles. The argmax policy requires every critical state to have the correct action as its strict maximum, a much harder convergence target.

**6. The safe learning-rate threshold is checkpoint-dependent**  
The maximum LR that avoids greedy-policy corruption rises and falls with each new checkpoint. At x=899: LR=1e-6 safe, LR=3e-6 corrupts. At x=1519: LR=3e-6 sometimes works. At x=2221: LR=2e-6 already corrupts. Each improvement raises the sensitivity of the policy and lowers the safe LR ceiling.

### What did not work

| Approach | Why it failed |
|----------|--------------|
| Double DQN | Unstable; never improved over plain Dueling DQN in the 10-minute budget |
| Prioritised Experience Replay (PER) | Training diverged (value estimates exploded) |
| Barrier exploration without selective replay | Failed crossing attempts flood the buffer with "die here" examples |
| PPO Behavioural Cloning from flag-get episodes | flag_get fires before `done`, causing detection bugs; when fixed, BC gradient was too small or destabilised training |
| State-conditional exploration (greedy until barrier, then random) | Even with threshold at x=2022, failed attempts one step before the barrier corrupted Q-values |
| KL-anchored PPO | KL=0.3–1.0 either prevented any learning or failed to stop degradation |
| k=8 frame stack | evaluate.py uses k=4 hardcoded — model shapes mismatched |
| BARRIER_BONUS ≥ 10 000 | Gradient magnitude overwhelmed earlier-stage Q-values |
| LR ≥ 3e-5 on any warm-started model | Catastrophic forgetting in the first few rollouts, every time |

## File map

| File | Description |
|------|-------------|
| `src/train.py` | Fully mutable training script — algorithm, architecture, reward, hyperparameters |
| `src/model.py` | `PolicyModel` class (CNN + policy head + value head + residual beyond_head) |
| `src/evaluate.py` | Read-only evaluation harness |
| `src/constants.py` | Fixed constants: time budget, action set, eval seeds |
| `results.tsv` | Per-experiment metrics (untracked by git) |
| `CHANGES.MD` | Detailed narrative log for every experiment (untracked by git) |

---

## Outlook

The experiments surfaced a set of clear, tractable problems. Each section below states the problem, explains why it matters, and sketches a concrete approach.

### 1. Bridge the stochastic–greedy gap to complete the level

**Problem:** PPO stochastic workers complete Mario 1-1 reliably within 10 minutes. The greedy policy reaches only x≈1521. Closing this gap would unlock the 10 000-point completion bonus — a 4× improvement over the current best.

**Why the gap exists:** Temperature sampling draws from a softmax distribution over actions; at T=0.5 roughly 21 % of steps are non-greedy. This "noise" lets the agent occasionally stumble through hard obstacles by chance. The argmax policy gets zero such luck and needs every critical state to be unambiguous.

**Concrete approaches:**

- *Temperature annealing.* Start at T=0.7 (workers explore freely, level gets completed stochastically) and decay T toward 0 over the run. As T→0 the policy is forced to become deterministic *while still training on completion trajectories*. This is essentially curriculum learning on the exploration budget.
- *Completion-episode distillation.* Every time a stochastic worker completes the level, record the full trajectory. After training, run a second pass where the policy is trained via supervised cross-entropy (`argmax = completed_action`) on these trajectories only, using a frozen CNN. Because the training data contains only winning paths, the greedy policy learns to copy them.
- *Population-based evaluation.* Keep a ring buffer of the last N completed-level trajectories. Use these as the "greedy evaluation" target: a checkpoint is saved only when it reproduces ≥1 trajectory in the greedy eval. This rewards policies that have internalised level completion, not just those that have the highest average x.

### 2. Fix the checkpoint overwrite problem permanently

**Problem:** Three separate experiments (exp 77, exp 92, exp 177) lost hard-earned checkpoints because the save criterion compared against a stale or just-reinitialised baseline rather than the global all-time best. Recovery required starting from scratch and cost many experiments.

**Why it matters:** Each overwrite effectively erases a breakthrough that took dozens of experiments to achieve. With better bookkeeping the x=2695 DQN result and x=2023 PPO would both still be available for further fine-tuning.

**Concrete approach:**

```python
# At training startup, unconditionally record:
GLOBAL_BEST_FILE = "MODELS/global_best.pt"
global_best = load_if_exists(GLOBAL_BEST_FILE)
baseline_combined = greedy_eval(model)  # always from warm-start model, never 0

# At any save point:
if new_combined > global_best_combined:
    torch.save(model.state_dict(), GLOBAL_BEST_FILE)
    global_best_combined = new_combined
```

Keeping `global_best.pt` separate from the working `model.pt` means training can reinitialise or experiment freely without touching the historical best. A secondary benefit: the researcher always knows the true ceiling and can target it explicitly.

### 3. Per-barrier curriculum with isolated sub-policies

**Problem:** The shared linear output layer couples all positions. Any gradient for x>1521 also changes the argmax at x=303 and x=722 — leading to corruption. This shared-weight coupling is the single most common failure mode across 203 experiments.

**Why it matters:** Almost every stall (x=899 for 30+ experiments, x=1521 for 10+ experiments) traces back to this. Fixing the coupling would allow higher learning rates and more aggressive exploration at the frontier without risking regressions elsewhere.

**Concrete approaches:**

- *Position-conditioned output heads.* Divide the level into zones (0–700, 700–1200, 1200–1600, 1600+). Maintain a separate small linear head for each zone, switching based on the current x position. Gradients for zone k never touch zone j weights. The shared CNN backbone still learns general visual features, but zone-specific decisions are isolated. This is similar to [multi-head policies in option-critic](https://arxiv.org/abs/1609.05140).
- *Modular residual heads.* The current `beyond_head` (a zero-init residual added on top of policy[-1]) is a step in this direction but still fails because its features are correlated with all x positions. A better design: assign one beyond_head per zone, with a hard mask ensuring zone k's head only receives gradient from zone k transitions.
- *Separate specialist networks.* Train a specialist model for the barrier region only (e.g. last 200 frames before x=1521), warm-starting its CNN from the main model but keeping its own output head. Use the main model for x<1400 and switch to the specialist near the barrier. Ensemble or gate the outputs at the junction.

### 4. Smarter reward shaping

**Problem:** The current reward (`Δx × 2 + Δscore × 0.3 − 0.1/step`) has worked but has two known weaknesses: (a) the score delta coefficient was found empirically and is likely suboptimal, and (b) there is no signal guiding the agent *toward* a specific obstacle when it is stuck.

**Why it matters:** Better reward shaping could accelerate learning past barriers and increase score (currently 700 out of ~3200 possible on 1-1).

**Concrete approaches:**

- *Potential-based reward shaping.* Define a potential function φ(x) = x / x_max. The shaped reward r′ = r + γφ(x′) − φ(x) adds a smooth pull toward higher x without changing the optimal policy (proven equivalent by [Ng et al. 1999](http://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)). This is strictly safer than arbitrary bonuses.
- *Adaptive barrier bonus.* Instead of a one-time bonus at a fixed x, use a rolling bonus: `bonus = k × max(0, x − rolling_max_x)`. Every new personal best for that worker earns a reward proportional to how far past the frontier it went. This continuously incentivises exploration without the "barrier already crossed, bonus spent" problem.
- *Intrinsic curiosity for score.* Mario's score items (coins, enemies) are clustered in known locations. Adding a small curiosity reward based on whether a new score event occurred (rather than magnitude) would incentivise the agent to find *new* score opportunities rather than replaying the same coin block over and over.

### 5. Architecture improvements

**Problem:** The current CNN (Conv 8×8/4 → Conv 4×4/2 → Conv 3×3/1 → FC 3136→512) was designed for Atari in 2015 ([DQN Nature paper](https://www.nature.com/articles/nature14236)). It has no temporal memory beyond 4 stacked frames and no skip connections to preserve fine-grained spatial detail.

**Why it matters:** Mario 1-1 contains obstacles that require precise timing (multi-frame jumps over gaps, enemy bounces). The 4-frame stack covers ~67 ms of real time — marginal for reliably learning jump timing at 24 fps.

**Concrete approaches:**

- *LSTM policy head.* Replace the flat FC → logits path with a single-layer LSTM (hidden size 256–512). The LSTM accumulates episode-level context across arbitrary time horizons, which is especially valuable for obstacles where the correct action depends on where the agent has been, not just the last 4 frames. Reference: [R2D2 (Kapturowski et al. 2019)](https://openreview.net/forum?id=r1lyTjAqYX).
- *Deeper CNN with skip connections.* Add residual skip connections between conv layers 1 and 3. This has two benefits: (a) gradients flow more directly to early layers (less vanishing), (b) fine-grained pixel-level features from layer 1 are concatenated with higher-level semantic features from layer 3, giving the FC head both resolution and abstraction. Demonstrated to help in [IMPALA (Espeholt et al. 2018)](https://arxiv.org/abs/1802.01561).
- *Larger frame stack (k=8) with evaluate.py fix.* k=8 (covering ~133 ms) was tried in exp 88 but crashed because evaluate.py hardcodes k=4. Patching evaluate.py (or adding a model metadata header to checkpoint files so evaluate.py can auto-detect k) would unlock this. A proper ablation comparing k=4 vs k=8 with otherwise identical settings has never been done.

### 6. Alternative RL algorithms

Several modern algorithms offer structural advantages over DQN and vanilla PPO for this problem.

**SAC (Soft Actor-Critic)**  
SAC ([Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)) maximises entropy alongside reward, which naturally balances exploration and exploitation without manual ε or temperature schedules. Its off-policy nature (like DQN) allows replay buffers, and its entropy term prevents collapse to a deterministic policy too early. Particularly well-suited for the stochastic–greedy convergence problem.

**DreamerV3**  
DreamerV3 ([Hafner et al. 2023](https://arxiv.org/abs/2301.04104)) learns a world model (predict next state and reward from compressed latent) and plans inside it using imagination rollouts. The key advantage for hard barriers: the agent can *imagine* paths past x=1521 without ever physically reaching x=1521 in training, sidestepping the catastrophic-forgetting problem entirely. DreamerV3 also shows strong data efficiency — important given the 10-minute constraint.

**MuZero**  
MuZero ([Schrittwieser et al. 2020](https://www.nature.com/articles/s41586-020-03051-4)) combines a learned model with MCTS planning. For Mario, MCTS could tree-search through the obstacle around x=1521, trying multiple action sequences in the learned model before committing. This is a direct solution to the "one wrong action at the barrier" problem that has dominated the experiment history.

**Go-Explore**  
Go-Explore ([Ecoffet et al. 2021](https://www.nature.com/articles/s41586-021-03528-y)) addresses hard-exploration problems by explicitly archiving visited states and returning to them deterministically for further exploration — exactly the use case here. It would archive the state just before x=1521, then systematically try all actions from that state in a separate exploration phase, keeping any that advance further. No replay buffer corruption, no gradient entanglement.

### 7. Longer training runs and multi-session accumulation

**Problem:** Each experiment has a hard 10-minute wall-clock limit. The most successful PPO improvements (exp 185, 190, 192) were captured by the finally-block, suggesting the model was *still improving* when training stopped.

**Why it matters:** The current limit forces the agent to make coarse jumps rather than allow slow, stable convergence. A 30-minute or 60-minute budget might close the greedy–stochastic gap completely.

**Concrete approach:**

- Extend `TIME_BUDGET` in `constants.py` to 1 800 s (30 min) for dedicated "convergence runs" after a new best is found. The hardware budget (GPU heat) is the practical limit, not any algorithmic constraint.
- Run multi-session PPO: each session warm-starts from the previous session's model and accumulates learning across N consecutive 10-minute windows, only keeping a new checkpoint if the combined score after all N sessions improves. This is how the score went 400→500→800→1519 across experiments 162→164→185 — the multi-session effect, but manual. Automating it would be straightforward.
