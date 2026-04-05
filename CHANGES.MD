# Changes Log

Each experiment appends a section here describing what was changed and why.

## Experiment 2 — Double DQN + larger replay buffer (discarded)

**What changed:** Switched to Double DQN. Increased replay buffer 10k→50k. Reduced batch size 128→64.

**Why:** Double DQN reduces Q-value overestimation by decoupling action selection (policy net) from action evaluation (target net). See [van Hasselt et al. 2015](https://arxiv.org/abs/1509.06461). Larger buffer = more diverse experience replay.

**Result:** total_reward=40 vs baseline 306 — worse. Discarded. Likely the smaller batch size hurt more than Double DQN helped within the short 5-minute budget.

## Experiment 3 — Disable render, faster epsilon decay, Huber loss, gradient clipping

**What changed:**
- `RENDER=False`: training no longer opens a game window. The visual rendering was consuming CPU/GPU cycles for no benefit.
- `EPS_DECAY=10000→3000`: With ~5k steps in the 5-minute budget, the old decay left epsilon ≈ 0.6 at end of training (still 60% random!). Now epsilon reaches ~0.05 by ~9k steps, so the agent becomes exploitative much faster.
- `EPS_END=0.02→0.05`: Small increase to leave a little exploration at the end.
- `TARGET_UPDATE=1000→500`: Update target net twice as often, better tracking of policy.
- `MEMORY_SIZE=10000→20000`: Slightly larger buffer for more diverse samples.
- `MSE → SmoothL1 (Huber) loss`: Standard DQN uses Huber loss, which is less sensitive to reward outliers. See [Mnih et al. 2015](https://www.nature.com/articles/nature14236).
- Gradient clipping (`clip_grad_norm_=10`): Prevents exploding gradients, standard practice.

**Why:** The agent was training for 5 minutes but still acting mostly randomly at evaluation time — the epsilon decay was too slow. Disabling rendering frees up compute for more actual learning steps.
