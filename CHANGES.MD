# Changes Log

Each experiment appends a section here describing what was changed and why.

## Experiment 2 — Double DQN + larger replay buffer (discarded)

**What changed:** Switched to Double DQN. Increased replay buffer 10k→50k. Reduced batch size 128→64.

**Why:** Double DQN reduces Q-value overestimation by decoupling action selection (policy net) from action evaluation (target net). See [van Hasselt et al. 2015](https://arxiv.org/abs/1509.06461). Larger buffer = more diverse experience replay.

**Result:** total_reward=40 vs baseline 306 — worse. Discarded. Likely the smaller batch size hurt more than Double DQN helped within the short 5-minute budget.
