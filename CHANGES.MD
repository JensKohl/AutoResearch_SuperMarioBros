# Changes Log

Each experiment appends a section here describing what was changed and why.

## Exp1 — Baseline (branch autoresearch/Apr8_2026)

**What changed:** Nothing. Fresh start from commit `418f190` — vanilla DQN, 4-frame
stack, frame-skip=3, distance-based reward shaping, Adam lr=1e-4, eps-greedy decay
over 10k steps, 10k replay buffer, batch 128.

**Why:** First run on a new branch must establish the baseline so later experiments
have a reference point. The training budget is now 10 min (`TIME_BUDGET=600`), so we
need a fresh baseline under that budget rather than reusing numbers from the old
5 min loop.

**Result:** `total_reward=435`, `max_x_dist=435`, `score=0`, `flag_get=False`,
`peak_vram_mb=115`. Agent gets stuck at the first major obstacle around x≈434 (the
pipe/goomba region) — same wall prior runs hit. This is our reference point.

## Exp2 — Double DQN + bigger buffer + longer exploration + Huber loss

**What changed:**
- Switched from vanilla DQN to **Double DQN**: the online network picks the next
  action, the target network evaluates it. Decouples action selection from value
  estimation and reduces the well-known overestimation bias of standard DQN
  (van Hasselt et al. 2015, <https://arxiv.org/abs/1509.06461>).
- `MEMORY_SIZE` 10k → **50k**: the 10 min training budget produces ~50k transitions,
  so a tiny 10k buffer constantly drops useful early experience and biases training
  toward recent (and already over-sampled) states.
- `EPS_DECAY` 10k → **50k**: with 10k the agent was already ~98% greedy after roughly
  the first minute, so it had no time to explore alternative ways around the x≈434
  obstacle the baseline keeps dying at.
- Loss `MSELoss` → **`SmoothL1Loss`** (Huber): standard for DQN, less sensitive to
  large TD-error outliers that are common with distance-based reward shaping.

**Why:** These are the "general settings" upgrade — the minimal set of modifications
that turn the textbook Nature-DQN into a modern DQN variant, without introducing
exotic machinery (no distributional heads, no noisy nets, no prioritized replay yet).
If this moves the needle, we have a better foundation to iterate from.

**Result:** `total_reward=596`, `max_x_dist=596`, `score=0`, `flag_get=False`,
`peak_vram_mb=115`. **+161 over baseline** and — importantly — the agent broke past
the x≈434 wall that has blocked every prior run on this problem. Keep. New foundation.

