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

## Exp3 — N-step returns (n=3)

**What changed:** The replay buffer now stores **3-step returns** instead of 1-step.
Each transition pushed is `(s_t, a_t, R_n = r_t + γ·r_{t+1} + γ²·r_{t+2}, s_{t+3}, done)`,
and the Bellman target uses `γⁿ` for the bootstrap term. Remaining partial windows
are flushed at episode end so we do not lose the last transitions.

**Why:** Super Mario's reward (distance + score) is effectively sparse relative to
the number of steps — with 1-step TD, the signal for "this jump helped me reach the
next pipe" has to propagate one Bellman backup at a time, which is very slow for a
10 min budget. N-step returns propagate reward `n` Bellman steps in a single update,
which is the main reason Rainbow (Hessel et al. 2017,
<https://arxiv.org/abs/1710.02298>) lists n-step as one of the highest-leverage DQN
add-ons. Combined with frame-skip=3 each update now looks ~9 raw frames ahead. We
keep `n=3` as a conservative starting point.

**Result:** `total_reward=315`, `max_x_dist=315`, `score=0`, `flag_get=False`.
**-281 vs Exp2 — discarded.** Hypothesis for the regression: with `n=3` the
distance-shaped reward (+2 per x-pixel) gets summed across 3 steps, effectively
tripling the TD-target magnitude, which destabilised the value estimates given that
`EPS_DECAY=50000` and early exploration already injects highly noisy returns. A
cleaner retry would either (a) normalise the distance reward, or (b) try `n=2`
first, or (c) couple n-step with gradient clipping. Reverted train.py to Exp2.

