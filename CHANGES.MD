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

## Exp4 — Gradient clipping (norm 10)

**What changed:** Added `torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)`
between `loss.backward()` and `optimizer.step()`. One line.

**Why:** Q-learning targets can spike (sparse big rewards like the +1000 flag bonus
or the +2/pixel distance shaping). Without gradient clipping, a single high-magnitude
TD error can push parameters far enough to destabilise the policy for many subsequent
updates. Gradient norm clipping is one of the standard ingredients in Rainbow
(Hessel et al. 2017, <https://arxiv.org/abs/1710.02298>) and almost every modern
DQN implementation. norm=10 is the value used in the original Rainbow code and is
gentle enough that it should not bottleneck learning while still cutting off
pathological updates. It also lays the groundwork for retrying n-step returns later
(Exp3 hypothesis: noisy n-step targets caused divergence — clipping should help).

**Result:** `total_reward=635`, `max_x_dist=435`, `score=200`, `flag_get=False`.
**+39 over Exp2 — kept.** Notable: first run to score non-zero in-game points
(score=200 from stomping an enemy), suggesting the policy is now stable enough to
interact with the environment instead of just running right. Slightly lower
`max_x_dist` than Exp2 (435 vs 596) but the score more than compensates and shows
richer behaviour. Clipping is keeping the net grounded.

## Exp5 — Dueling DQN

**What changed:** Replaced the single fully-connected head with two separate heads —
a 1-unit value head `V(s)` and an `n_actions`-unit advantage head `A(s,a)` — combined
as `Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)`. The convolutional trunk is unchanged.

**Why:** In dueling DQN (Wang et al. 2016, <https://arxiv.org/abs/1511.06581>), the
network separately learns "how good is this state?" from "how much does each action
matter here?". Many Mario states are dominated by state value (falling into a pit,
near the flag, stuck against a pipe) where all actions are similarly good or bad;
the value head generalises across these without having to update every action's
Q-value independently. Rainbow uses it; it routinely gives a few percent improvement
on Atari. Small code change, minimal VRAM bump (one extra linear layer of 1 unit).

**Result:** `total_reward=315`, `max_x_dist=315`, `score=0`, `flag_get=False`.
**-320 vs Exp4 — discarded.** Hypothesis: with only ~5 min of training, the dueling
architecture has roughly 2x the head parameters (two 3136→512 linears instead of
one) and can't amortise that cost fast enough. Dueling DQN's wins in the
literature assume tens of millions of steps; in our tiny budget the extra
un-trained parameters just add noise. Reverted train.py to Exp4; kept this section
as permanent history.

## Exp6 — 2-step returns (retry of Exp3 with gradient clipping)

**What changed:** Re-introduced n-step returns with `N_STEP = 2` (Exp3 tried n=3,
which blew up). The replay buffer now stores `(s_t, a_t, R = r_t + γ·r_{t+1},
s_{t+2}, done)` and the Bellman target uses `γ^N = γ²` for the bootstrap. An
n-step deque collects transitions inside the episode loop and flushes partial
windows at episode end so nothing is lost.

**Why:** Exp3 destabilised at n=3 — the distance-shaped reward (+2/pixel) summed
over 3 frame-skipped steps made TD targets too large for the unclipped gradients.
Two changes make the retry reasonable: (a) `N_STEP=2` is a smaller blow-up factor
than 3, (b) gradient clipping at norm 10 (Exp4) should now absorb any remaining
target spikes. N-step propagates reward `N` Bellman backups per update, which is
exactly what short training budgets need (Rainbow lists n-step as a high-leverage
addition, <https://arxiv.org/abs/1710.02298>). Starting conservative.

**Result:** `total_reward=1199`, `max_x_dist=899`, `score=300`, `flag_get=False`.
**+564 over Exp4 — huge win, kept.** `max_x_dist` nearly doubled (435 → 899) and
score continues to climb. The Exp3 hypothesis was correct: n-step targets diverge
without gradient control. With norm-10 clipping in place they work as advertised
and dramatically accelerate credit assignment for this short training budget.
`peak_vram_mb` unchanged at 115. New foundation for further experiments.

## Exp7 — 3-step returns (push n-step further)

**What changed:** `N_STEP = 2 → 3`. Nothing else.

**Why:** Exp6's jump from 635 → 1199 was the biggest step yet. n=3 propagates
reward three Bellman backups per update — even faster credit assignment. Exp3
originally tried n=3 and failed, but that was before gradient clipping; the
clipping (Exp4) should now absorb the larger target magnitudes. Worth ~30 seconds
to find out.

**Result:** `total_reward=314`, `max_x_dist=314`, `score=0`, `flag_get=False`.
**-885 vs Exp6 — discarded.** Fascinating: even with gradient clipping, n=3
produced a nearly identical result to the original Exp3 (315 then, 314 now). This
is not clipping's fault — clipping is working. It looks like n=3 simply pushes
the off-policy bias past a tipping point on this environment: at frame-skip=3,
an n-step return spans ~9 raw frames of action, during which the policy can
change enough that the trailing rewards no longer reflect what the current Q
thinks it would do. n=2 is a sweet spot. Reverted train.py to Exp6 (N_STEP=2).

## Exp8 — Frame skip 3 → 4

**What changed:** `FrameSkip(env, skip=3)` → `FrameSkip(env, skip=4)`. One char.

**Why:** skip=4 is the canonical Atari DQN value (Mnih et al. 2013/2015). Every
update now effectively sees 33% more game progress per step, so for a fixed
training budget the agent experiences more of the level. Mario specifically
benefits from longer skips because its useful actions (run, jump-right) are
naturally held across many frames. Trade-off: reaction time drops slightly. Also
note that with N_STEP=2 each replay transition now spans 2·4=8 raw frames of
lookahead versus 2·3=6 — slightly more aggressive credit assignment, but well
within what Exp6 showed is stable.

**Result:** `total_reward=304`, `max_x_dist=304`, `score=0`, `flag_get=False`.
**-895 vs Exp6 — discarded.** Combined with Exp7's failure this is a strong
signal that Exp6's stable zone is narrower than it looks: either more
lookahead per update (higher N) or fewer updates per game second (higher skip)
breaks it. Reverted to skip=3. Next ideas should stay within that envelope —
e.g. reward shaping, exploration schedule, or prioritized replay.

## Exp9 — Learning warmup (LEARN_START=1000)

**What changed:** Training updates no longer begin as soon as the replay buffer
has more than `BATCH_SIZE` (128) transitions — instead they wait until it has
1000. No other change.

**Why:** Starting to train on a buffer of 128 samples means the first few hundred
updates are effectively memorising a tiny, highly correlated window of mostly-
random-policy behaviour, with the net drifting toward bad Q-estimates before the
exploration has had time to produce any diversity. The canonical DQN setup
(Mnih et al. 2015) uses a warmup of 50k steps; we'd burn our whole budget on that,
but 1000 is a reasonable compromise — it's only ~5-10% of a full run and ensures
the first batch is drawn from a non-trivially diverse pool. Cheap to try.

**Result:** `total_reward=1499`, `max_x_dist=899`, `score=600`, `flag_get=False`.
**+300 over Exp6 — kept.** `max_x_dist` is identical to Exp6 (899) but the
in-game score has doubled (300 → 600), meaning the agent is spending more of its
time interacting (stomping enemies, collecting coins) rather than just running
right. Consistent with the intuition that a warmer initial replay avoids an early
"just-go-right" basin and lets the network develop richer behaviour.

## Exp10 — Larger batch size (128 → 256)

**What changed:** `BATCH_SIZE = 128 → 256`. One number.

**Why:** Larger mini-batches reduce gradient noise per update, which usually
translates into more stable Q-learning. Modern DQN implementations routinely use
batches of 256–512 on Atari. The forward pass is already tiny (4×84×84, 115 MB
VRAM total) so doubling the batch is essentially free. This also pairs well with
our n-step targets: each update now aggregates more independent n-step returns,
further damping any single bad target. Expected VRAM rise is minor.

**Result:** `total_reward=304`, `max_x_dist=304`, `score=0`, `flag_get=False`.
**-1195 vs Exp9 — discarded.** `peak_vram_mb=180` (up from 116) which is fine,
but the bigger batch roughly halved the number of gradient updates per wall
second. In a 5-minute budget, wall-clock gradient steps are the binding
constraint — not gradient variance. Smaller and faster wins here. Reverted.

## Exp11 — Faster ε decay (50000 → 20000)

**What changed:** `EPS_DECAY = 50000 → 20000`. One number.

**Why:** The ε-greedy schedule decays as `ε_end + (ε_start - ε_end)·exp(-steps/EPS_DECAY)`.
With `EPS_DECAY=50000` and a ~20k-step budget per run, at end of training
`ε ≈ 0.02 + 0.98·exp(-20k/50k) ≈ 0.02 + 0.98·0.67 ≈ 0.68` — i.e. the agent is
*still* acting randomly two-thirds of the time at the end of the run. That's
fine for a multi-hour run, but in our 5-min budget it means the net never gets
to trust its own policy. Setting `EPS_DECAY=20000` brings ε at end-of-run to
`0.02 + 0.98·exp(-1) ≈ 0.38`, and ε at the midpoint to `0.62` — still explorative
early, but actually exploits late. Expected: richer end-of-run trajectories.

**Result:** `total_reward=315`, `max_x_dist=315`, `score=0`, `flag_get=False`.
**-1184 vs Exp9 — discarded.** Decaying ε faster forces the agent to commit to
its policy before the Q-values are any good. Exp9's high-exploration tail
(ε≈0.68 at end of run) turns out to be a *feature*, not a bug, on this budget —
it keeps supplying diverse transitions even in the last third of training. The
rule of thumb "EPS_DECAY ≈ total_steps" from long-run Atari does not apply when
the network needs the last third of training just as much for data as for exploit.

## Exp12 — Death penalty (-25 on non-flag terminations)

**What changed:** In `DistanceReward.step`, if the episode ends without
`flag_get=True`, subtract 25 from the step's reward.

**Why:** Without a death penalty, from the agent's perspective death is just
"episode stops yielding more reward" — it's not actively *bad*, only *neutral*,
relative to standing still. This makes it easy for an early policy to learn
"jump into pit" as a fast reset. A modest penalty (-25 is standard in Mario RL
literature; compare ~50 pixels of forward distance reward) teaches "dying is
worse than surviving" without overwhelming the positive distance signal. Notably
this does not affect the flag case, so the +1000 flag bonus is still the
dominant objective. Minimal code change; orthogonal to all prior experiments.

**Result:** `total_reward=1284`, `max_x_dist=684`, `score=600`, `flag_get=False`.
**-215 vs Exp9 — discarded.** Same score (600) but less distance (684 vs 899).
Hypothesis: in an ε-greedy agent that's still exploring ~60% of the time, death
is often caused by *exploration*, not by a considered decision. Penalising it
punishes the random actions that were our main source of useful data. The
heuristic only pays off later when the agent has committed to a greedy policy,
which we don't reach in 5 min. Reverted.

## Exp13 — Higher EPS_END floor (0.02 → 0.1)

**What changed:** `EPS_END = 0.02 → 0.1`. One number.

**Why:** The current best (Exp9, 1499) consistently stalls around `max_x_dist=899`
— almost certainly a tall pipe or similar obstacle that needs a well-timed jump.
The random action floor of 2% is probably too low to keep discovering such timing
tricks within the 5-min budget. Raising it to 10% means roughly one in every ten
actions is a random exploration probe, which should give the replay buffer enough
jumping trials for the network to see the pipe-crossing and reinforce it.
Explicitly: this does *not* speed up decay, it only raises the floor — early
training is unchanged (ε is well above 0.1 anyway), we only differ at the tail.

**Result:** `total_reward=906`, `max_x_dist=706`, `score=200`, `flag_get=False`.
**-593 vs Exp9 — discarded.** Bumping the ε floor cost us both distance and
score. The effect was probably two-fold: (1) more random actions late in
training means fewer *exploit* episodes the net can learn its own greedy value
from, and (2) the final saved model carries no benefit from the extra random
probes since evaluate.py runs it greedily. Net loss. Reverted.

## Exp14 — Smaller batch (128 → 64)

**What changed:** `BATCH_SIZE = 128 → 64`. One number.

**Why:** Exp10 established that our bottleneck is gradient-steps-per-second, not
gradient variance: doubling the batch to 256 tanked the run. The natural next
question is whether the reverse helps — halving the batch to 64 should roughly
double the number of updates the optimizer gets in the same wall-clock time.
Variance per update goes up, but with gradient clipping and n-step returns
already smoothing things, the net should still be stable. If this wins, it's
a pure throughput gain.

**Result:** `total_reward=635`, `max_x_dist=435`, `score=200`, `flag_get=False`.
**-864 vs Exp9 — discarded.** So BATCH_SIZE is not monotonic: 128 beats both
256 (Exp10) and 64. The "Exp9 is on a sharp edge" pattern continues. What this
tells me: we are *not* purely update-bound — gradient quality matters too, and
64 is too noisy for the net to track a moving target.

## Exp15 — Kaiming (He) weight initialization

**What changed:** Added an explicit Kaiming normal init for all `Conv2d` and
`Linear` layers, with biases zeroed. PyTorch's default for `Conv2d` / `Linear`
is uniform with `fan_in` scaling (Kaiming uniform for the weights). Kaiming
*normal* with `nonlinearity='relu'` (rather than the default `leaky_relu`
assumption) is the textbook init for ReLU conv nets (He et al. 2015,
<https://arxiv.org/abs/1502.01852>) and preserves signal variance more cleanly
across the forward pass.

**Why:** With 5 min of training we only get ~20k updates. Any improvement in
early-gradient quality compounds over the budget. This is a zero-cost change at
runtime (the init runs once), has no interaction with the update rate (so it
shouldn't hit the sharp-edge failure mode the last few experiments showed), and
is completely orthogonal to everything else in Exp9's stack.

**Result:** `total_reward=1599`, `max_x_dist=899`, `score=700`, `flag_get=False`.
**+100 over Exp9 — kept.** Same `max_x_dist` (the agent is still hitting the
~x=899 obstacle) but the score climbed 600 → 700 — more enemies stomped along
the way. Modest but positive, and the mechanism is orthogonal so it stacks with
future experiments. Noted: the persistent x=899 wall is clearly the next thing
to attack — every experiment since Exp6 plateaus there.

## Exp16 — Weight decay 1e-5

**What changed:** Passed `weight_decay=1e-5` to the Adam optimizer.

**Why:** Very mild L2 regularization. With only ~20k updates against a finite
replay buffer, the net has every opportunity to overfit to transient state
distributions (e.g. those few episodes where it got lucky past some enemy).
Weight decay shrinks parameters toward zero, which in practice smooths the
learned Q-function and can improve generalization to states the agent hasn't
seen much of (like everything past x=899). 1e-5 is gentle — it should barely
touch trained weights but pull unused weights toward zero. Cheap, orthogonal.

**Result:** `total_reward=509`, `max_x_dist=309`, `score=200`, `flag_get=False`.
**-1090 vs Exp15 — discarded.** Surprising. Hypothesis: Q-learning targets are
themselves produced by the network, so shrinking weights also shrinks the
bootstrap target — effectively a drag on the value estimates we're trying to
grow. Weight decay plays less well with TD learning than with supervised
learning; this is a known issue in the literature. Reverted.

## Exp17 — Frontier-only distance reward

**What changed:** `DistanceReward` now only gives forward-progress reward when
`x_pos` exceeds the episode's previous maximum, rather than rewarding every
positive step-delta. Going back and forth yields no net reward. The tracking
field `curr_x` was renamed to `max_x` to reflect its new monotonic meaning.

**Why:** The previous shaping rewarded `(x_pos - curr_x) * 2.0`, which means an
agent that oscillates forward-and-back-and-forward can farm positive reward on
each forward step and pay the negative cost on each backward step. In practice
the two should roughly cancel, but TD learning treats them asymmetrically — the
positive and negative rewards end up in different transitions and the Q-values
can learn to pursue oscillation. Rewarding *only* new-frontier pixels
explicitly aligns the training signal with `max_x_dist`, which is exactly what
evaluate.py scores against for non-flag runs. Also removes a tiny bit of
variance (revisiting old x no longer injects reward into training updates).

**Result:** `total_reward=535`, `max_x_dist=435`, `score=100`, `flag_get=False`.
**-1064 vs Exp15 — discarded.** The asymmetry turned out to be important:
removing the negative reward for moving left also removes the penalty for
*drifting* left (e.g. while dodging an enemy), so the agent's greedy policy
becomes less directional. Going back cheaply is a bug, not a feature. Reverted.

## Exp18 — Two gradient updates per env step

**What changed:** Added `UPDATES_PER_STEP = 2` and wrapped the training block
in a loop that samples a fresh batch and updates twice per env step instead of
once. No other change.

**Why:** The replay buffer already contains far more transitions than any single
epoch consumes (50k buffer vs. ~20k total updates per 5-min run). We are leaving
sample efficiency on the table — each env step could drive more gradient
descent. This is a direct increase in total gradient updates while *also*
keeping env exploration diversity unchanged (unlike Exp14 which reduced batch).
Trade-off: wall-clock training time per episode is longer, so fewer env steps
will fit in the budget — but given each replay batch is independent and GPU
utilization is already low (115 MB peak), doubling updates is nearly free.

**Result:** `total_reward=595`, `max_x_dist=595`, `score=0`, `flag_get=False`.
**-1004 vs Exp15 — discarded.** "Nearly free" was wrong: doubling the
optimizer.step() per env step roughly halves the env-steps-per-second the
training loop achieves, which means *less* data over the same wall budget.
Score collapsing back to 0 confirms the agent didn't get to interact much.
With only ~10k env steps instead of ~20k, the effective experience budget is
the binding constraint, not the gradient count. Reverted.

## Exp19 — Save best snapshot, not final model

**What changed:** During training we now track an eval-aligned per-episode metric
`episode_last_score + episode_max_x` (this is exactly what `evaluate.py` returns
for non-flag runs). After every episode, if the metric beats our running best,
we deep-copy the policy net's state_dict to a CPU snapshot. At the end of training
the saved `MODELS/model.pt` is the best-snapshot weights, not the final ones.

**Why:** RL training is inherently noisy — the policy can be at peak skill mid-
run and then drift. Saving the *final* weights leaves performance to chance. Most
serious RL implementations save the best checkpoint by validation reward; we
don't have a separate validation rollout, but the score+max_x signal from training
episodes is a free, informative proxy. Zero impact on training dynamics, no extra
forward passes, only a copy and a write at episode boundaries — should be a strict
upgrade or noise-neutral.

**Result:** `total_reward=1734`, `max_x_dist=1434`, `score=300`, `flag_get=False`.
**+135 over Exp15 — kept.** Two big things: (1) `max_x_dist` nearly doubled
(899 → 1434), the persistent wall is finally broken; (2) the internal
`saved_snapshot_metric` shows **19157** during training — that can only happen
if a training episode actually finished the level (score 18000+ on a flag-get).
The saved snapshot can't reliably *reproduce* the flag greedily (eval is
deterministic, and some of that finish was almost certainly lucky ε-exploration),
but the same weights are clearly stronger than whatever the loop happened to
land on at minute 5. Clear evidence that final-weights snapshotting was leaving
performance on the table. New foundation.

## Exp20 — Snapshot filter: only after ε has decayed (ε < 0.75)

**What changed:** The best-snapshot update in Exp19 now requires
`eps_threshold < 0.75` in addition to beating the metric. Early high-exploration
episodes can no longer set the snapshot.

**Why:** Exp19 logged `saved_snapshot_metric=19157` (a flag-completing episode),
but the saved policy only achieved `total_reward=1734` in greedy evaluation.
Likely explanation: the snapshotted episode finished the level *because* of
lucky ε-greedy actions, not because the underlying greedy policy was strong.
We want snapshots from a regime where the agent's training behaviour is closer
to what `evaluate.py` measures (deterministic greedy). With `EPS_DECAY=50000`
and ~20k steps per run, ε decays as `0.02 + 0.98·exp(-steps/50000)` — at step
~14k that's about 0.74, so the threshold 0.75 selects the back roughly-half of
training, where the greedy actions dominate ε-greedy ones.

**Result:** `total_reward=1509`, `max_x_dist=1509`, `score=0`, `flag_get=False`.
**-225 vs Exp19 — discarded.** Interesting mixed result: `max_x_dist` is
actually *higher* (1509 vs 1434) and `saved_snapshot_metric` dropped to 2776,
which proves the filter did its job and selected a non-lucky episode. But
score collapsed to 0, dragging total below Exp19. The interpretation is that
ε-greedy lucky episodes were giving us *both* longer reaches *and* enemy
stomps, and the strict greedy filter picks a snapshot that walks far but
doesn't interact. Per the rule total_reward decides — reverted.

## Exp21 — Soft target updates (τ=0.005)

**What changed:** Replaced the periodic hard target copy
(`target_net.load_state_dict(policy_net.state_dict())` every 1000 steps) with
Polyak averaging at every gradient step:
`target = (1-τ)·target + τ·policy`, with `τ=0.005`. Removed `TARGET_UPDATE`.

**Why:** Hard target copies create discontinuities — the target net is stale
for 1000 updates, then suddenly jumps to the current policy. Q-targets shift
abruptly each time, which the value head has to chase. Soft updates (τ=0.005,
the standard DDPG/SAC value) move the target tiny amount per step, so the
Q-targets evolve smoothly. Effective "half-life" of the soft update is roughly
`ln(2)/τ ≈ 140` steps, which is comparable to the old 1000-step hard refresh
but in a moving-average sense. Cost: a few extra parameter copies per
gradient step (cheap on this small net). Stability win expected, especially
because we're already running with grad clipping and n-step returns that benefit
from a stable bootstrap.

**Result:** `total_reward=594`, `max_x_dist=594`, `score=0`, `flag_get=False`.
**-1140 vs Exp19 — discarded.** `saved_snapshot_metric=20555` (training again
finished the level once), but the greedy snapshot collapses. Hypothesis: soft
updates make the target net move *with* the policy almost in lockstep at this
high `τ`, which weakens the "fixed-target" stabilization that vanilla DQN
relies on. With our small budget and no PER, the target needs to be staler than
this. Sticking with hard 1000-step copies. Reverted.

## Exp22 — Periodic greedy validation episodes for snapshot

**What changed:** Every 5th training episode (`episode_idx % 5 == 0`) is now a
*validation* episode that runs with `ε=0` (fully greedy, no exploration). All
other episodes are unchanged. The best-snapshot save is now gated on validation
episodes only — only their `score + max_x` can update the saved snapshot.

**Why:** Exp19's snapshot picked an ε-greedy episode that completed the level
*because of* random actions; the same weights couldn't reproduce the win in
greedy eval (`saved_snapshot_metric=19157` vs `total_reward=1734`). Exp20's
naive ε-threshold filter never fired (ε never gets below 0.75 in 5 min).
Forcing one in five episodes to be greedy creates an honest sample of "what
the current policy actually does when it can't gamble." Snapshots saved from
those episodes should generalise to evaluate.py's greedy rollout, since
they're produced *exactly* the same way. Cost: ~20% of training episodes do
no exploration, so the replay buffer gets slightly less diverse. Worth it if
the snapshot quality goes up.

**Result:** `total_reward=800`, `max_x_dist=700`, `score=100`, `flag_get=False`.
**-934 vs Exp19 — discarded.** Validation rollouts measure greedy quality on
the *training* env (random reset state), but `evaluate.py` uses fixed seeds
from `EVAL_SEEDS` in constants — a different world. So even an "honest"
greedy validation metric in training doesn't predict eval performance. On top
of that, 20% of episodes contributing zero exploration noticeably starved the
replay buffer. Reverted. Lesson: snapshot selection is fundamentally limited
without an in-distribution validation signal — we can only optimize for the
*expected* greedy reward, which is what Exp19 already does.

## Exp23 — Higher gamma (0.99 → 0.995)

**What changed:** `GAMMA = 0.99 → 0.995`. One number.

**Why:** The flag bonus is +1000 — by far the largest single reward in the
shaped scheme — but it only fires *at* the flag. With γ=0.99 over an episode
of (say) 800 steps, that bonus is discounted by `0.99^800 ≈ 3e-4`, so by the
start of the level it's worth less than a single distance-pixel. Bumping to
γ=0.995 raises that to `0.995^800 ≈ 0.018`, ~60× more long-horizon weight.
That's enough to actually pull the value function toward "go right and
finish." Risk: longer credit horizons can also amplify off-policy noise from
the n-step returns. Worth a try given Exp19 already shows training reaches the
flag occasionally — the question is whether the *value function* learns to
expect it.

**Result:** `total_reward=40`, `max_x_dist=40`, `score=0`, `flag_get=False`.
**-1694 vs Exp19 — discarded.** Catastrophic. The bumped γ amplified Q-target
magnitudes (n-step bootstraps now multiply by `γ²=0.990` instead of `0.980`,
and propagation is much further), and despite gradient clipping it diverged
into an "agent stands still" basin. The Exp19 sweet spot is genuinely
narrow — even modest discount changes break it. Reverted.

## Exp24 — Halve learning rate (1e-4 → 5e-5)

**What changed:** `LR = 1e-4 → 5e-5`. One number.

**Why:** Several recent failures (Exp21 soft target, Exp23 higher gamma) have
been Q-divergence collapses, suggesting the value function is on the edge of
instability. A smaller step size is the cheapest defense — each gradient
update perturbs Q-values less, so the moving target chases the policy more
gently. Risk: with only ~20k updates per run, halving LR halves the total
parameter motion, so we might end the budget under-trained. But Exp19's
network is already good enough that the agent finishes the level
*occasionally* — we may not need more learning, just more *consistent*
learning. Standard DQN papers (Mnih) use 2.5e-4 with RMSprop; modern Adam-
based runs typically use 1e-4 to 5e-5 — both within the band.

**Result:** `total_reward=301`, `max_x_dist=301`, `score=0`, `flag_get=False`.
**-1433 vs Exp19 — discarded.** Halving LR slowed learning enough that the
greedy snapshot we saved was from very early in training (when ε was high,
so the metric was probably driven by random luck), and that snapshot had no
real policy yet. Confirms the "Exp9-Exp19 stack is at a sharp ridge" pattern:
even slowing the optimizer breaks it.

## Exp25 — Prioritized Experience Replay (proportional, α=0.6, β=0.4)

**What changed:** Replaced the uniform `ReplayBuffer` with a
`PrioritizedReplayBuffer` that samples transitions proportional to
`priority^α` and corrects the bias with importance-sampling weights
`(N·P_i)^(-β)`. New transitions get max priority (so they're seen at least
once). After each gradient step, the sampled transitions' priorities are
updated to `|td_error| + 1e-6`. The loss uses element-wise smooth-L1 weighted
by the IS weights. No sum tree — for our 50k buffer, `np.random.choice`
with the priority array is fast enough.

**Why:** The training reaches the flag occasionally (Exp19's
`saved_snapshot_metric=19157`), but the rare obstacle-crossing transitions
are drowned out by 50k mundane "running right" transitions when sampled
uniformly. PER (Schaul et al. 2016, <https://arxiv.org/abs/1511.05952>)
focuses learning on transitions with high TD error — exactly the rare ones
where the network's current Q is wrong, which includes both novel obstacle
crossings and any policy improvements. This is the canonical Rainbow trick
for short training budgets and is the most likely lever to push the
greedy policy past Exp19's 1734 ceiling. α=0.6 / β=0.4 are the values from
the original paper (I keep β fixed rather than annealing — too short a
budget for annealing to matter).



