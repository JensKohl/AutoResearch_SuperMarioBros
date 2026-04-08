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

## Experiment 4 — Death penalty, stronger distance reward, higher LR (discarded)

**What changed:**
- Added death penalty: `-50` reward when Mario loses a life (`info['life']` decreases).
- Increased distance reward multiplier: `2.0 → 3.0`.
- Increased flag bonus: `1000 → 5000`.
- Increased learning rate: `1e-4 → 3e-4`.

**Why:** With eval getting `score=0` and `max_x_dist=315`, the agent moves right but doesn't push through obstacles. Explicit death penalty + stronger rightward incentive should teach the agent to avoid dying while pressing forward.

**Result:** total_reward=310 vs 315 — marginally worse. Discarded.

## Experiment 5 — Dueling DQN architecture

**What changed:** Replaced the single FC output head with two separate streams:
- **Value stream**: estimates V(s) — how good is being in this state
- **Advantage stream**: estimates A(s,a) — how much better is each action vs average
- Combined as: `Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))`

**Why:** Dueling DQN ([Wang et al. 2016](https://arxiv.org/abs/1511.06581)) helps in states where the choice of action doesn't matter much (e.g., running on flat ground). The value stream can learn general "how good is this state" independently from the advantage stream's "which action is best". This typically improves sample efficiency — important when we only have 5 minutes of training.


## Experiment 6 — Dueling + Double DQN targeting

**What changed:** Added Double DQN targeting to the existing Dueling DQN. The target Q-value now uses the policy net to select the best next action, and the target net only to evaluate it:
```python
next_actions = policy_net(next_states).max(1)[1]
next_q_values = target_net(next_states).gather(1, next_actions)
```

**Why:** Standard DQN overestimates Q-values because it uses the same (max) operation for both selecting and evaluating actions. Double DQN ([van Hasselt et al. 2015](https://arxiv.org/abs/1509.06461)) decouples these, reducing overestimation bias. Dueling + Double DQN is a classic combination. (Earlier Exp2 tried this but failed due to other combined changes; here we isolate it.)

## Experiment 7 — 2 gradient updates per environment step (replay ratio=2)

**What changed:** The training loop now performs 2 gradient updates per environment step instead of 1.

**Why:** With only 147MB VRAM in use and GPU at ~76°C, the GPU is mostly idle between environment steps. The replay ratio (gradient steps per env step) is a key sample efficiency lever. Doubling it means 2x more learning from the same experience, without collecting more data. This is a standard technique in modern RL (see [Fedus et al. 2020 "Revisiting Fundamentals of Experience Replay"](https://arxiv.org/abs/2007.06700)).

**Result:** total_reward=315 vs 554 — major regression. Likely too many gradient steps relative to env steps caused overfitting/instability. Discarded.

## Experiment 8 — Score bonus in training reward

**What changed:** Added `score_delta * 0.3` to the training reward. When Mario hits a ? block, collects a coin, or kills an enemy, the in-game score increases and now generates a training signal.

**Why:** The evaluation metric is `score + max_x_dist`. Our training reward only incentivizes moving right. Score=0 in eval suggests Mario is running under blocks without jumping. Adding score bonus:
1. Aligns training reward with eval metric.
2. Encourages jumping at blocks — which also teaches Mario to jump over obstacles.
3. Killing enemies (Goombas) removes them from the path, potentially allowing more progress.

Coefficient 0.3 means killing a Goomba (100 score pts → +30 reward) is significant but doesn't dominate the distance signal.

## Experiment 9 — Mild death penalty (-15)

**What changed:** Added a death penalty of -15 when Mario loses a life (`info['life']` decreases).

**Why:** The agent stalls around x=700 (first pipe + Goomba cluster in World 1-1). Currently dying has no explicit negative signal — the agent only loses future rewards. A death penalty helps the agent explicitly learn to avoid dying, which should help it navigate the harder section. Coefficient -15 is mild to avoid overpowering the distance and score signals.

**Result:** total_reward=314 vs 1600 — severe regression. Death penalty (tried twice: -50 in Exp4, -15 here) consistently hurts. Agent becomes overly cautious near x=700. Do not try death penalties again. Discarded.

## Experiment 10 — Increase score bonus coefficient (0.3 → 0.5)

**What changed:** Raised the score bonus coefficient from 0.3 to 0.5.

**Why:** Exp8 showed that score bonus 0.3 gave a huge jump (554→1600). A higher coefficient means the agent gets stronger signal to kill enemies and collect coins — which also teaches it to jump. With a Goomba kill worth 100 score pts, coefficient 0.5 gives +50 reward per kill, making it ~25 pixels worth of distance signal. This might encourage more aggressive interaction with the level.

**Result:** total_reward=534 vs 1600 — regression. Higher score signal causes agent to farm score near the start instead of progressing. 0.3 is the sweet spot. Discarded.

## Experiment 11 — Higher discount factor (GAMMA 0.99 → 0.995)

**What changed:** Raised the discount factor from 0.99 to 0.995.

**Why:** A higher γ makes the agent care more about future rewards. With γ=0.99, a reward 200 steps away is worth only 0.99^200 ≈ 0.135 today. With γ=0.995, it's worth 0.995^200 ≈ 0.37 — nearly 3× more valued. The level is long (~3000 pixels) so the agent needs to plan far ahead. A longer horizon should help it push through challenging sections instead of settling for local reward.

**Result:** total_reward=1299 vs 1600. max_x_dist improved (700→899 best ever!), but score dropped (900→400). Longer horizon makes agent push further right but skip score. Net negative. Discarded. Will try compromise GAMMA=0.992.

## Experiment 12 — GAMMA compromise (0.99 → 0.992)

**What changed:** GAMMA=0.992, midpoint between 0.99 (best total_reward=1600) and 0.995 (best x_dist=899).

**Why:** γ=0.995 pushed distance further (899) but reduced score collection. γ=0.99 kept both balanced (700 dist, 900 score). γ=0.992 is a compromise — slightly longer horizon without fully abandoning the score signal.

**Result:** total_reward=315 — severe regression. GAMMA is very sensitive; 0.99 is optimal. Do not change GAMMA. Discarded.

## Experiment 13 — Soft target network update (Polyak averaging)

**What changed:** Replaced hard target network copy every 500 steps with Polyak averaging every step: `θ_target = τ*θ_policy + (1-τ)*θ_target` where τ=0.005.

**Why:** Hard target updates create discontinuous jumps in the target, which can destabilize Q-value estimates. Soft updates (used in TD3, SAC, and DDPG) produce a smoothly-moving target. With τ=0.005, the target net tracks the policy net with a lag of ~200 steps (1/τ), similar to the 500-step hard update but smoother. Standard in modern deep RL.

**Result:** total_reward=315 — regression. Per-step parameter copying may add overhead or the soft update pace doesn't suit 5-minute training. Discarded.

## Experiment 14 — Reduced action space (RIGHT_ONLY, 4 actions)

**What changed:** Replaced PRO_MOVEMENT (6 actions) with RIGHT_ONLY (4 actions): `['right', 'right+A', 'right+B', 'right+A+B']`. Removed `['A']` (jump in place) and `['A', 'B']` (run-jump in place).

**Why:** Removing actions that are almost never useful in World 1-1 simplifies the learning problem. The agent doesn't need to jump in place — it should always be moving right. Fewer actions = smaller output head = easier Q-value estimation. Note: evaluate.py imports `make_env` from train.py, so this change applies consistently to both training and evaluation.

**Result:** total_reward=909. max_x_dist=309, score=600. Agent farms score near start but can't progress — blocking obstacles can't be navigated without precise jump control. Discarded.

## Experiment 15 — Larger network hidden size (512 → 1024)

**What changed:** Increased the hidden layer size in both value and advantage streams from 512 to 1024 units.

**Why:** With only 147MB VRAM currently, there's room for a larger network. More parameters = more capacity to represent complex state-action mappings. World 1-1 has varied terrain (pipes, enemies, platforms) that might require higher-capacity representations. The model parameter count roughly doubles, expected VRAM ~250-280MB — still well within limits.

**Result:** total_reward=315, VRAM=213MB. Larger network needs more data than 5-minute budget allows. Discarded.

## Experiment 16 — Warmup period + larger buffer (MEMORY_SIZE 20k→30k)

**What changed:**
- Added warmup period: no gradient updates for the first 2000 steps (agent acts randomly during this time).
- Increased replay buffer from 20k to 30k transitions.

**Why:** Starting to learn from just 128 random transitions is early — the initial training data is very limited and can lead to unstable early learning. A 2000-step warmup ensures the buffer has diverse experiences before any gradient updates happen. Larger buffer improves long-term diversity without changing the core algorithm.

**Result:** total_reward=534. Warmup period removes 2000/12000 training steps (17%), reducing learning time significantly. Not worth it. Discarded.

## Experiment 17 — Save best-model checkpoint during training

**What changed:** Instead of saving the final policy net, save the model whenever a new best x-position is reached in any training episode. The model file always holds the best policy seen so far during training.

**Why:** DQN training is non-monotonic — the policy can degrade after peak performance (catastrophic forgetting, over-exploitation). If the agent reached x=900 mid-training but ended at x=400, we'd previously save the x=400 model. Now we save the x=900 model. This is "best-of-training" evaluation rather than "end-of-training".

**Result:** total_reward=314. Interesting: training BestX=1677 (!!) but eval=314. Root cause: the best x was achieved at epsilon=0.104 — exploration luck doesn't generalize to greedy evaluation. Checkpointing based on exploration performance is misleading. Discarded.

## Experiment 18 — "New ground" bonus (session-wide max x record)

**What changed:** Added a `global_max_x` that persists across episodes. When Mario breaks the session-wide furthest-right record, he gets a bonus of `(x_pos - global_max_x) * 5.0`.

**Why:** In Exp17, we saw the agent reach x=1677 during exploration but eval only got x=314. The greedy policy never learned to jump past the obstacle at x=314. The new-ground bonus creates a strong, targeted incentive to push past obstacles: jumping over the x=314 pipe gives a one-time reward proportional to how far past the record it gets. This is a form of "curiosity" or "novelty" reward — intrinsic motivation to explore new territory. Once the record is set, the normal distance reward maintains it.

**Result:** total_reward=435, score=0. New-ground bonus dominates score signal — agent rushes right without collecting anything. Discarded.

## Experiment 19 — N-step returns (n=3)

**What changed:** Added a `NStepBuffer` that accumulates 3 transitions, computes the 3-step discounted return `G = r₀ + γr₁ + γ²r₂`, and stores `(s₀, a₀, G, s₃, done)` in the main replay buffer. The TD target uses `γ³` for bootstrapping: `Q_target = G + γ³ * max_a Q(s₃, a)`.

**Why:** Standard 1-step DQN only propagates the reward from the very next step. N-step returns carry the reward signal further back in time, helping the agent learn that actions taken now lead to rewards 3 steps later. This is especially important for Mario where jumping over a pipe (action at step t) pays off several steps later when you're safely past it. See [Mnih et al. 2016 "Asynchronous Methods"](https://arxiv.org/abs/1602.01783).

**Result:** total_reward=314 — regression. N-step returns introduced bugs or disrupted the balance. Discarded.

## Experiment 20 — Greedy-checkpoint: save best model when eps < 0.15

**What changed:** Save the model (checkpoint) whenever a new best episode x-position is achieved AND epsilon < 0.15 (mostly-greedy). Fallback to saving final model if greedy phase never triggers.

**Why:** Exp17 revealed: training BestX=1677 at eps=0.104, but final model only got x=314 in eval. The model from step ~8630 was better than the model at end-of-training. Saving when eps<0.15 captures the policy from when the agent is mostly greedy (>85% deterministic), avoiding the pure-exploration phase artifacts. The threshold 0.15 corresponds to ~step 6650 (eps_decay=3000, ~166s into training), giving ~134s of greedy-checkpoint window.

**Result:** BestGreedyX=1431 during training, but eval only x=308. The model checkpoint at eps=0.10 still can't reproduce its performance with eps=0.00. Root cause: successful "jump past obstacle" transitions are too rare in the buffer (~10 out of 12k) and sampled only ~1% per update. Moving to PER (Exp21) to address this. Discarded.

## Experiment 21 — Prioritized Experience Replay (PER, α=0.6)

**What changed:** Replaced the uniform replay buffer with a Prioritized Experience Replay buffer. Transitions are sampled proportional to their TD error (`priority = |δ|^α` with α=0.6). Priorities are updated after each gradient step.

**Why:** Analysis of Exp17/20 revealed the core problem: successful "jump past obstacle" transitions are rare (~10 out of 12k total) and get sampled only ~1% per gradient update. With uniform sampling, the agent can never learn from these rare success cases. PER fixes this by sampling high-TD-error transitions (i.e., the "surprises" where the agent performed much better than expected) more frequently. See [Schaul et al. 2016](https://arxiv.org/abs/1511.05952). No importance sampling weights for simplicity (slight bias but workable).

**Result:** total_reward=313 — PER didn't help. The core exploration-exploitation gap persists. Discarded.

## Experiment 22 — Faster epsilon decay (EPS_DECAY 3000→2000)

**What changed:** Reduced EPS_DECAY from 3000 to 2000. With 12k total training steps, epsilon now hits ~0.10 by step ~4600 (vs ~6600 with 3000). The agent spends more of its training time in the mostly-greedy regime, getting more reinforcement signal for the greedy policy's actual decisions.

**Why:** The greedy policy gets stuck at x=314. More training time as the greedy policy = more chances to reinforce "jump at x=314" if the agent ever learns it. Faster decay means the greedy policy is more thoroughly trained.

**Result:** total_reward=594, score=0. Faster decay → more distance, less score. Confirms: score learning requires exploration (eps~0.3-0.5 range). Discarded.

## Experiment 23 — Warm-start from previous model.pt

**What changed:** If MODELS/model.pt exists, load it as the initial policy network weights instead of random init. Reduce EPS_START to 0.5 (since the agent already has a learned policy, less exploration is needed). The target net is also initialized from the loaded model.

**Why:** Each training run throws away the learned policy and starts from scratch. Warm-starting allows the agent to build on what it learned in the previous run. The prior run reached x=594 consistently — starting from that policy with more training should let it push further. EPS_START=0.5 preserves enough exploration to discover improvements while leveraging the existing policy.

**Result:** total_reward=894, score=300, x=594. Warm-start improved from the 594 starting model (score went 0→300). Progressive improvement visible. But still below 1600 best. model.pt preserved for Exp24 warm-start chain. Discarded (< 1600).

## Experiment 24 — Warm-start chain iteration 2 (EPS_START=0.3)

**What changed:** Same warm-start from model.pt, but with EPS_START=0.3 (less exploration than Exp23's 0.5). Current model.pt got 894 in eval. With 30% exploration, the agent mostly exploits the existing policy while still discovering improvements.

**Why:** Progressive warm-start chain: 594→894 (Exp23 with 0.5). Now trying to push further with less exploration. The existing policy is better than random — exploiting it more should refine it further.

**Result:** total_reward=434, score=0. Warm-start from a poor model with low exploration fails. The 1600 model.pt was lost; the warm-start chain can't recover without a high-quality starting model. Discarded.

## Experiment 25 — Fixed random seed (seed=42) for reproducibility

**What changed:** Added fixed seeds at the start of training: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `cuda.manual_seed(42)`.

**Why:** The 1600 result from Exp8 might be a lucky run (training has high variance). Fixed seeds make each run deterministic, allowing us to understand if the algorithm reliably achieves 1600+ or if it depends on random luck. If seed 42 gives ~1600, the algorithm is good. If it gives 314, the 1600 was a lucky initialization.

**Result:** total_reward=706, score=0, dist=706. Key diagnostic: algorithm CONSISTENTLY reaches x≈700 but score is random. The 1600's score=900 was favorable random exploration. Need to make score collection more reliable. Discarded (< 1600).

## Experiment 26 — Score bonus 0.3 → 0.4 (targeted for score reproducibility)

**What changed:** Score bonus coefficient raised from 0.3 to 0.4.

**Why:** Diagnostic (Exp25/seed=42) showed: algorithm consistently reaches x≈700 but score is unreliable. Exp8's 1600 had score=900. Score=0 leaves 900 points on the table. A stronger score signal (0.4 vs 0.3) might make score-collecting jumps more consistently rewarded. Note: 0.5 was too high (Exp10: 534), 0.3 was lucky (Exp8: 1600). Trying 0.4 as middle ground.

**Result:** total_reward=994, score=400, dist=594. Confirmed score/distance tradeoff: higher score bonus → less distance. 0.3 remains best coefficient for total_reward. Discarded.

## Experiment 27 — Score bonus 0.3 → 0.2 (weaker, more distance-focused)

**What changed:** Score bonus coefficient from 0.3 to 0.2.

**Why:** Confirmed trade-off: 0.5→534 (dist=434), 0.4→994 (dist=594), 0.3→1600 (dist=700, lucky). The 0.3 result was lucky — with a fixed seed it gives 706 dist but 0 score. A weaker score signal (0.2) prioritizes distance more while still giving some score incentive. Hypothesis: with 0.2, the agent might consistently get dist=800-900 with score=100-200, beating 1000+ total.

**Result:** total_reward=315, score=0. 0.2 too weak. Score bonus optimum confirmed at 0.3. Discarded.

## Experiment 28 — Switch to PPO (Proximal Policy Optimization)

**What changed:** Complete algorithm switch from DQN (value-based, off-policy) to PPO (policy gradient, on-policy). Key components:
- **ActorCritic network**: shared conv backbone, separate actor (logits) and critic (value) heads. `forward()` returns logits for evaluate.py compatibility (`DQN = ActorCritic` alias).
- **Collect N=512 steps**, compute GAE advantages (λ=0.95), then update with PPO clipping (ε=0.2) for 4 epochs in mini-batches of 64.
- **Entropy bonus (0.01)**: built-in exploration incentive — no epsilon-greedy needed.
- **GAE**: reduces variance in advantage estimates → more stable policy updates.
- Kept same reward shaping (DistanceReward, score bonus 0.3).

**Why:** DQN's exploration-exploitation gap was consistently preventing the greedy policy from replicating the training policy's performance (BestX=1677 during training but eval=314). PPO directly optimizes the policy via policy gradient — the greedy policy IS the learned policy, no gap. Entropy bonus ensures consistent exploration without epsilon-greedy's instability. PPO is the standard algorithm for 2D platform games. See [Schulman et al. 2017](https://arxiv.org/abs/1707.06347).

**Result:** total_reward=296, score=0. 137 episodes with uniform 1255.9 reward = converged to fixed policy but can't reach far in eval. PPO is less data-efficient than DQN+replay in 5-minute regime. Discarded. Back to DQN.

## Experiment 29 — Boltzmann (softmax) exploration

**What changed:** Replaced epsilon-greedy with Boltzmann exploration. Instead of binary random/greedy, actions are sampled proportional to `softmax(Q(s,a) / T)` where temperature T decays from 2.0 to 0.05 with the same schedule as epsilon.

**Why:** Epsilon-greedy wastes exploration budget on pure random actions. With Boltzmann:
- At high T: near-uniform sampling (lots of exploration, but guided by Q-values)
- At low T: nearly greedy (best action almost always chosen)
- Transitions are smooth — no sudden switch from "explore" to "exploit"
- Crucially: if Q(jump) > Q(run) even slightly, jump is preferred even during "exploration"

This may help the agent discover jumping more consistently because the Q-value signal from early successful jumps immediately biases future exploration toward jumping.

## Experiment 30 — Color (RGB) observations instead of grayscale

**What changed:** `PreprocessFrame` no longer converts to grayscale. Observations are now RGB (3 channels per frame, 84x84x3). Frame stack produces 4*3=12 channels total. DQN first conv layer updated from `Conv2d(4, ...)` to `Conv2d(12, ...)`.

**Why:** Grayscale loses color information that could help the agent distinguish game elements. In grayscale, brown Goombas and green pipes both appear as mid-gray — the agent can't differentiate them visually. In RGB, Goombas (brown), pipes (green), coins (yellow), and ground (tan) are easily distinguishable. Better visual discrimination could help the agent learn that Goombas require jumping, coins are worth collecting, etc. This is a classic observation richness tradeoff: more channels but same spatial resolution.

**Result:** total_reward=435, score=0, max_x_dist=435. Worse than baseline. Color adds channels (12 vs 4) but reduces learning efficiency per step — less episodes fit in time budget. Discarded.

## Experiment 31 — Native game reward only (remove DistanceReward wrapper)

**What changed:** Removed the `DistanceReward` wrapper from the training pipeline. The agent now trains on the native `gym_super_mario_bros` reward signal only.

**Why:** The DistanceReward wrapper adds `(x_pos - prev_x)*2.0 + score_delta*0.3 - 0.1 + flag*1000`. These manual shaping terms might actually hurt because: (1) the -0.1/step penalty punishes careful play; (2) the 2.0 distance multiplier makes "run right blindly" dominate over "jump strategically". The native reward from the Mario environment already encodes distance (1 reward per x-pos unit) and score, and was designed specifically for this game. Using native reward is simpler and more directly aligned with what evaluate.py actually measures. Simplification win if it works.

**Result:** total_reward=313, score=0, max_x_dist=313. Without distance shaping the agent barely moves past starting area. Native reward alone insufficient. Discarded.

## Experiment 32 — k=2 frame stack (reduced from k=4)

**What changed:** `FrameStack(env, k=2)` instead of k=4. DQN first conv layer updated from `Conv2d(4, 32, ...)` to `Conv2d(2, 32, ...)`.

**Why:** Halving input channels means faster forward/backward passes per step → more Q-value updates fit in the 5-minute budget → potentially better-converged policy. Mario's key physics (velocity, jump arc) are largely visible in 2 consecutive frames. The temporal context of 4 frames may be overkill for a 2D platformer where most decisions depend only on current and previous frame. More updates per minute could overcome the exploration-exploitation gap.

**Result:** crash. evaluate.py hardcodes FrameStack(env, k=4) so model trained with k=2 (2 input channels) gets 4 channels at eval time. Cannot change k without breaking evaluate.py. Discarded.

## Experiment 33 — Faster epsilon decay (EPS_DECAY=1500, EPS_END=0.02)

**What changed:** EPS_DECAY 3000→1500, EPS_END 0.05→0.02. At step 6000: eps ≈ 0.02 + 0.98*exp(-4) ≈ 0.038. Agent is ~96% greedy in the second half of training.

**Why:** The main hypothesis for our exploration-exploitation gap: the greedy eval policy (eps=0) behaves very differently from training policy (eps=0.05-0.18). If we make the training policy near-greedy sooner, the Q-values are updated on states that more closely reflect what the greedy policy encounters. Less training data from "random flailing", more from "what a competent greedy policy actually does". This should close the gap between training reward and eval reward.

**Result:** total_reward=315, score=0, max_x_dist=315. Faster greedy convergence starves exploration — agent never discovers jumping. Discarded.

## Experiment 34 — Higher learning rate (LR=3e-4)

**What changed:** LR 1e-4 → 3e-4.

**Why:** In a fixed 5-minute window, a higher learning rate might converge Q-values faster than the conservative 1e-4. Adam handles larger LRs well due to gradient normalization. Gradient clipping (max norm 10.0) prevents instability. This is a "train more efficiently in less time" bet.

**Result:** total_reward=593, score=200 (1 kill!), max_x_dist=393. Agent learned to jump but sacrificed distance. LR too high — converges to kill-chasing instead of balanced right-movement. Discarded.

## Experiment 35 — LR=2e-4 (compromise between 1e-4 and 3e-4)

**What changed:** LR 1e-4 → 2e-4.

**Why:** Exp34 (LR=3e-4) showed the agent consistently jumped (score=200) but sacrificed distance (393). Exp default (LR=1e-4) gets distance=700 but score=0 usually. LR=2e-4 is a midpoint — hypothesis: agent learns to jump (from faster LR) while still advancing far (from slower-converging distance Q-values). If we can get dist=500 + score=400+, we beat the 700 consistent baseline.

**Result:** total_reward=314, score=0, max_x_dist=314. Worse than both 1e-4 and 3e-4. LR=2e-4 is not a sweet spot. Discarded.

## Experiment 36 — LR=3e-4 + distance multiplier 3.0

**What changed:** LR 1e-4 → 3e-4. Distance reward multiplier 2.0 → 3.0.

**Why:** Exp34 (LR=3e-4) showed the agent reliably learned to jump (score=200) but sacrificed distance (x=393). Hypothesis: with dist_mult=2.0 and score_bonus=0.3, the kill reward (60) equals running 30px (60), making killing too attractive at LR=3e-4. At dist_mult=3.0, running 30px = 90 reward vs kill = 60 — distance is 50% more valuable. This should push the LR=3e-4 agent to advance further right while still using the higher LR's learned jump behavior.

**Result:** total_reward=535, score=100, max_x_dist=435. Higher LR consistently gets some kills but at the cost of distance. The kill-chasing pattern dominates over going right. Discarded.

## Experiment 37 — Reward clipping to [-1, 1]

**What changed:** Clip reward to [-1, 1] per step in DistanceReward (after all shaping terms).

**Why:** The original Atari DQN paper (Mnih et al. 2015) clips rewards to [-1, 1]. This prevents large rewards from dominating Q-value estimates and helps training stability. In our case, distance rewards can be 0-40/step and kill bonuses 60/step — these large values may cause Q-value overestimation and unstable gradients. With clipping, all rewards are bounded, making Q-values converge more reliably. The information content is preserved (positive=good, negative=bad, larger=better) but the scale is normalized.

**Result:** total_reward=653, score=100, max_x_dist=553. Agent learned to jump (score=100) AND advance (553px). Reward clipping balances signals. But 653 < 700 (baseline without clipping). Will try with higher LR. Discarded.

## Experiment 38 — Reward clipping to [-5, 5]

**What changed:** Clip reward to [-5, 5] per step after all shaping terms.

**Why:** Exp37 tried [-1, 1] clipping (got 653). The problem: [-1, 1] collapsed the reward hierarchy — distance steps (2-4) and kill bonuses (60) both clipped to 1, destroying the differentiation. With [-5, 5]: typical movement steps (0-4) are unclipped or slightly clipped, kill bonus (60 → 5) is still 1.5-5x a regular step (meaningfully larger), and flag bonus (1000 → 5) is preserved as a strong signal. This preserves relative ordering (kills > movement > idle) while preventing extreme values from dominating gradients.

**Result:** total_reward=296, score=0, max_x_dist=296. Worse than [-1,1] clipping. The [-5,5] clip still distorts the reward signal enough to hurt learning. Reward clipping direction abandoned. Discarded.

## Experiment 39 — Noisy Networks (replace epsilon-greedy with learned noise)

**What changed:** Added `NoisyLinear` layer class (factorized Gaussian noise, Fortunato et al. 2017). Replaced all `nn.Linear` layers in the value and advantage streams with `NoisyLinear`. Removed epsilon-greedy — action selection is always greedy (noise in weights provides exploration). `reset_noise()` called before each action selection and each training update.

**Why:** Epsilon-greedy's biggest flaw: it explores randomly, uniformly across all actions, regardless of state. In Mario, random exploration wastes most of its budget on suboptimal actions. Noisy Networks provide *state-dependent* exploration: the network is uncertain in states it hasn't learned well, and confident in states it knows. This means in novel situations (Goomba in front), the network explores more; in familiar situations (open runway), it's consistent. At eval time, `model.eval()` sets `training=False`, disabling noise entirely — the policy is fully deterministic. This directly eliminates the training/eval gap: both training and eval use the same deterministic greedy policy, plus controlled noise during training. See [Fortunato et al. 2017](https://arxiv.org/abs/1706.10295).

**Result:** total_reward=434, score=0, max_x_dist=434. Noisy nets did not outperform epsilon-greedy in the 5-minute window. The noise may not be well-calibrated for this game with default sigma=0.5. Higher VRAM (241MB vs 147MB). Discarded.

## Experiment 40 — Slower epsilon decay (EPS_DECAY=5000)

**What changed:** EPS_DECAY 3000 → 5000. At step 6000 (typical end of training): eps ≈ 0.05 + 0.95*exp(-1.2) ≈ 0.35.

**Why:** With EPS_DECAY=3000, the agent becomes mostly greedy (~82%) by end of training. We observed (Exp25) that consistent runs get dist=700 but score=0 — the greedy policy doesn't jump. More sustained exploration means the agent discovers jumping more frequently throughout training, potentially collecting enough "jump is good" experiences to update Q-values in favor of jumping even in the greedy policy. Contrast with EPS_DECAY=1500 (Exp33, failed) which collapsed exploration too fast.

**Result:** total_reward=312, score=0, max_x_dist=312. More exploration = less convergence = worse policy. Agent stays too random throughout. Discarded.

## Experiment 41 — 64x64 observation resolution (reduced from 84x84)

**What changed:** PreprocessFrame default shape changed from (84, 84) to (64, 64). DQN linear layers updated from Linear(3136, ...) to Linear(1024, ...) to match new conv output size (64x64 through same conv stack → 4×4×64=1024).

**Why:** Smaller observations = smaller state tensors = faster forward/backward passes = more Q-value updates per 5-minute budget. With 84x84: 4*84*84=28224 values per state. With 64x64: 4*64*64=16384 values per state — 42% smaller. The extra training steps may compensate for the slight loss of visual detail. The DQN was originally designed for Atari's 84x84 standard, but Mario's game elements (pipes, Goombas, gaps) are large enough to be clearly visible at 64x64. evaluate.py imports PreprocessFrame from train.py so this applies to both training and evaluation consistently.

**Result:** total_reward=695, score=100 (1 kill), max_x_dist=595. VRAM dropped to 74MB. Agent got a kill AND good distance, but 695 < 700 baseline. Will try with higher LR to see if that combination improves further. Discarded.

## Experiment 42 — 64x64 observations + LR=3e-4

**What changed:** 64x64 observations (from 84x84), Linear(1024, ...) to match, LR=3e-4.

**Why:** Combining insights from Exp41 (64x64 gets kills, uses less VRAM/time) and Exp34 (LR=3e-4 reliably learns to jump). Hypothesis: faster inference (64x64) means more training steps, and higher LR means Q-values learn jump behavior faster. Together: agent both advances AND kills, potentially beating 1600. Each alone was suboptimal but the combination might be synergistic.

**Result:** total_reward=308, score=0, max_x_dist=308. Combination failed - less than either alone. LR=3e-4 and 64x64 are not synergistic. Discarded.

## Experiment 43 — BATCH_SIZE=256

**What changed:** BATCH_SIZE 128 → 256.

**Why:** Larger batches reduce gradient variance — each update is computed from 256 diverse transitions instead of 128, giving more stable Q-value estimates. In a 5-minute window with limited data, reducing variance might matter more than exploring the loss landscape widely. More stable updates could lead to a more reliable greedy policy at eval time.

**Result:** total_reward=594, score=0, max_x_dist=594. Larger batch no help - fewer updates per episode mean agent explores less. Discarded.

## Experiment 44 — RMSprop optimizer (original DQN paper setup)

**What changed:** Optimizer changed from Adam(lr=1e-4) to RMSprop(lr=2.5e-4, alpha=0.95, eps=0.01).

**Why:** The original Mnih et al. 2015 DQN paper used RMSprop with these exact hyperparameters. While Adam has become popular for DRL, RMSprop was specifically chosen for the original Atari experiments and may be better suited for this problem. Adam's adaptive moment estimates can sometimes cause issues in non-stationary RL settings. RMSprop with alpha=0.95 (momentum for squared gradients) is more conservative and may give more stable Q-value updates.

**Result:** total_reward=734, score=300 (3 kills!), max_x_dist=434. KEPT. RMSprop reliably learned to jump on enemies — scored 300 points vs 0 with Adam. Total 734 beats the consistent Adam baseline of ~706. RMSprop momentum on squared gradients may stabilize Q-value updates for the kill-discovery problem.

## Experiment 45 — RMSprop + score_bonus=0.5

**What changed:** score_bonus 0.3 → 0.5. Building on Exp44 (RMSprop, 734, score=300).

**Why:** Exp44 showed RMSprop consistently learns to jump on enemies (score=300, 3 kills). With score_bonus=0.3, each kill reward = 200*0.3=60. Distance reward for running 30px = 60. So kills and distance are equally valued. Increasing to 0.5: kill reward = 100, distance reward for 30px = 60. Kill is now ~1.67x more valuable. Hypothesis: RMSprop will find it even more worthwhile to seek out enemies and jump, while still advancing. With Adam+0.5 we got 534 — but RMSprop may handle this differently.

**Result:** total_reward=534, score=100 (1 kill), max_x_dist=434. Higher bonus actually REDUCED kills (100 vs 300 with 0.3). 0.3 is the RMSprop sweet spot too. Discarded.

## Experiment 46 — RMSprop + 64x64 observations

**What changed:** PreprocessFrame shape (84,84)→(64,64), Linear(3136,...)→Linear(1024,...), on top of RMSprop baseline (Exp44).

**Why:** Adam+64x64 (Exp41) got score=100, dist=595, total=695. RMSprop+84x84 (Exp44) got score=300, dist=434, total=734. Hypothesis: RMSprop's superior kill-learning ability, combined with 64x64's faster inference (more training steps per 5 mins), could get score=300+ AND dist=500+, pushing total well above 800.

**Result:** total_reward=314, score=0, max_x_dist=314. 64x64 destroys RMSprop kill-learning — back to Adam-level performance. Smaller resolution hurts RMSprop more than Adam. Discarded.

## Experiment 47 — RMSprop + EPS_DECAY=4000

**What changed:** EPS_DECAY 3000 → 4000, on top of RMSprop baseline (Exp44, score=300, dist=434).

**Why:** RMSprop's agent reliably kills 3 Goombas (up to x=434) but dies at the first big gap. With EPS_DECAY=3000, by step ~4000 the agent is mostly greedy. More exploration time (4000 decay) might help discover the jump pattern needed to clear the gap at x=434-500. Moderate increase — not so much exploration that Q-values don't converge, but enough to randomly discover the gap-clearing jump and reinforce it.

**Result:** total_reward=697, score=100, max_x_dist=597. Better distance (597 vs 434) but fewer kills (100 vs 300). More exploration trades kill-learning for distance exploration. 697 < 734 baseline. EPS_DECAY=3000 is RMSprop sweet spot. Discarded.

## Experiment 48 — RMSprop + EPS_END=0.02

**What changed:** EPS_END 0.05 → 0.02, on top of RMSprop baseline (Exp44).

**Why:** The agent with RMSprop+EPS_DECAY=3000 learns to kill (score=300) but dies at x=434. With EPS_END=0.05, the agent is 5% random at evaluation. With 0.02, it's 98% greedy — the learned kill policy dominates. If the Q-values for "kill enemies and advance" are already good, lower EPS_END means the greedy policy executes them more consistently without random noise disrupting the sequence.

**Result:** total_reward=534, score=100 (1 kill vs 300 before). Lower EPS_END hurts — the 5% random exploration in EPS_END=0.05 was helping discover kills. Discarded.

## Experiment 49 — RMSprop + MEMORY_SIZE=50000

**What changed:** MEMORY_SIZE 20000 → 50000, on top of RMSprop baseline (Exp44).

**Why:** With 20000 buffer and ~6000 training steps, all experiences fit in buffer (never overwritten). But with 50000 — the buffer is so large that diverse early experiences (including lucky kills and gap crossings from high-exploration phase) stay available throughout all training. More diverse sampling = better Q-value estimates across all states the agent might encounter at eval time.

**Result:** total_reward=934, score=500 (5 kills!), max_x_dist=434. NEW RECORD. KEPT. Larger buffer dramatically improved kill behavior — 5 kills vs 3 kills at MEMORY_SIZE=20000. More diverse kill experiences in the replay buffer means stronger Q-value signal for jumping on enemies.

## Experiment 50 — MEMORY_SIZE=100000

**What changed:** MEMORY_SIZE 50000 → 100000, on top of Exp49 baseline (RMSprop, 934).

**Why:** Exp49 showed larger buffer dramatically improves kill behavior (3→5 kills). Hypothesis: 100000 buffer will retain even more diverse experiences across more episode types, potentially helping the agent also learn to cross the gap at x=434. The buffer is still primarily filled by the ~6000 training steps, so the bottleneck is experience diversity not capacity.

**Result:** total_reward=694, score=100, max_x_dist=594. 100k buffer hurts — too many old diverse experiences dilute the kill signal. 50k is the sweet spot. dist increased to 594 but kills dropped to 1. Discarded.

## Experiment 51 — score_bonus=0.4 (on RMSprop+50k baseline)

**What changed:** score_bonus 0.3 → 0.4, on top of Exp49 (RMSprop+MEMORY_SIZE=50000, 934).

**Why:** Exp49 got 5 kills (score=500) at dist=434. With score_bonus=0.4, each kill is worth 200*0.4=80 reward vs running 30px=60 reward. Kills are now 33% more valuable relative to movement. Hypothesis: agent kills 6-8 enemies instead of 5, pushing score to 600-800, total to 1034-1234. Previous Adam+0.4 experiment (Exp21) got total_reward=994 (dist=594,score=400) — significantly better with RMSprop's superior kill-learning.

**Result:** total_reward=314, score=0. Higher score bonus completely disables kill-learning with RMSprop+50k. 0.3 is uniquely optimal. Discarded.

## Experiment 52 — dist_mult=3.0 on RMSprop+50k baseline

**What changed:** Distance reward multiplier 2.0 → 3.0, on top of Exp49 (RMSprop+50k, 934).

**Why:** The agent kills 5 enemies but stalls at x=434. With dist_mult=2.0: run 30px=60 reward, kill=60 reward (equal). With dist_mult=3.0: run 30px=90 reward, kill=60 (distance is 50% more valuable). Hypothesis: stronger right-movement signal pushes the agent past the x=434 obstacle while RMSprop+50k still enables some kills. If we get dist=600+score=300, total=900+.

**Result:** total_reward=1001, score=300 (3 kills), max_x_dist=701. NEW RECORD. KEPT. dist_mult=3.0 pushed the agent past x=434 to x=701, trading 2 kills (5→3) for 267 extra distance (434→701). Total 1001 > 934. Right-movement reward dominance enables clearing the early obstacle section.

## Experiment 53 — dist_mult=4.0

**What changed:** dist_mult 3.0 → 4.0, on top of Exp52 (1001, dist=701, score=300).

**Why:** At dist_mult=3.0: run 30px=90 reward, kill=60 (dist 50% more valuable). The agent reached x=701 with 3 kills. At dist_mult=4.0: run 30px=120, kill=60 (dist 2x more valuable). Stronger signal to advance right. Hypothesis: agent advances further (x=800-900+) trading some kills for distance. If we get dist=900+score=100, total=1000. If we get dist=1000+score=0, total=1000. Need dist=1000+score=200 to beat 1001 meaningfully.

**Result:** total_reward=436, score=0. Too much distance emphasis: agent tries to rush but dies at x=436 because it stopped learning to jump. dist_mult=3.0 is the sweet spot. Discarded.

## Experiment 54 — BATCH_SIZE=64

**What changed:** BATCH_SIZE 128 → 64, on top of Exp52 (dist_mult=3.0+RMSprop+50k, 1001).

**Why:** Smaller batch = more gradient updates per unit of experience. With 128 batch: 1 update/step starts when buffer has 128. With 64 batch: 1 update/step starts 2x earlier, so total updates ≈ 2x. More gradient steps in the 5-minute window means the agent can potentially learn the later level sections (past x=700) that require more precise Q-values.

**Result:** total_reward=299, score=0. Smaller batches cause too much gradient variance with RMSprop. Agent fails completely. BATCH_SIZE=128 is better. Discarded.

## Experiment 55 — TARGET_UPDATE=200

**What changed:** TARGET_UPDATE 500 → 200, on top of Exp52 (dist_mult=3.0+RMSprop+50k, 1001).

**Why:** More frequent target network updates (every 200 steps instead of 500) keep the target Q-values more current. This reduces the "staleness gap" between policy net and target net, which can cause overestimation bias. With the agent now reaching x=700, it encounters new game states — more up-to-date targets may help it learn these new states' Q-values faster.

**Result:** total_reward=789, score=100, max_x_dist=689. More frequent target updates hurt — instability from too many target shifts. TARGET_UPDATE=500 is the right cadence. Discarded.

## Experiment 56 — dist_mult=3.5

**What changed:** dist_mult 3.0 → 3.5.

**Why:** At 3.0: dist=701, score=300, total=1001. At 4.0: dist=436, score=0, total=436 (failed — too much). At 3.5: 30px=105 reward vs kill=60 (dist 75% more valuable). Hypothesis: agent gets dist=750-800 with score=100-200, total=850-1000. May not beat 1001 but provides data on the sensitivity of this parameter.

**Result:** total_reward=694, score=100, max_x_dist=594. Worse than 3.0. dist_mult=3.0 is confirmed optimal — 3.5 trades too many kills for not enough extra distance. Discarded.

## Experiment 57 — Centered RMSprop

**What changed:** `centered=True` in RMSprop optimizer.

**Why:** Centered RMSprop normalizes gradients by their variance (second moment minus mean squared gradient) rather than the raw second moment. This is more like gradient normalization, reducing the influence of large average gradients. In RL's non-stationary environment, gradient magnitudes change over training — centered normalization adapts better to these shifts. Some implementations of DQN specifically use centered RMSprop.

**Result:** total_reward=315, score=0. Centered RMSprop breaks the training completely. Standard RMSprop (uncenered) is better for this task. Discarded.

## Experiment 58 — Remove step penalty

**What changed:** Removed `reward -= 0.1` step penalty from DistanceReward.

**Why:** The -0.1/step penalty was meant to discourage idling, but with dist_mult=3.0 the distance reward (~15/step when moving) already dominates over any idling incentive. The step penalty adds small negative bias to all transitions, possibly making the agent overly risk-averse (avoiding longer exploration trajectories). Removing it simplifies the reward. If results are equal or better, it's a simplification win.

**Result:** total_reward=823, score=100, max_x_dist=723. Slightly better distance (723 vs 701) but fewer kills (100 vs 300). Step penalty helps maintain kill behavior. 823 < 1001. Discarded.

## Experiment 59 — MEMORY_SIZE=75000

**What changed:** MEMORY_SIZE 50000 → 75000.

**Why:** 50000 gave best result (1001), 100000 was worse (694). The buffer interacts with training time — long episodes fill ~20k-60k transitions per 5 minutes. 75000 is a refined midpoint. Might retain slightly more diverse early-exploration experiences without the dilution problem of 100000.

**Result:** total_reward=536, score=100, max_x_dist=436. Worse than 50k. 50000 is the confirmed optimal buffer size. Discarded.

## Experiment 60 — Asymmetric distance reward (backward penalty 2x)

**What changed:** Distance reward now asymmetric: `max(delta_x, 0) * 3.0 + min(delta_x, 0) * 6.0`. Moving left is penalized at 6.0 per pixel, moving right rewards at 3.0 per pixel.

**Why:** The agent at x=701 might be bouncing back from obstacles instead of committing to jumping over them. With symmetric reward (delta_x * 3.0), going back 10px = -30 reward, jumping right 10px = +30 reward — symmetric risk. With asymmetric: going back 10px = -60, jumping right 10px = +30. The agent has 2x more to lose by retreating, making it more likely to commit to forward jumps even at risk of death. This could help navigate the obstacle at x=701.

**Result:** total_reward=923, score=200 (2 kills), max_x_dist=723. Slightly more distance (723 vs 701) but fewer kills (200 vs 300). Total 923 < 1001. Asymmetric penalty not worth the complexity. Discarded.

## Experiment 61 — EPS_DECAY=2500

**What changed:** EPS_DECAY 3000 → 2500. At step 6000: eps ≈ 0.05 + 0.95*exp(-2.4) ≈ 0.133 (87% greedy). Vs 3000: 82% greedy.

**Why:** Slightly faster convergence to exploit the learned kill+advance policy. If the agent has learned sufficient Q-values for jumping and advancing by step 4000 (vs 5000 with 3000 decay), it spends more of training time exploiting this knowledge, potentially reinforcing it more strongly.

**Result:** total_reward=694, score=100, max_x_dist=594. Faster decay starves exploration. EPS_DECAY=3000 confirmed optimal for this setup. Discarded.

## Experiment 62 — New territory bonus (exploration reward for x_max)

**What changed:** Added `best_x` tracking to DistanceReward. When the agent reaches a new maximum x position within an episode, it gets a bonus of `(x_pos - best_x) * 2.0` on top of the normal distance reward. This resets each episode.

**Why:** The agent is stuck at x=701 — it knows how to get there but not how to go further. This exploration bonus creates an extra incentive specifically for breaching new territory. First time reaching x=702 in an episode: reward += (702-701)*2=2 extra. First time reaching x=800: extra (800-701)*2=198 bonus. This should specifically motivate the agent to push past previously-seen positions, potentially learning to navigate x>701.

**Result:** total_reward=311, score=0. The first-visit exploration bonus creates massive Q-value inflation in early episodes, destabilizing training. Cannot mix episodic exploration bonuses with global Q-values. Discarded.

## Experiment 63 — Vanilla DQN targeting (disable Double DQN)

**What changed:** Removed Double DQN. Target Q-values now computed as `target_net(next_states).max(1)[0]` — target network selects AND evaluates the best next action.

**Why:** Double DQN reduces overestimation. But in RL, some optimism can be beneficial — it acts like an implicit exploration bonus (optimistic initialization). With standard DQN targeting, the Q-values may be slightly overestimated, making the agent more "hopeful" about unexplored states and more willing to try new things at x>701. Worth testing whether removing this correction helps with this specific setup.

**Result:** total_reward=314, score=0. Vanilla DQN Q-value overestimation is catastrophic — completely fails. Double DQN is essential for this setup. Discarded.

## Experiment 64 — Gradient clipping = 20.0

**What changed:** Gradient clip norm 10.0 → 20.0.

**Why:** RMSprop already normalizes gradient magnitudes by the second moment. The additional 10.0 gradient norm clip might be unnecessarily cutting off useful large gradient signals, particularly in novel states (x>700). Increasing to 20.0 allows larger gradient steps when the network encounters new states, potentially learning the later level section faster.

**Result:** total_reward=922, score=200, max_x_dist=722. Slightly different distribution (more distance, fewer kills) but 922 < 1001. Gradient clip at 10.0 is optimal. Discarded.

## Experiment 65 — RMSprop lr=5e-4

**What changed:** RMSprop lr 2.5e-4 → 5e-4.

**Why:** Higher LR might allow faster learning of new territory at x>701. The 1001 baseline with lr=2.5e-4 gets stuck at x=701 — the Q-values may not converge fast enough in the 5-minute window to learn the precise jump needed to clear the gap. Doubling the LR gives 2x more gradient movement per update, potentially bridging this gap.

**Result:** total_reward=435, score=0. Higher LR destabilizes RMSprop training. lr=2.5e-4 is the optimal LR for RMSprop in this setup. Discarded.

## Experiment 66 — Standard DQN (remove Dueling)

**What changed:** Replaced Dueling DQN (separate value V(s) and advantage A(s,a) streams) with a standard single-stream DQN: `conv → Linear(3136,512) → ReLU → Linear(512,n_actions)`.

**Why:** Dueling DQN excels when the value of a state is relatively action-independent — this helps in many Atari games. But in Mario, different actions at the same state have very different Q-values (jumping vs standing when a Goomba is near), so the dueling decomposition may not be as beneficial. A simpler single-stream network has fewer parameters to optimize in the same 5-minute window, potentially leading to faster convergence.

**Result:** total_reward=1527, score=400 (4 kills), max_x_dist=1127. NEW RECORD. KEPT. Removing Dueling DQN was the key breakthrough — standard single-stream DQN reaches x=1127 vs x=701 with Dueling. Less parameter overhead means faster convergence to good policies in the 5-minute window.

## Experiment 67 — score_bonus=0.2 (standard DQN baseline)

**What changed:** score_bonus 0.3 → 0.2, on top of Exp66 (standard DQN, 1527).

**Why:** With standard DQN, the agent now reaches x=1127. Reducing score bonus makes distance more dominant — agent should advance even further. If we can get dist=1300+ with any score ≥ 0, total could beat 1527. At score_bonus=0.2: kill reward = 200*0.2=40 vs running 30px = 90 reward (dist 2.25x more valuable).

**Result:** total_reward=434, score=0. Back to old barrier. score_bonus=0.3 is the optimal for standard DQN too. Discarded.

## Experiment 68 — dist_mult=2.5 (standard DQN baseline)

**What changed:** dist_mult 3.0 → 2.5, on top of Exp66 (standard DQN, 1527).

**Why:** At dist_mult=3.0: kill=60, run 30px=90 (dist 50% more valuable). Agent gets 4 kills and dist=1127. At dist_mult=2.5: kill=60, run 30px=75 (dist 25% more valuable). Agent should maintain more of the kill-learning behavior while still advancing. If we get 5-6 kills (score=500-600) at dist=900-1000, total could beat 1527.

**Result:** total_reward=314, score=0. dist_mult=3.0 is uniquely optimal — 2.5 collapses completely. The optimal is remarkably narrow. Discarded.

## Experiment 69 — score_bonus=0.35

**What changed:** score_bonus 0.3 → 0.35.

**Why:** Standard DQN at 0.3 gets 4 kills (score=400) at dist=1127. A tiny increase in score bonus might get 5 kills (score=500) while maintaining dist=1000+, beating 1527. Previous experiments with Dueling DQN showed 0.3 was optimal, but standard DQN behaves differently.

**Result:** total_reward=734, score=300, max_x_dist=434. Even 0.35 causes distance regression (434 vs 1127). score_bonus=0.3 is uniquely optimal across all DQN variants. Discarded.

## Experiment 70 — EPS_DECAY=3500 (standard DQN)

**What changed:** EPS_DECAY 3000 → 3500.

**Why:** Standard DQN now reaches x=1127. The obstacles at x=1127-3000 include more complex elements (multiple Goombas, Koopas, gaps). Slightly more exploration might help discover how to navigate these new sections. With EPS_DECAY=4000 and Dueling DQN, we got 697 (worse than 734). But standard DQN behaves very differently and might benefit from more exploration time.

**Result:** total_reward=315, score=0. Standard DQN also needs EPS_DECAY=3000. More exploration at any decay hurts. Discarded.

## Experiment 71 — Wider FC layer (1024)

**What changed:** FC hidden layer 512 → 1024.

**Why:** The agent now navigates to x=1127, encountering diverse game states in a section not seen with previous configs. More representational capacity (1024 vs 512 hidden neurons) might help encode the richer variety of states at x=700-1200 (different enemies, gaps, platforms). Previous attempts with larger networks failed when using Dueling DQN, but without dueling the architecture is simpler and might benefit more directly from wider layers.

**Result:** total_reward=434, score=0. Wider FC layer makes Q-value convergence slower in 5 minutes. 512 neurons is the right size. Discarded.

## Experiment 72 — GAMMA=0.995

**What changed:** GAMMA 0.99 → 0.995.

**Why:** World 1-1 takes 300+ frames to complete. With GAMMA=0.99, rewards 200 steps away are worth only 13.5% of immediate rewards. With GAMMA=0.995, they're worth 36.8% — 2.7x more valuable. Less discounting means the agent more strongly values being further right in the level (closer to the flag), potentially providing the motivation to push past x=1127.

**Result:** total_reward=822, score=100, max_x_dist=722. Higher gamma causes instability in Q-value estimation. GAMMA=0.99 is optimal. Discarded.

## Experiment 73 — Adam optimizer with standard DQN

**What changed:** Optimizer changed from RMSprop(lr=2.5e-4) back to Adam(lr=1e-4).

**Why:** All RMSprop experiments used Dueling DQN when Adam was established as baseline. Now with standard DQN (the breakthrough), Adam might actually perform better — the dueling architecture was the problem, not the optimizer. Standard DQN + Adam might be cleaner than standard DQN + RMSprop.

**Result:** total_reward=678, score=0, max_x_dist=678. Adam with standard DQN gets no kills and less distance. RMSprop is still essential. Discarded.

## Experiment 74 — 4-layer conv network

**What changed:** Added 4th conv layer Conv2d(64, 64, 3, 1). Feature map: 84→20→9→7→5, so 5x5x64=1600. Updated FC to Linear(1600, 512).

**Why:** Extra conv layer = more spatial abstraction = better positional invariance. Standard DQN reaches x=1127 with 3 conv layers. The 4th layer further abstracts position-specific patterns, potentially helping the agent generalize game element recognition across different positions in the level.

**Result:** total_reward=515, score=200, max_x_dist=315. 4th conv layer too slow to converge in 5 minutes. 3-layer conv confirmed optimal. Discarded.

## Experiment 75 — Global best-x exploration bonus

**What changed:** Added `global_best_x` to DistanceReward that persists across episodes. First time the agent reaches any new x position EVER during training, it gets `(x_pos - global_best_x) * 1.0` bonus. This resets to 0 only when the wrapper is created (start of training).

**Why:** Sparse global exploration reward creates a monotonic motivation gradient toward new territory. First run forward: agent gets bonus for every pixel beyond the historical max (like conquering new land). This never fires twice for the same territory. Creates curriculum-like behavior: early episodes get rewarded for x=0-300, later episodes must beat that to get any bonus. The agent is specifically incentivized to push beyond its previous best, helping it navigate past x=1127. Unlike per-episode exploration bonuses (Exp62, failed), the global version is consistent across the Q-value learning.

**Result:** total_reward=435, score=0. Global exploration bonus creates Q-value inflation. Discarded.

## Experiment 76 — Centered frame normalization ([-1, 1])

**What changed:** Frame normalization changed from `/ 255.0` ([0,1] range) to `/ 128.0 - 1.0` ([-1,1] range).

**Why:** Zero-centered inputs are standard best practice in deep learning — gradient flow is improved when activations are centered around 0. The conv layers receive inputs in [-1,1] rather than [0,1], which means the initial weights are better suited (Kaiming init assumes symmetric inputs). This is a minor normalization change that could improve convergence stability.

**Result:** total_reward=296, score=0. Centering inputs around [-1,1] instead of [0,1] destroys performance. /255.0 normalization is optimal. Discarded.

## Experiment 77 — Alive bonus +0.01 (replace step penalty)

**What changed:** Step penalty `reward -= 0.1` replaced with `reward += 0.01`.

**Why:** The -0.1 step penalty might make the agent risk-averse — dying quickly (0 steps) is only slightly worse than surviving. With +0.01 alive bonus: surviving 200 steps = +2.0 extra reward, while dying at step 50 = +0.5 extra. Combined with dist_mult=3.0, the agent still primarily optimizes distance but now also values staying alive. This might help navigate the challenging section at x>1127 by making survival more important.

**Result:** total_reward=305, score=0. Alive bonus disables the urgency that drives forward progress. Step penalty -0.1 is essential to the optimal setup. Discarded.

## Experiment 78 — MEMORY_SIZE=40000

**What changed:** MEMORY_SIZE 50000 → 40000.

**Why:** 50000 is confirmed optimal vs 20000 and 75000. But 40000 might also work — it would be 20% smaller, potentially reducing the weight of old low-quality random-phase experiences in the replay distribution while still retaining enough diversity. This is a fine-tuning experiment around the confirmed sweet spot.

**Result:** total_reward=314, score=0. 40000 worse than 50000. Buffer sweet spot is exactly 50000. Discarded.

## Experiment 79 — Xavier weight initialization

**What changed:** Added `_init_weights()` method to DQN that applies `xavier_uniform_` to all Conv2d and Linear layers, with zero bias init. Called after policy_net is created.

**Why:** PyTorch's default weight initialization uses Kaiming uniform (He initialization) for conv layers, which is designed for ReLU activations. Xavier uniform is designed for symmetric activations and may give better initial gradient flow through the Q-network. With the current small 5-minute training window, faster convergence in the first few episodes could meaningfully boost the final reward. Xavier init ensures the variance of activations is preserved across layers, potentially leading to more stable early learning.

**Result:** total_reward=299, score=0. Xavier init hurts performance. PyTorch default Kaiming init (designed for ReLU) is better. Discarded.

## Experiment 80 — TARGET_UPDATE=1000 (less frequent target sync)

**What changed:** TARGET_UPDATE 500 → 1000.

**Why:** The target network in DQN provides stable Q-value targets during gradient updates. Updating it less frequently (every 1000 steps instead of 500) gives the policy longer to converge toward each set of targets before they shift. In the original DQN paper, target updates were done every 10,000 steps. With a 5-minute budget and ~15k steps, 1000 means ~15 target updates vs ~30 at 500 — each target is more stable, potentially reducing oscillations and improving convergence in the critical early training phase.

**Result:** total_reward=922, score=200. Less frequent target sync hurts — 500 steps is optimal for this 5-minute window. Discarded.

## Experiment 81 — RMSprop weight_decay=1e-5 (L2 regularization)

**What changed:** Added `weight_decay=1e-5` to RMSprop optimizer.

**Why:** L2 regularization penalizes large weights, preventing overfitting to specific experiences in the replay buffer. With only 5 minutes of training, the network may overfit to certain patterns it sees early (especially after the random exploration phase). A small weight decay nudges weights toward smaller values, encouraging the network to find more generalizable Q-value representations. 1e-5 is a mild regularization — enough to improve generalization without hindering learning speed.

**Result:** total_reward=311, score=0. Weight decay hurts — the network needs full capacity in 5-minute budget. Discarded.

## Experiment 82 — Death penalty -50 on non-flag termination

**What changed:** Added `reward -= 50.0` when `terminated=True` and `flag_get=False`.

**Why:** Currently the agent has no explicit penalty for dying beyond missing out on future distance rewards. Adding a strong death penalty (-50) creates a clear negative signal for dying. This might make the agent more careful — e.g., it might learn to jump over enemies rather than into them, or be more cautious at gaps. The agent that reaches x=1127 with 4 kills might be dying on a 5th enemy encounter; a death penalty could teach it to avoid that situation.

**Result:** total_reward=314, score=0. Death penalty makes agent too cautious — avoids progress. -0.1 step penalty alone is better for forward drive. Discarded.

## Experiment 83 — BATCH_SIZE=256

**What changed:** BATCH_SIZE 128 → 256.

**Why:** Larger batches reduce gradient variance — each update uses more diverse experiences from the replay buffer, giving the network a better signal. With MEMORY_SIZE=50000 and 5 minutes of training, the buffer fills up quickly and a larger batch might sample more representative Q-value targets. This is a standard technique for stabilizing DQN training.

**Result:** total_reward=922, score=200. Larger batch slows learning — fewer total updates per minute, hurting convergence in the time-limited window. 128 is optimal. Discarded.

## Experiment 84 — RMSprop momentum=0.9

**What changed:** Added `momentum=0.9` to RMSprop optimizer.

**Why:** RMSprop with momentum combines the adaptive learning rate of RMSprop with the velocity accumulation of SGD with momentum. This can speed convergence by allowing the optimizer to "coast" through flat gradient regions and accelerate on consistent gradients. In the original DQN paper, RMSprop was used without momentum. However, adding momentum may help the optimizer escape local plateaus and converge faster in the 5-minute training window. momentum=0.9 is a standard value.

**Result:** total_reward=1352, score=500 (5 kills!), dist=852. Momentum helps kill-learning but reduces distance coverage. RMSprop without momentum better for forward progress. Discarded.

## Experiment 85 — EPS_DECAY=4000 (slower exploration decay)

**What changed:** EPS_DECAY 3000 → 4000.

**Why:** With EPS_DECAY=3000, epsilon drops from 1.0 to near EPS_END (~0.05) by step ~12,000. With 4000, the agent stays more exploratory until ~step 16,000. The agent consistently fails at x=1127 — possibly because it hasn't explored that region enough during training. Slower decay means more random exploration throughout the run, potentially discovering states and transitions past that barrier. The cost is slightly less exploitation time, but if the agent discovers better strategies in the barrier region, the net effect could be positive.

**Result:** total_reward=158, score=0. Slower decay means too much random exploration — not enough exploitation. EPS_DECAY=3000 confirmed optimal. Discarded.

## Experiment 86 — GAMMA=0.995 (higher discount factor)

**What changed:** GAMMA 0.99 → 0.995.

**Why:** Higher gamma means the agent values future rewards more. At GAMMA=0.99, a reward 100 steps away is worth `0.99^100 ≈ 0.37` of its face value. At GAMMA=0.995, the same reward is worth `0.995^100 ≈ 0.61`. This longer time horizon might help the agent learn to navigate obstacles that require multi-step planning. For a level that takes 300-400 game time units, the flag reward might be too heavily discounted at 0.99. 0.995 could help the agent "see" the flag reward from further away.

**Result:** total_reward=735, score=300. Higher GAMMA causes training instability — longer horizon makes targets harder to estimate. 0.99 is optimal. Discarded.

## Experiment 87 — Prioritized Experience Replay (PER)

**What changed:** Replaced uniform `ReplayBuffer` with `PrioritizedReplayBuffer`. Experiences are sampled proportional to their TD error^alpha (alpha=0.6). Importance sampling weights (IS weights) correct for the biased distribution (beta annealed 0.4→1.0 over 50k steps). TD errors are updated after each gradient step.

**Why:** The agent consistently fails around x=1127 — a specific obstacle region. With uniform replay, every transition is sampled equally. PER prioritizes high-error transitions (like the ones near the x=1127 barrier) for more frequent training. This was introduced in [Schaul et al. 2016](https://arxiv.org/abs/1511.05952) and shown to improve DQN by ~2x on Atari. For our case, replaying the difficult transitions more often should help the agent learn to navigate past the current bottleneck.

**Result:** total_reward=1012, score=300, dist=712. PER is promising (score matches 4 kills at lower distance) but doesn't beat baseline. Overhead of priority computation may reduce total updates in 5 minutes. Discarded.

## Experiment 88 — N-step returns (n=3)

**What changed:** Replaced 1-step TD with 3-step returns. A deque of size 3 accumulates recent transitions. When full (or at episode end), the earliest transition is stored in the replay buffer with the 3-step discounted return: R = r₀ + γr₁ + γ²r₂. The bootstrap target uses GAMMA^3 for the next state value.

**Why:** 1-step TD has high bias — reward signals propagate slowly backward through the Q-value estimates. N-step returns use actual rewards for n steps before bootstrapping, reducing this bias. This is one of the components of Rainbow DQN and has been shown to significantly improve Atari performance. For Mario, where rewards are spread across many steps (killing an enemy while moving forward), 3-step returns should propagate the reward signal faster and help the agent learn more efficient sequences of actions.

**Result:** total_reward=534, score=100. N-step returns reduce sample efficiency here — storing only every-3rd transition means less total replay. 1-step TD is better in 5-minute budget. Discarded.

## Experiment 89 — LR=2e-4

**What changed:** RMSprop learning rate 2.5e-4 → 2e-4.

**Why:** 2.5e-4 was found superior to 1e-4 (too slow) in earlier experiments. But we haven't tried values between 1e-4 and 2.5e-4. A slightly lower learning rate (2e-4) could improve stability — less overshooting of Q-value estimates while still learning fast enough in 5 minutes.

**Result:** total_reward=704, score=0. Slower learning rate hurts in the 5-minute budget. 2.5e-4 is optimal. Discarded.

## Experiment 90 — Remove 3rd conv layer (2 conv layers only)

**What changed:** Removed the 3rd conv layer (Conv(64,64,3,1)) from the DQN architecture. FC input changes from 3136 to 5184 (64×9×9). FC hidden layer stays at 512.

**Why:** Removing Dueling DQN (Exp66) was the biggest improvement — fewer parameters converge faster in 5 minutes. Following the same logic: maybe 3 conv layers is too many. 2 conv layers have sufficient capacity to extract spatial features at 84×84 (8×8 and 4×4 kernels cover the relevant patterns), while fewer params = faster optimization. The 3rd conv layer (3×3 stride 1) mainly adds local refinement that may not be needed here.

**Result:** total_reward=313, score=0. 3 conv layers are essential for spatial reasoning. Removing the 3rd conv layer drastically reduces performance. Discarded.

## Experiment 91 — Step penalty -0.05 (halved)

**What changed:** Step penalty reduced from -0.1 to -0.05 per step.

**Why:** -0.1 creates urgency to move right. But it also penalizes the agent for pausing to jump over enemies or navigate tricky sections. A softer -0.05 penalty might allow more careful navigation in challenging areas while still providing forward pressure. The agent gets 4 kills then dies — maybe it's rushing and needs to be more patient.

**Result:** total_reward=315, score=0. Softer penalty leads to less urgency and stalls progress. -0.1 confirmed optimal. Discarded.

## Experiment 92 — RMSprop alpha=0.9

**What changed:** RMSprop alpha 0.95 → 0.9.

**Why:** Alpha controls the decay of the running average of squared gradients. Lower alpha (0.9) means the optimizer gives more weight to recent gradient magnitudes vs older ones — faster adaptation to the current loss landscape. This can help when the reward structure changes during training (as the agent moves into new areas of the level). 0.95 was the original DQN paper value; 0.9 might better suit the dynamic reward environment here.

**Result:** total_reward=305, score=0. alpha=0.9 too aggressively forgets gradient history, leading to noisy updates. 0.95 confirmed optimal. Discarded.

## Experiment 93 — Gradient clip=1.0

**What changed:** Gradient clipping norm 10.0 → 1.0.

**Why:** Tighter gradient clipping prevents large gradient steps that might destabilize the Q-network. With clip=10.0, gradients can be large when the Q-values are far from targets (e.g., when the agent discovers a new area). Clip=1.0 forces smaller, more conservative updates. This might help the network converge to a more stable policy in the later training phase when the agent has already learned the early part of the level and needs finer-grained updates to improve.

**Result:** total_reward=314, score=0. Clip=1.0 too restrictive — slows learning. 10.0 confirmed optimal. Discarded.

## Experiment 94 — TARGET_UPDATE=250 (more frequent target sync)

**What changed:** TARGET_UPDATE 500 → 250.

**Why:** More frequent target updates keep the value estimates more current. With only ~15k training steps in 5 minutes, 250 means ~60 target syncs (vs ~30 at 500). This could help early in training when the policy changes rapidly and targets lag far behind. We know 500 > 1000, so maybe 250 > 500 too.

**Result:** total_reward=312, score=0. More frequent sync causes instability. 500 is the sweet spot. Discarded.

## Experiment 95 — Milestone bonuses at x checkpoints

**What changed:** Added one-time bonuses of +200 when the agent first crosses x=500, 800, 1000, 1200, 1500, 2000, 2500, 3000 within an episode. Uses `max_x` tracking to ensure each bonus fires only once per episode.

**Why:** The agent consistently dies around x=1127. With pure continuous shaping, it has no special incentive to push past that specific barrier — getting there is worth ~1127*3=3381 cumulative reward, but the "last mile" reward for crossing from 1127 to 1200 is only ~(1200-1127)*3=219. A discrete +200 bonus at x=1200 doubles the incentive to push past that obstacle. The milestones are designed to create intermediate goals that guide the agent through the whole level, not just reward local progress.

**Result:** total_reward=313, score=0. Large discrete bonuses disrupt the Q-value scale and destabilize training. Continuous distance shaping is better. Discarded.

## Experiment 96 — Soft Polyak target updates (τ=0.01)

**What changed:** Replaced hard target network copy (every 500 steps) with per-step Polyak averaging: `θ_target ← 0.01*θ_policy + 0.99*θ_target`. The `TARGET_UPDATE` constant is no longer used.

**Why:** Hard target updates cause periodic discontinuities in the Q-value landscape — every 500 steps the target jumps. Soft updates blend smoothly, giving the policy network stable but gradually improving targets at every step. This approach is used in DDPG, TD3, and SAC and is known to improve convergence stability significantly. With TAU=0.01, the target network effectively tracks an exponential moving average of the policy weights over ~100 steps.

**Result:** total_reward=299, score=0. Soft updates too slow to converge in 5 minutes — targets lag behind needed for stable early learning. Hard copy every 500 steps is better. Discarded.

## Experiment 97 — MSE loss instead of Huber (SmoothL1)

**What changed:** Loss function changed from `nn.SmoothL1Loss()` (Huber) to `nn.MSELoss()`.

**Why:** Huber loss (SmoothL1) is designed to be robust to outlier large TD errors — it transitions from L2 (quadratic) to L1 (linear) for errors > 1. But our shaped rewards are relatively bounded and consistent. MSE penalizes large errors more heavily, which could force the Q-network to fit high-value states more aggressively. In some DQN variants, MSE converges faster when the reward scale is well-controlled. This is a simple but fundamental change to the training signal.

**Result:** total_reward=267, score=0. MSE amplifies large TD errors and destabilizes training. Huber loss is essential for stable Q-learning. Discarded.

## Experiment 98 — LR decay: halve at midpoint (2.5e-4 → 1.25e-4)

**What changed:** Added a one-time LR step at the halfway point of training (t=150s): LR drops from 2.5e-4 to 1.25e-4. Uses `StepLR(gamma=0.5)` triggered manually when the wall-clock time crosses TIME_BUDGET/2.

**Why:** In the first half of training, a higher LR speeds up convergence. In the second half, the agent has a rough policy and needs finer gradient updates to improve. Reducing LR at the midpoint prevents overshooting the Q-value estimates when the agent is already performing well. Learning rate scheduling is a standard technique in deep learning that improves final convergence.

**Result:** total_reward=302, score=0. LR decay hurts — constant 2.5e-4 maintains the needed gradient magnitude throughout training. Discarded.

## Experiment 99 — Smaller FC hidden layer (512→256)

**What changed:** FC hidden layer reduced from 512 to 256 neurons. Total FC params drop from ~1.6M to ~800k.

**Why:** The biggest improvement this session was removing Dueling DQN (Exp66) — fewer parameters converge faster in 5 minutes. Following the same logic, reducing the FC layer from 512 to 256 is the next step toward even simpler. 256 is still large enough to represent complex Q-value mappings from the 3136-dim conv output. Fewer params = faster per-step update + less overfitting to early random experiences.

**Result:** total_reward=534, score=100. 512 neurons optimal — 256 lacks capacity to represent the Q-value landscape. Discarded.

## Experiment 100 — 4 gradient updates per environment step

**What changed:** The training loop now performs 4 gradient updates per environment step (instead of 1). Each update samples a fresh batch from the replay buffer.

**Why:** The 5-minute budget limits the total number of environment steps (~15k). Each step takes ~20ms to execute. The GPU has spare capacity — the forward+backward pass is fast. By doing 4 updates per step, we effectively train for 4x longer on the same experience. This is a standard technique ("update-to-data ratio") used in SAC and other off-policy methods to improve sample efficiency. The risk is overfitting to early experiences — mitigated by our diverse 50k replay buffer.

**Result:** total_reward=276, score=0. 4x updates causes severe overfitting to replay buffer and destabilizes Q-values. 1 update/step optimal. Discarded.

---

## Note: SIMPLE_MOVEMENT switch after Exp100

After 34 consecutive failed experiments (Exp67-100), the current DQN config appears to have hit a hard ceiling at 1527 (dist=1127, score=400, 4 kills). All architectural tweaks (Dueling, PER, N-step, Polyak, different losses, LR schedules, weight decay, gradient clipping, batch sizes, etc.) consistently fail to beat this. Starting with Exp101, switching to a fundamentally different approach: **SIMPLE_MOVEMENT (7 actions instead of 12)**. Fewer actions = simpler policy space = potentially much faster convergence to a better policy within the 5-minute window.


## Experiment 101 — SIMPLE_MOVEMENT: 7 actions instead of 12 (PRO_MOVEMENT)

**What changed:** Replaced `PRO_MOVEMENT` (12 actions) with a custom `SIMPLE_MOVEMENT` (7 actions): NOOP, right, right+A (jump right), right+B (run right), right+A+B (run+jump right), A (jump in place), left. Since `evaluate.py` imports `make_env` from `train.py`, this change applies to both training and evaluation.

**Why:** After 34 consecutive failed experiments (Exp67-100), the DQN with PRO_MOVEMENT appears stuck at a hard ceiling of 1527 (x=1127, 4 kills). The Q-network must learn a policy over 12 actions — 7 of which involve left movement or unusual combinations rarely useful in World 1-1. By reducing to 7 essential actions, the Q-network output shrinks from 12 to 7 neurons, and the policy space is much simpler. Following the same principle that removed Dueling DQN (Exp66) for our biggest win, fewer actions = faster convergence in 5 minutes. The missing actions (left+B, left+A, etc.) are not needed to complete World 1-1.

**Result:** total_reward=436, score=0. Fewer actions hurt — PRO_MOVEMENT's left+A, left+B appear necessary for dodging/retreating from enemies. Discarded.

## Experiment 102 — Halved conv channels (32→32→32 instead of 32→64→64)

**What changed:** Conv layer channels reduced from `32→64→64` to `32→32→32`. FC input shrinks from 3136 to 1568. FC hidden layer stays at 512.

**Why:** The biggest win was removing Dueling DQN (fewer params, faster convergence). SIMPLE_MOVEMENT failed (fewer actions doesn't help). Now trying fewer parameters in the conv stack. Half the conv channels means ~4x fewer conv weights. With only 5 minutes of training, the gradient updates might converge to a better policy faster when there's less to optimize.

**Result:** total_reward=594, score=0. Halved conv channels reduce representational capacity too much. 32→64→64 is the right balance. Discarded.

## Experiment 103 — NoisyNet (learned exploration)

**What changed:** Replaced epsilon-greedy exploration with NoisyNet. The two FC layers in DQN (`Linear(3136,512)` and `Linear(512,n)`) are replaced with `NoisyLinear` layers that use factorized Gaussian noise. Noise parameters (sigma) are learned end-to-end. Epsilon is set to 0. Noise is reset before each forward pass.

**Why:** Epsilon-greedy adds uniform random noise regardless of state — it explores equally in well-understood states and novel states. NoisyNet ([Fortunato et al. 2017](https://arxiv.org/abs/1706.10295)) adapts exploration to the current state: uncertain states get more noise, confident states get less. This is one of the Rainbow DQN components. For Mario, the agent needs nuanced exploration — it should be decisive in the early level (well-learned) and exploratory near the x=1127 barrier (uncertain). NoisyNet can provide this automatically.

**Result:** total_reward=312, score=0. NoisyNet without epsilon-greedy explores too little early on, never reaching the level sections where kills happen. Epsilon-greedy is better for this cold-start problem. Discarded.

## Experiment 104 — LR=3e-4 (slightly higher)

**What changed:** RMSprop learning rate 2.5e-4 → 3e-4.

**Why:** We confirmed 2.5e-4 > 2e-4 (Exp89). 2.5e-4 is also better than 1e-4. But we haven't tried anything above 2.5e-4. A slightly higher LR might converge to better Q-values faster in the 5-minute window.

**Result:** total_reward=314, score=0. LR=3e-4 causes Q-value instability. 2.5e-4 is the confirmed optimum. Discarded.

## Experiment 105 — Warm-up: start learning after 5000 transitions

**What changed:** Learning begins only after 5000 transitions in the replay buffer (vs 128 before).

**Why:** Starting gradient updates after only 128 random transitions means the Q-network is learning from extremely limited, correlated experience. The first few thousand steps are purely random anyway (epsilon=1.0). By waiting until 5000 diverse transitions are collected, the first gradient update has a much richer, more representative sample of the state space. This is common practice in DQN implementations — the original Atari paper used 50k warm-up steps. 5000 is scaled to our shorter budget.

**Result:** total_reward=675, score=0. 5000 warm-up steps cost too much training time in the 5-minute budget. Immediate learning from 128 transitions is better. Discarded.

## Experiment 106 — Replay buffer reset at midpoint (t=150s)

**What changed:** At the halfway point of training (150s), the replay buffer is cleared while keeping the network weights. The second half of training uses only the experiences generated by the now-better policy.

**Why:** In the first half, the agent fills the buffer with low-quality random/early experiences. By t=150s, the policy has improved significantly — it's getting to x=700+ and killing enemies. But the 50k-entry buffer still contains many of those early garbage experiences. Clearing it forces the second half to learn from high-quality on-policy data. This is similar to the two-phase training used in AlphaGo: broad exploration phase → focused exploitation phase.

**Result:** total_reward=1926 (NEW RECORD, +27% vs 1527), score=800 (8 kills!), dist=1126. Buffer reset at midpoint dramatically improves kill-learning in the second phase. The cleaner high-quality replay drives better policy convergence. Kept.

## Experiment 107 — Buffer reset at t=100s (1/3 point instead of 1/2)

**What changed:** Buffer reset trigger moved from TIME_BUDGET/2 (150s) to TIME_BUDGET/3 (100s).

**Why:** If earlier reset is better, resetting at 1/3 gives the policy 200 seconds of clean experience instead of 150. The tradeoff: less warm-up time before the reset (lower-quality initial policy), but more time for high-quality learning after.

**Result:** total_reward=778, score=100. Resetting too early (100s) means the policy isn't good enough yet — the second phase doesn't benefit. 150s (Exp106) is the sweet spot. Discarded.

## Experiment 108 — Buffer reset at t=200s (2/3 point)

**What changed:** Buffer reset at 200s (2/3 into training) instead of 150s (1/2).

**Why:** 150s > 100s for the reset. Maybe 200s > 150s too — more warmup means a better policy when the reset happens, so the clean buffer phase learns from even better experience.

**Result:** total_reward=434, score=0. Too late reset (200s) leaves only 100s for clean-buffer learning — not enough time. 150s is optimal. Discarded.

## Experiment 109 — Buffer reset + epsilon restart at t=150s

**What changed:** At t=150s: clear buffer AND reset `steps_done=0` to restart the epsilon decay curve. The second phase gets fresh exploration (eps~1.0 again) combined with the improved network weights.

**Why:** In Exp106, the buffer reset helped but the second phase still had low epsilon (steps_done≈7500, eps≈0.06). The clean buffer phase might benefit from more exploration — the better-initialized network would explore from its current level of competence rather than exploit a near-greedy policy. Restarting epsilon gives the second phase a new warm exploration phase, potentially discovering better routes past x=1127.

**Result:** total_reward=314, score=0. Restarting epsilon also restarts random exploration — the agent loses its exploitation knowledge just when the clean buffer is most valuable. Buffer reset alone (Exp106) is better. Discarded.

## Experiment 110 — Buffer reset + target sync at t=150s

**What changed:** At t=150s: clear buffer AND sync target_net to policy_net (in addition to Exp106's buffer reset).

**Why:** After 150s of training, the target network might lag significantly behind the policy network. At the reset point, syncing target to policy gives the clean second phase a consistent starting point: target and policy are aligned, buffer is fresh. This should reduce the initial instability after the reset.

**Result:** total_reward=436, score=0. Target sync at reset destabilizes training — the natural lag between policy and target provides useful regularization. Exp106 (buffer reset only) is better. Discarded.

## Experiment 111 — Buffer reset to 25k (smaller) at t=150s

**What changed:** At t=150s, buffer resets to capacity 25000 instead of 50000.

**Why:** In the second phase (150-300s), the agent generates ~7500 new steps. A 50k buffer will only be ~15% fresh at the end. A 25k buffer will be ~30% fresh, giving more weight to the most recent (best-quality) experiences. This recency bias could improve the Q-network's accuracy on states the current policy actually visits.

**Result:** total_reward=434, score=0. 25k buffer wraps too quickly — diversity lost. 50k (Exp106) is better. Discarded.

## Experiment 112 — Two buffer resets at t=150s and t=225s

**What changed:** Added a second buffer reset at t=225s (3/4 of training). Phase 1: 0-150s random/early policy. Phase 2: 150-225s medium policy builds better experience. Phase 3: 225-300s only the best experiences from the improved policy.

**Why:** If one buffer reset at 150s helps, two resets might be even better. The second reset at 225s would give the final 75 seconds an ultra-clean, high-quality replay buffer from the already-improved phase-2 policy.

**Result:** total_reward=722, score=0. Second reset at 225s leaves only 75s for final phase — too little time. One reset at 150s (Exp106) is optimal. Discarded.

## Experiment 113 — Buffer reset at 150s + RMSprop momentum=0.9

**What changed:** Added `momentum=0.9` to RMSprop on top of Exp106's buffer reset.

**Why:** Exp84 (momentum alone, no reset) got 1352 with 5 kills. Exp106 (reset alone, no momentum) got 1926 with 8 kills. Together, momentum's faster convergence + buffer reset's quality improvement might compound. The clean second phase with momentum could drive the agent past x=1127.

**Result:** total_reward=1099, score=200. Momentum+reset not synergistic. The momentum destabilizes the clean second phase. Exp106 (reset only) remains best. Discarded.

## Experiment 114 — Buffer reset at t=120s

**What changed:** Buffer reset at 120s instead of 150s.

**Why:** 150s > 100s. Maybe 120s is the actual sweet spot — slightly earlier gives 180s of clean experience instead of 150s, while still having enough warmup time.

**Result:** total_reward=895, score=600 (6 kills). Too-early reset at 120s: high kills but very low distance — agent knows how to kill but policy not mature enough to advance. 150s is optimal. Discarded.

## Experiment 115 — Buffer reset at 150s + dist_mult 3.0→4.0 in phase 2

**What changed:** At t=150s: clear buffer AND increase `dist_mult` from 3.0 to 4.0. The DistanceReward wrapper now exposes `dist_mult` as a mutable attribute.

**Why:** After 150s, the agent is at x=1127 and knows how to kill enemies. The barrier ahead needs stronger incentive to push past. Increasing dist_mult to 4.0 in the second phase gives ~33% more reward for each step forward. Previously, dist_mult=3.5 failed on its own because it was too aggressive from the start. But with the agent already well-trained at 3.0, switching to 4.0 in the clean phase might provide just enough extra motivation to push past x=1127.

**Result:** total_reward=304, score=0. Increased dist_mult destabilizes the clean second phase. The curriculum timing doesn't help when the reward scale shifts. Exp106 alone remains best. Discarded.

## Experiment 116 — Buffer reset at 150s + LR halved to 1.25e-4

**What changed:** At t=150s: clear buffer AND reduce LR from 2.5e-4 to 1.25e-4.

**Why:** The second phase benefits from more conservative updates since the policy is already decent. Lower LR means smaller Q-value adjustments, potentially reducing oscillation as the agent fine-tunes its behavior in the cleaner buffer. This is classic two-phase training: high LR for fast exploration, lower LR for precise convergence.

**Result:** total_reward=534, score=100. LR reduction at reset slows down learning in the critical second phase. Constant 2.5e-4 is better. Discarded.

## Experiment 117 — Buffer reset at 150s + EPS_DECAY=2000

**What changed:** EPS_DECAY reduced from 3000 to 2000, combined with Exp106's buffer reset at 150s.

**Why:** With EPS_DECAY=2000, the agent reaches near-minimum epsilon (~0.1-0.2) around step 2500-3000, which corresponds to ~t=150s. This means the second phase (after buffer reset) starts with the policy already well-exploiting rather than still exploring. The clean buffer gets filled with almost pure on-policy greedy behavior, potentially leading to tighter Q-value estimates.

**Result:** total_reward=313, score=0. Faster decay hurts — EPS_DECAY=3000 is optimal even with the buffer reset. Discarded.

## Experiment 118 — Wider first conv (64 instead of 32) + buffer reset

**What changed:** First conv layer channels 32→64 (Conv(4,64,8,4)). Rest of architecture unchanged (FC input stays at 3136 since 64×7×7=3136). Combined with Exp106's buffer reset at 150s.

**Why:** The current first conv layer (32 channels) may not have enough feature detectors to reliably identify enemies (Goombas, Koopas are ~16×16 pixels at 84×84 resolution). Doubling to 64 channels gives more feature diversity in the first layer. Since FC input size doesn't change (3136), this adds only ~2x more conv-1 parameters while keeping the fully-connected network identical.

**Result:** total_reward=307, score=0. More first-layer channels slow convergence. Standard 32→64→64 is optimal. Discarded.

## Experiment 119 — Batch normalization in conv layers + buffer reset

**What changed:** Added BatchNorm2d after each conv layer. Policy_net switches to `.eval()` for single-sample action selection (required for correct BatchNorm statistics), then back to `.train()` for gradient updates. Combined with buffer reset at 150s.

**Why:** BatchNorm normalizes feature activations within each batch, improving gradient flow and potentially accelerating convergence. For DQN with 84×84 inputs, the conv activations can have large variance — BatchNorm keeps them in a stable range. This could help the network learn more efficiently in both the initial exploration phase and the clean second phase.

**Result:** total_reward=1299, score=400. BatchNorm is less effective than the clean DQN — the extra complexity and eval/train switching overhead hurts more than the normalization helps. Discarded.

## Experiment 120 — A2C (Advantage Actor-Critic)

**What changed:** Completely replaced DQN+replay buffer with A2C. New `ActorCritic` network shares the same conv backbone as DQN but has separate actor (policy logits) and critic (value) heads. Training: collect 128-step trajectories, compute n-step returns with GAE, update with policy_loss + 0.5*value_loss - 0.01*entropy. RMSprop(lr=7e-4). At save time, weights are exported in DQN-compatible format so evaluate.py still works. The DQN class is kept for evaluate.py compatibility.

**Why:** After 50+ failed experiments from the DQN 1527 ceiling (and a breakthrough to 1926 with buffer reset), DQN variants still can't push past x=1127. A2C is fundamentally different: on-policy learning, no replay buffer staleness, policy gradient instead of value regression. A2C is known to work well in structured environments and might find a different optimal policy path through the level. References: [Mnih et al. 2016 A3C paper](https://arxiv.org/abs/1602.01783).

**Result:** total_reward=314, score=0. A2C stochastic policy does not transfer well to greedy evaluation — entropy regularization makes the greedy (argmax) policy ineffective. DQN+buffer reset (1926) remains the best approach. Discarded.

## Experiment 121 — Frontier bonus (×2 extra when pushing past episode best x)

**What changed:** DistanceReward now tracks `max_x` per episode. When `x_pos > max_x` (new frontier), gives extra `(x_pos - max_x) * 2.0` bonus on top of the regular `delta_x * 3.0`. Combined with buffer reset at 150s.

**Why:** The agent is stuck at x=1127. When it goes beyond that point (a new frontier), it should get a stronger pull to keep going. The regular `delta_x * 3.0` reward applies everywhere. The frontier bonus makes "going further than ever before" specifically more rewarding — like a personal record incentive. This asymmetry between retreating and advancing at the frontier encourages the agent to push past the barrier it keeps dying at.

**Result:** total_reward=1022, score=300. Frontier bonus disrupts reward scale — agents prefer the frontier bonus over sustained progress. Buffer reset alone (1926) is better. Discarded.

---

## Exp122: SIMPLE_MOVEMENT (7 actions) + buffer reset at 150s

**What changed:** Replaced PRO_MOVEMENT (12 actions) with SIMPLE_MOVEMENT (7 actions): NOOP, right, right+A, right+B, right+A+B, A, left. Combined with the buffer reset at 150s (current best: 1926).

**Why:** Fewer actions = smaller Q-table output head = less to learn in 5 minutes. With 7 actions instead of 12, the DQN has 42% fewer output units and can converge faster on the action distribution. SIMPLE_MOVEMENT alone (Exp101) scored only 436 — likely because it lacked left+A/left+B for retreating from enemies. However, when combined with the buffer reset (which produces higher-quality second-half experience), the agent may learn a good 7-action policy more effectively than an imperfect 12-action one. The evaluate.py imports make_env from train.py so this change applies to both training and evaluation.

**Result:** total_reward=693, score=0, max_x_dist=693. SIMPLE_MOVEMENT restricts retreating options too much — agent gets stuck or dies earlier. PRO_MOVEMENT with buffer reset (1926) remains best. Discarded.

---

## Exp123: 3-step returns + buffer reset at 150s

**What changed:** Added n-step returns (n=3) to the DQN training. Instead of storing single-step (s, a, r, s', done) transitions, we accumulate 3 steps of discounted reward: G = r_t + γ*r_{t+1} + γ²*r_{t+2}, then push (s_t, a_t, G, s_{t+3}, done) to the replay buffer. The Q-target also uses γ³ (gamma^3) for bootstrapping. Combined with buffer reset at 150s.

**Why:** The agent is stuck at x=1127 (enemy encounter). With 1-step TD, reward signals from successfully navigating the enemy area must propagate back one step at a time — slow credit assignment. With n=3, each gradient update propagates useful signal 3 steps back instead of 1. This is a key component of Rainbow DQN [Hessel et al. 2017]. For a 5-minute training budget where every gradient step counts, faster credit propagation could help the agent learn the critical action sequences needed to pass x=1127.

**Result:** total_reward=304, score=0, max_x_dist=304. Much worse — 3-step returns destabilize early training, likely because the accumulated reward scale mismatches the Q-value expectations. Discarded.

---

## Exp124: 2 gradient updates per env step + buffer reset at 150s

**What changed:** After each environment step, run 2 gradient updates (instead of 1) by sampling the replay buffer twice. Everything else unchanged: same model, same optimizer, same buffer reset at 150s.

**Why:** Within a fixed 5-minute budget, the agent can only collect so many env steps. But the GPU has spare capacity between steps. Doing 2 gradient updates per step doubles the learning signal extracted from each experience without collecting more data. The tradeoff is slightly fewer env steps per second (more GPU compute per step), but the net effect on a GPU is likely positive because GPU utilization is low with 1 update per step. This is similar to the "update ratio" hyperparameter in SAC/TD3 research where higher ratios often improve sample efficiency.

**Result:** total_reward=694, score=100, max_x_dist=594. Worse — double updates per step likely causes Q-value overestimation and instability from reusing the same experience too aggressively. Discarded.

---

## Exp125: Partial buffer prune (keep newest 50%) at 150s

**What changed:** At t=150s instead of clearing the entire replay buffer, we prune the oldest 50% of experiences and keep the most recent 50%. The newer half consists of transitions from a more competent policy (epsilon already decayed), so they're higher quality. The second phase starts with ~25k decent experiences rather than 0.

**Why:** The full buffer reset (Exp106, 1926) eliminates early random noise but also throws away the most recent good experiences, forcing a cold start. The 2nd phase then takes time to re-fill with quality data before meaningful training can happen. By keeping the newest 50%, we skip that cold start and let the 2nd phase train from good data immediately. Think of it as a "soft reset" — aggressively pruning the noisy part while preserving the gold.

**Result:** total_reward=315, score=0, max_x_dist=315. Much worse — the retained "old" experiences contaminate the second phase. The full clean reset (1926) is better. Discarded.

---

## Exp126: Buffer reset at 100s (1/3 budget) — longer clean phase

**What changed:** Moved the replay buffer reset from t=150s (TIME_BUDGET/2) to t=100s (TIME_BUDGET/3). This gives the clean second phase 200s instead of 150s.

**Why:** At t=100s, epsilon has decayed to ~0.08 (EPS_DECAY=3000, ~10k steps at ~100 steps/s). The policy is already mostly exploitative. By resetting earlier, we sacrifice 50s of "contaminated" buffer training but gain 50s of additional clean-phase gradient updates. More quality training time in the second phase could help the agent learn the tricky x=1127 area with more gradient updates from high-quality experiences.

**Result:** total_reward=534, score=100, max_x_dist=434. Worse — policy at 100s is not mature enough; second phase starts too weak. 150s reset remains optimal. Discarded.

---

## Exp127: Asymmetric x-reward (forward 3.0x, retreat 1.0x) + buffer reset at 150s

**What changed:** Modified DistanceReward to apply different multipliers for rightward vs. leftward movement: `delta_x * 3.0` when moving right, `delta_x * 1.0` when moving left. Previously both directions used the same 3.0x multiplier (retreat was heavily penalized).

**Why:** The agent is stuck at x=1127 where a goomba likely blocks the path. With symmetric 3.0x penalties, retreating even one step costs -3x reward, discouraging tactical backpedaling. A skilled player would briefly retreat to time a jump over the enemy. With asymmetric rewards (1.0x retreat penalty), the agent can move left without catastrophic reward loss, potentially learning to dodge-and-advance patterns at the x=1127 barrier. Forward progress is still strongly rewarded at 3.0x.

**Result:** total_reward=534, score=100, max_x_dist=434. Worse than 1926, retreat penalty reduction did not help. Discarded.

---

## Exp128: Wider FC layer (1024) + buffer reset at 150s

**What changed:** Increased the fully-connected hidden layer from 512 to 1024 neurons. The conv feature extraction is unchanged (3136 output). FC is now 3136→1024→n_actions vs. previous 3136→512→n_actions.

**Why:** The agent consistently reaches x=1127 but fails to learn the dodge/jump timing to pass that barrier. More FC capacity means more parameters dedicated to mapping visual features to action decisions. With 1024 neurons, the network can represent more complex action-value functions. The VRAM increase is minimal (3136*1024 vs 3136*512 ≈ +3M params ≈ +12MB). In 5 minutes on GPU, this extra capacity may allow the agent to learn subtler distinctions in the Q-value landscape near x=1127.

**Result:** total_reward=314, score=0, max_x_dist=314. Much worse — larger model needs more time to converge, 5-min budget is not enough for 1024 neurons. 512 remains optimal. Discarded.

---

## Exp129: Death penalty (-50) + buffer reset at 150s

**What changed:** Added explicit -50 reward on death (when `terminated=True` and `flag_get=False`). The base env already gives a small negative reward on death (game clock penalty), but a larger explicit penalty makes survival more strongly incentivized.

**Why:** The agent dies at x=1127 repeatedly. A larger death penalty makes the cost of failing at that barrier very high, potentially training the agent to be more cautious — learning to jump over enemies precisely rather than rushing into them. With the current reward structure, dying "costs" relatively little (just delta_x going to 0 and -0.1/step). A -50 penalty significantly increases the cost of every death and should push the agent toward risk-averse strategies like well-timed jumps.

**Result:** total_reward=314, score=0, max_x_dist=314. Worse — death penalty makes agent too conservative, stops progressing. Discarded.

---

## Exp130: TARGET_UPDATE=1000 + buffer reset at 150s

**What changed:** Doubled target network update frequency from every 500 steps to every 1000 steps.

**Why:** Less frequent target updates means the Q-value targets used in training are more stable for longer. In the original DQN paper [Mnih et al. 2015], TARGET_UPDATE was set to 10000 steps. Our current 500 (every ~5 seconds) might cause instability near the x=1127 barrier where Q-values need to precisely distinguish similar states. With 1000-step intervals (~10 seconds), the bootstrap targets don't "chase" the policy network as aggressively, potentially allowing cleaner convergence.

**Result:** total_reward=314, score=0, max_x_dist=314. Worse — more stable targets but agent learns less effectively without frequent syncs. Discarded.

---

## Exp131: MEMORY_SIZE=20000 + buffer reset at 150s

**What changed:** Reduced replay buffer from 50k to 20k transitions.

**Why:** After the buffer reset at t=150s, the agent collects ~15000 steps in the remaining 150s (at ~100 steps/sec). With a 50k buffer, the clean phase only fills 30% of capacity — the agent is always sampling a relatively sparse buffer. With 20k, the clean phase fills 75% of the buffer, and sampling covers a denser, more varied set of quality experiences. More importantly, once the buffer is ~full, the oldest experiences are evicted, keeping the training focused on the agent's most recent (and best) behavior.

**Result:** total_reward=801, score=100, max_x_dist=701. Better than many experiments but still worse than 1926. 50k buffer provides more diverse sampling in the clean phase. Discarded.

---

## Exp132: EPS_DECAY=10000 (sustained exploration) + buffer reset at 150s

**What changed:** Increased EPS_DECAY from 3000 to 10000. With EPS_DECAY=10000, epsilon decays much slower: at the 150s reset point (~15k steps), epsilon is still ~0.26. By 300s (~30k steps), epsilon is ~0.097.

**Why:** With EPS_DECAY=3000, the agent becomes fully exploitative (~eps=0.05) by ~30s and stays there for the entire 270s. If the policy learned by 30s is suboptimal at the x=1126 barrier, it stays stuck. With EPS_DECAY=10000, the second phase (after the buffer reset) still has epsilon ~0.26, meaning 26% of actions are exploratory. This randomness could help the agent try different actions at the x=1126 barrier (e.g., jumping at slightly different times) and potentially discover the successful trajectory through that obstacle.

**Result:** total_reward=308, score=0, max_x_dist=308. Worse — too much exploration prevents policy convergence; model never learns to reliably exploit the good path. Discarded.

---

## Exp133: SuperMarioBros-v2 (simplified graphics) + buffer reset at 150s

**What changed:** Changed environment from `SuperMarioBros-v0` (full NES graphics) to `SuperMarioBros-v2` (simplified rectangular graphics with reduced visual noise). Both train and evaluate use make_env from train.py, so this change applies to both.

**Why:** The v2 environment strips out decorative background elements (clouds, bushes, detailed sprites) and renders enemies and obstacles as simple colored rectangles. The CNN should more easily learn to detect enemies and obstacles from simplified graphics vs. complex pixel-art, potentially converging faster within the 5-minute budget. The level layout and gameplay mechanics are identical between v0 and v2 — only the visual representation changes.

**Result:** total_reward=435, score=0, max_x_dist=435. Worse — simplified graphics may reduce visual detail needed to distinguish enemies. v0 remains better. Discarded.

---

## Exp134: Max-x only reward (no retreat penalty) + buffer reset at 150s

**What changed:** Rewrote DistanceReward to track `max_x` per episode instead of `curr_x`. Reward is only given for exceeding the episode's previous maximum x position: `(x_pos - max_x) * 3.0`. Retreating from max_x gives zero x-reward (not negative). Step penalty (-0.1) and score bonus (+0.3) unchanged.

**Why:** The current implementation penalizes retreating at 3x rate, which discourages the agent from briefly backing up to time enemy jumps. By rewarding only new progress and ignoring retreats, the agent can freely move left/right to optimize timing without reward-penalty for tactical positioning. The step penalty (-0.1) still incentivizes overall speed. This could help the agent learn retreat-and-jump patterns at x=1126 that require brief backward movement.

**Result:** total_reward=434, score=0, max_x_dist=434. Worse — max-x only reward is too sparse; agent learns to stop advancing because moving left/right within known territory has no positive signal. Discarded.

---

## Exp135: Remove score bonus (0 instead of 0.3) + buffer reset at 150s

**What changed:** Removed the `score * 0.3` bonus from DistanceReward. The agent now receives no extra reward for killing enemies or collecting coins. Reward is purely distance-based (delta_x * 3.0) plus the step penalty and flag bonus.

**Why:** The current score bonus (+30 per kill) might be incentivizing the agent to STOP and stomp enemies at x=1126 rather than dodge past them. If the agent can jump OVER the enemy without stomping (getting 0 score but advancing to x=1127+), the score bonus creates a distraction. By removing the score bonus, the agent is incentivized purely to advance rightward, potentially discovering a dodge path past x=1126 that doesn't require killing the enemy.

**Result:** total_reward=525, score=100, max_x_dist=425. Worse — score bonus helps agent learn to engage enemies; removing it weakens the policy. Discarded.

---

## Exp136: Variance probe — trivial change (1 → 1.0) on same config

**What changed:** Trivial cosmetic: changed `(1 - dones)` to `(1.0 - dones)` in target Q computation. Functionally identical.

**Why:** After 30+ experiments all failing to improve on 1926, this run probes whether 1926 is reliably reproducible (low variance) or was a lucky outlier (high variance). If we get ~1900+ again, the config is stable and future experiments are working against a reliable baseline. If we get ~500, the baseline has high stochasticity and we should try harder to find a genuinely better algorithm.

**Result:** total_reward=306, score=0, max_x_dist=306. Identical code as 1926 baseline but got 306. Confirms extremely high variance in training — 1926 was likely a lucky run. Need more stable algorithm. Discarded.

---

## Exp137: C51 Distributional DQN + buffer reset at 150s

**What changed:** Replaced standard DQN with C51 distributional DQN (Bellemare et al. 2017). Instead of predicting a single expected Q-value per action, the network predicts a PROBABILITY DISTRIBUTION over 51 discrete return values (atoms) from V_MIN=-50 to V_MAX=500. Action selection uses the expected value (sum of z_i * p_i). Training uses distributional Bellman target projected onto the support, with cross-entropy loss instead of Huber/MSE.

**Why:** The variance probe (Exp136) showed that running the exact same 1926 code produced only 306. This extreme variance suggests the standard DQN training is unstable. C51 is known to produce more stable, more accurate value estimates because:
1. Distributional estimates are more informative than scalars — the network explicitly represents return uncertainty
2. Cross-entropy loss is more stable than regression (MSE/Huber) for learning distributions
3. Multimodal returns (like the x=1126 barrier where some trajectories get very high reward and others get low) are better captured by distributions

Buffer reset at 150s preserved. evaluate.py compatible: forward() returns expected Q-values [batch, n_actions].

**Result:** first run V_MAX=500 too narrow: eval got 40 (train=1300 score). Second run V_MAX=2000: eval got 314 (train=1400 score). C51 trains well but greedy eval consistently fails — distributional Q-values are hard to calibrate for stable argmax in this env. Discarded.

---

## Exp138: Best-model checkpointing (save peak second-phase model) + buffer reset

**What changed:** During the second phase (after buffer reset at 150s), save `model.pt` whenever the training performance proxy (game_time + score) hits a new peak, rather than saving only the final model at t=300s. The `finally` block no longer overwrites this best model.

**Why:** Exp136 (variance probe) showed running identical code produced 306 vs 1926. This extreme variance suggests the model's quality degrades after it peaks — the final model at t=300s is often WORSE than the peak model at some earlier time. By saving the best-performing second-phase model and using that for evaluation, we sidestep the end-of-training degradation. This is analogous to "early stopping" in supervised learning.

**Result:** total_reward=895, score=200, max_x_dist=695. Better than many recent experiments (less degradation) but still below 1926 record. The best training proxy (time+score) does not directly align with eval metric (x_dist+score). Discarded.

---

## Exp139: Save model at peak x_pos (clean phase) instead of final model

**What changed:** Instead of saving `model.pt` at the end of training, save it whenever a new maximum x_pos is achieved during the clean second phase (after buffer reset at 150s). This directly optimizes model selection for the eval metric: `max_x_dist`.

**Why:** Exp138 tried saving at peak `time+score` (training proxy) and got 895 — better than pure variance but not 1926. The training proxy (time+score) doesn't perfectly align with the eval metric (x_dist+score). By saving at peak x_pos directly, we save the model at the exact moment it was at its BEST in terms of what the evaluator measures. If the model achieves x=1126 at some point during training but degrades by t=300s, we'd save the 1126 model rather than the degraded one.

**Result:** total_reward=676, score=0, max_x_dist=676. The peak x_pos model achieves different area (pipe section, no kills) vs baseline (1126, 8 kills). Peak-x checkpoint doesn not reliably improve eval. Discarded.

---

## Exp140: MEMORY_SIZE=5000 (rolling ~50s buffer) + buffer reset at 150s

**What changed:** Dramatically reduced replay buffer from 50k to 5k transitions. With ~100 steps/sec, 5000 transitions = ~50 seconds of experience. The deque(maxlen=5000) continuously evicts old transitions, so training always uses the most recent 50 seconds of gameplay.

**Why:** Variance probe (Exp136) showed training is highly stochastic. One cause: the 50k buffer mixes transitions from many different policy "versions" (eps=1.0 early, eps=0.05 later). A 5k rolling buffer ensures training is always on-policy — the agent continuously retrains on its CURRENT behavior, not 5-minute-old behavior. Combined with the buffer reset at 150s (which eliminates first-phase noise), the second phase would always train on the most recent ~50s of clean experience. This is closer to online learning (like PPO's policy update style) but within the DQN framework.

**Result:** total_reward=695, score=100, max_x_dist=595. Worse — small buffer reduces diversity; gradient updates become too correlated with current behavior and destabilize training. Discarded.

---

## Exp141: Polyak soft target update (τ=0.005) + buffer reset at 150s

**What changed:** Replaced the hard target network copy every 500 steps with Polyak soft updates every step: `target = 0.995*target + 0.005*policy`. This continuously but slowly updates the target network.

**Why:** Hard target copies (every 500 steps) cause discrete jumps in the Q-value targets. Between copies, targets are stale by up to 500 steps. This creates training oscillations. Soft updates (Polyak averaging with τ=0.005) make the target change smoothly — at each step, 0.5% of the policy net leaks into the target. This is more stable and is the standard in DDPG/SAC. For DQN, studies show soft updates can reduce oscillations and improve convergence stability. May reduce the extreme variance seen in Exp136.

**Result:** total_reward=296, score=0, train_best_score=1500 (15 kills). Soft updates destabilize eval despite good training metrics — target continuously changing prevents stable Q-value convergence for greedy exploitation. Hard copies every 500 steps remain better. Discarded.

---

## Exp142: Fixed random seed (42) for reproducible training

**What changed:** Added fixed random seeds at the start of train(): `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`.

**Why:** Exp136 showed identical code gives 306 or 1926 (6x difference). This extreme variance makes it impossible to tell if algorithm changes are genuinely better. A fixed seed makes training deterministic (or near-deterministic for CUDA), so future experiments can reliably compare against this baseline. If seed=42 gives ~1926, we have a reproducible foundation to build on.

**Result:** total_reward=801 with seed=42. Better than pure variance (306) but short of 1926. seed=0 gave 315. Retrying with seed=42+cudnn.deterministic gave 272 (slower due to determinism overhead), seed=42 gave 801, seed=0 gave 315. Fixed seeds do not reproduce 1926 — that was a specific lucky stochastic trajectory. Discarding all seed experiments.

---

## Exp143: 64×64 frame resolution (vs 84×84) + buffer reset at 150s

**What changed:** Reduced observation frame size from 84×84 to 64×64. The conv network output shrinks from 3136 (7×7×64) to 1024 (4×4×64), so the FC layer is now 1024→512→n_actions.

**Why:** Smaller frames = faster forward/backward pass = more gradient updates per second within the 5-minute budget. With 84×84, the conv layers process 7056 values per frame; with 64×64, only 4096 — a 42% reduction. More gradient steps in the same time may lead to better policy convergence, even if each step uses slightly lower-resolution features. The spatial detail lost might be acceptable since the key signal (enemy presence/position) is detectable at 64×64 resolution.

**Result:** total_reward=823, score=100, max_x_dist=723. Better than some experiments but below 1926. Smaller model trains faster (62MB vs 109MB VRAM) but 84x84 resolution still better for distinguishing enemies. Discarded.

---

## Exp144: Frontier exploration boost (eps=30% near global max_x, clean phase only)

**What changed:** Added targeted exploration: in the clean second phase (after buffer reset at 150s), whenever the agent is within 100 pixels of the global all-time max_x, epsilon is boosted to at least 30% (from ~5%). This forces random action sampling near the x=1126 barrier.

**Why:** The agent consistently dies at x~1126 with the standard 5% epsilon. With 5% epsilon and 12 actions, the probability of trying any specific action at x=1126 is only 5%/12 ≈ 0.4%. Over the entire training run, the agent might only try the correct "jump over enemy" action there 1-2 times. With 30% epsilon at the frontier: 30%/12 ≈ 2.5% per action, giving ~6x more exploration attempts at the critical point. The clean-phase restriction prevents this from hurting early exploration.

**Result:** total_reward=314, score=0, train_best_score=20950 (impossible — score overflow or bug). The frontier boost caused massive instability: 30% random near max_x polluted the replay buffer. Discarded.

---

## Exp145: Mixed precision training (AMP/FP16) + buffer reset at 150s

**What changed:** Added PyTorch Automatic Mixed Precision (AMP) to the training loop. Forward and backward passes now run in FP16 where possible (via `autocast()`), with a `GradScaler` to prevent FP16 gradient underflow. Gradient unscaling is done before clipping.

**Why:** On NVIDIA GPUs with Tensor Cores (RTX series), FP16 matrix multiplications are 2-8x faster than FP32. More training steps per second = more gradient updates within the 5-minute budget = better policy convergence. The GradScaler ensures numerical stability by scaling gradients to prevent FP16 underflow. This is a pure speedup with no algorithmic change.

**Result:** total_reward=435, score=0, train_best_score=700. AMP speedup did not improve convergence quality in 5 min — possibly AMP overhead offset any gains, or FP16 precision hurt gradient quality. Discarded.

---

## Exp146: Buffer reset at episode boundary (after 150s, on next done=True)

**What changed:** Added `and done` to the buffer reset condition. Previously reset happened at exactly t=150s mid-episode. Now it waits until the current episode ends (done=True) after the 150s mark.

**Why:** The original reset at t=150s might cut off a good episode mid-run. If the agent is at x=1100 and making progress at t=150s, the reset clears valuable ongoing-episode transitions. By waiting until the episode ends (done=True), we complete the current episode fully before clearing the buffer. The actual reset now happens at 150s + (remaining episode time), typically 150-200s depending on episode length.

**Result:** total_reward=694, score=100, max_x_dist=594. Worse — waiting for episode end delays the clean phase too much (less clean training time). Mid-episode reset at exactly 150s is better. Discarded.

---

## Exp147: Reset optimizer state at buffer reset (150s)

**What changed:** At the buffer reset (t=150s), also reinitialize the RMSprop optimizer (creating a new instance). This discards the accumulated running average of squared gradients from phase 1.

**Why:** RMSprop accumulates gradient magnitude history (the denominator in the update rule). Phase-1 gradients are noisier (random actions, mixed buffer). These accumulated statistics could bias the phase-2 learning rate per-parameter in unhelpful directions. By resetting the optimizer, phase 2 starts with fresh per-parameter learning rates. This is analogous to "learning rate warmup" but achieves it by starting clean rather than ramping up.

**Result:** total_reward=912, score=600 (6 kills), max_x_dist=312. Higher kill count but very low x_dist — optimizer reset changes the learning dynamics to favor kill-farming over advancement. Below 1926. Discarded.

## Exp148: Reward Clipping to [-1, 1]

**What changed:** Clip rewards stored in the replay buffer to the range [-1, 1]:
```python
memory.push(state, action, np.clip(reward, -1.0, 1.0), next_state, done)
```

**Why:** The original DQN paper (Mnih et al. 2015) used reward clipping to [-1, 1] as a training stability technique. Without clipping, large rewards (like the flag bonus of 1000) can cause very large Q-value targets, destabilizing training. Clipping normalizes the reward scale so gradients don't blow up. The tradeoff is that the agent loses the ability to distinguish between different magnitudes (a flag completion and a small coin look the same in magnitude). However, the clip still preserves the sign, so the agent knows good from bad — and the relative frequency of positive vs negative transitions may be enough signal.

**Result:** total_reward=289, max_x_dist=289, score=0. Reward clipping removes signal magnitude. Discarded.

## Exp149: N-step Returns (n=3)

**What changed:** Instead of pushing single-step (s, a, r, s', done) transitions to the replay buffer, accumulate n=3 consecutive transitions and push the n-step return: r0 + γ*r1 + γ²*r2, with s' being the state 3 steps ahead. The Q-learning target becomes: Q = n_step_return + γ^3 * max_a Q(s_n, a). Early episode termination is handled by clearing the n-step buffer on done. Also update the discount factor in the Bellman target to γ^3 instead of γ^1.

**Why:** N-step returns reduce the bias in Q-value estimates by using more real experience before bootstrapping. In standard 1-step DQN, each update only propagates reward information backward by 1 step. With n=3, reward from 3 steps ahead is directly attributed to the current state-action pair without needing 3 separate training updates to propagate. This is particularly useful in Mario where rewards are dense (distance-based) — the agent can more directly learn that "running right leads to more reward" rather than needing many Bellman iterations. Reference: Sutton & Barto RL book, Chapter 7 (n-step TD).

**Result:** total_reward=434, max_x_dist=434, score=0. N-step buffering slows replay buffer fill (3x fewer pushes). Agent progresses further than reward-clipped version but far below baseline. Discarded.

## Exp150: Save Best Checkpoint Instead of Final Model

**What changed:** During training (after the buffer reset at t=150s), save `model.pt` whenever a new best episode reward is seen. The final `finally` block no longer overwrites this — it only saves if no checkpoint was written (i.e., training crashed before getting past warmup).

**Why:** The current code saves the FINAL model (the one at the exact end of training). But the best-performing model during training might appear before the end — training can oscillate, and the last episode's model might have degraded from a prior peak. By tracking the best episode reward post-buffer-reset and checkpointing whenever a new best is seen, we preserve the peak-performing weights for evaluation. This is a simple change that could recover performance lost to late-training instability.

**Result:** total_reward=823, max_x_dist=723, score=100. Best-episode checkpoint reaches x=723 but still below 1926. Episode reward metric (includes step penalties) doesn't directly optimize eval x_pos. Discarded.

## Exp151: Save Checkpoint at Best X Position (Post Buffer Reset)

**What changed:** After the buffer reset (t=150s), save `model.pt` whenever the agent reaches a new record x_pos during training. The final model save is bypassed if any checkpoint was written. This preserves the model that reached the furthest point in the level, rather than the model at the end of training.

**Why:** Exp150 showed that saving the best-episode checkpoint helps (x=723 vs ~300-400). But episode reward includes step penalties (-0.1/step) and distance bonuses that don't directly match eval. The eval's total_reward is dominated by max_x_dist (or flag_get). By saving when x_pos hits a new record, we directly checkpoint the model that produced the best eval-relevant behavior — the furthest point reached. This should align training checkpointing with the eval objective better than episode reward.

**Result:** total_reward=1234, max_x_dist=1234, score=0. Best x_pos checkpoint consistently reaches further (1234) than saving the final model. Below 1926 but a notable improvement in consistency. Discarded.

## Exp152: Early Buffer Reset (t=100s) + Best-X Checkpoint

**What changed:** Two combined changes:
1. Buffer reset timing moved from t=150s (half budget) to t=100s (one-third of budget). This gives 200 seconds instead of 150 seconds for clean phase-2 training.
2. Best-x checkpoint: save model.pt whenever the agent reaches a new max x_pos (after buffer reset). This was Exp151 individually.

**Why:** Exp151 showed best-x checkpoint alone gives total_reward=1234. The hypothesis is that more phase-2 training time will push the agent further right. Moving the reset from t=150s to t=100s gives 33% more clean training time. The tradeoff is less phase-1 time to build up initial Q-value estimates before the reset. But since phase-1 is exploratory (high epsilon), 100s should be enough to learn basic movement direction — the real learning happens in phase 2 with more targeted experiences.

**Result:** total_reward=314, max_x_dist=314, score=0. Earlier reset at t=100s does not give the model enough time to build useful Q-value estimates before the reset. Phase-1 learning is too shallow, phase-2 starts from near-zero understanding. Discarded.

## Exp153: Best-X Checkpoint + Epsilon Warmup at Buffer Reset

**What changed:** Three changes combined:
1. Best-x checkpoint (same as Exp151): save model.pt when agent reaches a new record x_pos after buffer reset.
2. Epsilon warmup at buffer reset: at t=150s when the buffer resets, also set `steps_done = steps_done // 2`. This halves the accumulated step count, temporarily raising epsilon back up to allow exploration in the newly clean phase.
3. Buffer reset timing stays at t=150s (midpoint) — confirmed better than t=100s.

**Why:** Exp151 (best-x checkpoint alone at t=150s) gave 1234. The epsilon at t=150s is very low (~0.05), meaning phase 2 is almost pure exploitation. But the clean buffer means the agent can actually explore new trajectories productively. By halving steps_done, epsilon gets boosted (e.g., from ~0.05 back to ~0.12) at the reset, then decays naturally. This should help the agent explore deeper into the level in phase 2 before settling into exploitation. Combined with best-x checkpointing, the best position reached during this exploratory phase 2 is preserved.

**Result:** total_reward=300, max_x_dist=300, score=0. Epsilon warmup at reset over-explores in phase 2. Halving steps_done raises epsilon too much, producing poor-quality phase-2 experiences that negate the clean buffer benefit. Discarded.

## Exp154: Best-X Checkpoint at Episode End (not mid-episode)

**What changed:** Instead of saving model.pt mid-episode when a new x_pos record is hit, save it at the END of the episode that set the new x_pos record (only in phase 2, after buffer reset). Track `episode_max_x` per episode and update `best_x_pos_ever` after episode ends.

**Why:** Exp151 saved mid-episode when current_x hit a new record. This meant the saved model might be mid-update (only trained on the experience up to that exact step). By saving at episode end instead, the model has processed the entire episode's batch updates before being checkpointed. This should give a more fully-trained version of the model that reached the best x, rather than a snapshot taken in the middle of an episode.

**Result:** total_reward=43, max_x_dist=43. End-of-episode save is counterproductive: the model at episode end has trained on the death/failure experiences after the x record was set, degrading the weights. The mid-episode checkpoint (Exp151) was better. Discarded.

## Exp155: X-Milestone Bonuses + Best-X Mid-Episode Checkpoint

**What changed:** Two changes:
1. Add one-time milestone bonuses to DistanceReward: +50 reward when Mario first crosses x=500, 1000, 1500, 2000, 2500, 3000. Milestones reset each episode.
2. Best-x mid-episode checkpoint (same as Exp151): save model.pt when x_pos hits a new record during training (after buffer reset).

**Why:** The agent consistently stops around x=1234 despite the distance reward encouraging rightward movement. Milestone bonuses provide explicit signal that crossing certain thresholds is highly rewarded — like sub-goals. The +50 bonus at x=1000 (roughly where the agent gets stuck) is comparable to ~16 steps of continuous progress, providing a strong motivation to push past that specific barrier. Combined with best-x checkpoint to preserve the weights that achieved the furthest position.

**Result:** total_reward=314, max_x_dist=314, score=0. Milestone bonuses of +50 distort Q-value estimates significantly. The agent underperforms even vs the best-x checkpoint alone. Discarded.

## Exp156: 2 Gradient Steps Per Env Step + Best-X Checkpoint

**What changed:** Two changes:
1. Best-x mid-episode checkpoint (same as Exp151).
2. 2 gradient updates per env step: for each environment step, sample two different random batches and do two gradient steps instead of one.

**Why:** The training is env-step-limited (only 5 minutes of game time). By doing 2 gradient updates per env step, we effectively double the number of Q-value updates without needing more environment interaction. This is the "higher replay ratio" technique used in SAC (Soft Actor Critic) and similar methods. With 2 updates per step, the policy converges faster to better Q-value estimates, potentially helping the agent push past the x=1234 barrier more quickly. The cost is ~2x GPU compute per env step, but since we're not GPU-bound (env simulation is the bottleneck), this should be nearly free.

**Result:** total_reward=434, max_x_dist=434, score=0. Higher replay ratio (2 updates/step) causes Q-value instability. Worse than best-x checkpoint alone (1234). Discarded.

## Exp157: Best-X Checkpoint + LR Reduction for Phase 2

**What changed:**
1. Best-x mid-episode checkpoint (same as Exp151).
2. At the buffer reset (t=150s), reduce LR from 2.5e-4 to 1e-4 for phase 2.

**Why:** Phase 1 uses LR=2.5e-4 which is good for fast learning from diverse noisy data. But once the buffer is reset with clean experiences in phase 2, a lower LR allows for more precise, stable refinement. This is similar to learning rate annealing — use a large LR early to explore the loss landscape, then a smaller LR to converge more stably. Combined with best-x checkpoint to preserve the furthest model.

**Result:** total_reward=311, max_x_dist=311, score=0. LR reduction at buffer reset slows phase-2 adaptation too much. Agent cannot learn quickly enough from new clean experiences. Discarded.

## Exp158: Wider Conv Layers (4->64->128->64) + Best-X Checkpoint

**What changed:**
1. Wider conv network: changed from 4→32→64→64 to 4→64→128→64. This doubles the channels in the first two layers while keeping the same stride/kernel config. The final conv layer uses 64 channels (same as before), so the FC input remains 64*7*7=3136. The wider network has ~2.5x more conv parameters but the same FC layer.
2. Best-x mid-episode checkpoint (same as Exp151).

**Why:** The standard DQN conv channels (32→64→64) might not be wide enough to capture all relevant visual features in complex Mario sections. Wider channels allow the network to learn more diverse feature detectors — important for distinguishing fine-grained differences like enemy positions, pipe heights, and gap widths. Current VRAM is only 109MB (very low), so we have plenty of room to grow.

**Result:** total_reward=311, max_x_dist=311, score=0. Wider conv uses 135MB VRAM (up from 109MB) and more parameters slows convergence in the 5-minute budget. Discarded.

## Exp159: Count-Based Intrinsic Exploration + Best-X Checkpoint

**What changed:**
1. Count-based exploration bonus: for each env step, add `0.3 / sqrt(visit_count)` to the reward stored in the replay buffer, where visit_count counts how many times this x_bucket (x_pos // 50) has been visited. Counts persist across episodes but NOT across the buffer reset.
2. Best-x mid-episode checkpoint (same as Exp151).

**Why:** The agent gets stuck around x=1234. Count-based exploration (also called pseudo-count or sqrt-count bonus from Bellemare et al. 2016) provides intrinsic motivation to visit new x positions. The first time a bucket is visited, the bonus is 0.3. After 9 visits it's 0.1. After 100 visits it's 0.03. This gently pushes the agent to explore beyond its usual frontier. The bonus is small enough (max 0.3) not to overwhelm the extrinsic rewards (distance reward can be ~10+ per step). Combined with best-x checkpoint to preserve the furthest model reached.

**Result:** total_reward=635, max_x_dist=435, score=200. Count-based bonus encourages kill-farming over advancement. Agent kills enemies for score but doesn't advance far. Discarded.

## Exp160: Higher Score Bonus (0.3 -> 0.5)

**What changed:** Increased the score multiplier in DistanceReward from 0.3 to 0.5. Score comes from killing enemies, collecting coins, breaking blocks, etc.

**Why:** The best result so far (1926) had score=800 (equivalent to ~8 Goomba kills). The eval formula gives equal weight to score and x_dist. If we incentivize killing enemies more (0.5 multiplier vs 0.3), the agent might learn to kill more enemies while advancing, boosting both score and x_dist. The multiplier 0.3 was chosen earlier to avoid kill-farming, but 0.5 might strike a better balance — still advancing but killing en-route.

**Result:** total_reward=913, max_x_dist=313, score=600. Higher score bonus (0.5) again induces kill-farming: agent kills 6 enemies for 600 score but only advances to x=313. Confirms 0.3 is better balance. Discarded.

## Exp161: Faster Epsilon Decay (1500) + Best-X Checkpoint

**What changed:**
1. EPS_DECAY reduced from 3000 to 1500 — epsilon decays to near-minimum twice as fast.
2. Best-x mid-episode checkpoint.

**Why:** With EPS_DECAY=3000, epsilon at the end of training (steps_done ~8000) is still ~0.12 — the agent is 12% random even at the end. With EPS_DECAY=1500, epsilon converges to ~0.05 much faster (by steps_done ~5000), giving more exploitation time in phase 2. The hypothesis is that faster exploitation lets the agent apply its learned policy more consistently in the critical second half, potentially pushing further right. Combined with best-x checkpoint to preserve the furthest model.

**Result:** total_reward=534, max_x_dist=434, score=100. Faster decay reduces phase-1 exploration too much, agent exploits a poorly-formed policy. EPS_DECAY=3000 remains better. Discarded.

## Exp162: GAMMA=0.97 (Shorter Horizon) + Best-X Checkpoint

**What changed:**
1. GAMMA reduced from 0.99 to 0.97 — effective horizon ~33 steps (~1.6 seconds of game time vs ~5 seconds at 0.99).
2. Best-x mid-episode checkpoint.

**Why:** GAMMA=0.99 has a long effective horizon. For precise local navigation (jumping over a specific enemy at x=1234), a shorter horizon might be more appropriate — the agent needs to focus on the next 1-2 seconds of consequences, not 5 seconds ahead. Shorter GAMMA also reduces Q-value overestimation (smaller values = more stable gradients) and can speed up convergence. Combined with best-x checkpoint.

**Result:** total_reward=594, max_x_dist=594, score=0. Shorter horizon (GAMMA=0.97) limits long-term planning needed for multi-step navigation. Below 1234. Discarded.

## Exp163: Prioritized Experience Replay (PER) + Best-X Checkpoint

**What changed:**
1. Replaced uniform ReplayBuffer with Prioritized Experience Replay (PER). Transitions with higher TD error are sampled more often (probability proportional to |TD_error|^0.6). Importance sampling weights (beta=0.4) correct for the sampling bias. Priorities are updated after each batch.
2. Best-x mid-episode checkpoint.

**Why:** PER (Schaul et al. 2016) is one of the most impactful DQN improvements. Instead of sampling all experiences uniformly, it focuses on "surprising" transitions (high TD error) — situations where the model's prediction was very wrong. This improves sample efficiency significantly: the agent spends more training time on situations it doesn't understand well, which should help learn the difficult obstacle at x=1234. Combined with best-x checkpoint to preserve the model at its furthest point.

**Result:** total_reward=1079, max_x_dist=679, score=400 (4 kills). PER focuses training on difficult transitions but this biases toward kill situations over distance. Good score but limited x advancement. Below 1926. Discarded.

## Exp164: No Buffer Reset + Best-X Checkpoint (Second Half Only)

**What changed:** Two changes from the baseline:
1. Removed the buffer reset at t=150s entirely — the buffer accumulates all 300 seconds of experiences (no reset).
2. Save best-x checkpoint in the second half (t > 150s) when a new x_pos record is hit.

**Why:** The buffer reset (Exp106, currently part of baseline) was the key breakthrough early on. But what if removing it combined with the best-x checkpoint does better? With no reset, the buffer contains diverse experiences from the entire training run — both exploratory phase-1 and exploitative phase-2 transitions. This diversity might produce a model that's more robust (trained on a wider range of situations). The best-x checkpoint in the second half ensures we save the model at its furthest reach once the policy has matured. This ablates whether the "clean buffer" benefit of the reset outweighs the "diverse buffer" benefit of no reset.

**Result:** total_reward=306, max_x_dist=306, score=0. Removing buffer reset is catastrophic — the random phase-1 experiences dilute learning so much the model barely advances. Buffer reset is essential. Discarded.

## Exp165: NoisyNet (Parametric Noise in FC Layers) + Best-X Checkpoint

**What changed:**
1. Added `NoisyLinear` layers to replace both FC layers in DQN. NoisyLinear adds learnable Gaussian noise to weights: `w = w_mu + sigma * eps` where eps ~ N(0,1). During training, noise provides state-dependent exploration. During evaluation (model.eval()), noise is disabled (uses only w_mu).
2. Removed epsilon-greedy entirely — the network's noise provides exploration instead.
3. Best-x mid-episode checkpoint (same as Exp151, only in phase 2).
4. Buffer reset at t=150s unchanged.

**Why:** NoisyNet (Fortunato et al. 2017) is a well-known DQN improvement that replaces epsilon-greedy with learnable parameter noise. The key advantage: noise is state-dependent and parameterized, so the agent explores more intelligently (high noise in uncertain states, low noise in confident states). Epsilon-greedy applies the same random exploration regardless of state. NoisyNet has been shown to improve performance in many Atari games. The noise is trained alongside Q-values, learning which states need more exploration. This should help the agent explore past x=1234 more effectively.

**Result:** total_reward=595, max_x_dist=595, score=0. NoisyNet without epsilon-greedy explores poorly early in training. The parametric noise alone is insufficient for phase-1 exploration. VRAM increased to 139MB. Discarded.

## Exp166: Dual Buffer Reset (t=100s, t=200s) + Best-X Checkpoint in Phase 3

**What changed:**
1. Two buffer resets instead of one: first at t=100s (1/3 budget), second at t=200s (2/3 budget).
2. Best-x checkpoint activates only in the final phase (after the second reset, t > 200s).

**Why:** Single buffer reset at t=150s gives the agent 150s of clean phase-2 training. A dual reset gives three phases of increasing quality:
- Phase 1 (t=0-100s): pure exploration with high epsilon, random-ish experiences
- Phase 2 (t=100-200s): better policy filling a clean buffer
- Phase 3 (t=200-300s): best policy filling the cleanest buffer yet

The second reset ensures phase 3 has only the highest-quality experiences. Best-x checkpoint only in phase 3 means we save the model at its absolute best (trained on the cleanest data). The tradeoff: each phase is shorter (100s vs 150s), and phase 3 has only 100s to fill the buffer before training ends. This tests whether quality over quantity wins for the buffer.

**Result:** total_reward=434, max_x_dist=434, score=0. Each 100s phase is too short. Phase-3 has insufficient time to accumulate enough quality experience to train a good policy. Single reset at t=150s remains optimal. Discarded.

## Exp167: TARGET_UPDATE=200 + Best-X Checkpoint

**What changed:**
1. TARGET_UPDATE reduced from 500 to 200 — target network synced every 200 steps instead of 500.
2. Best-x mid-episode checkpoint.

**Why:** More frequent target network updates means the Bellman targets are more current. In phase 2 where the policy is rapidly improving (clean buffer, mature policy), updating targets every 200 steps instead of 500 gives the Q-values a more accurate bootstrap target. This could help the agent converge faster to the correct Q-values for states near x=1234, potentially allowing it to push further. The risk is instability (oscillations due to targets changing too fast), but with the clean phase-2 buffer, this should be manageable.

**Result:** total_reward=314, max_x_dist=314, score=0. More frequent target sync (200) causes instability — targets change too fast for stable Q-value learning. TARGET_UPDATE=500 confirmed better. Discarded.

## Exp168: BATCH_SIZE=256 + Best-X Checkpoint

**What changed:**
1. BATCH_SIZE increased from 128 to 256.
2. Best-x mid-episode checkpoint.

**Why:** Larger batch = more stable gradient estimates (less variance per update). With 256 samples per batch, the Q-value targets are computed over twice as many diverse experiences, reducing the noise in each gradient step. This is especially beneficial in phase 2 where the buffer is clean but relatively sparse (fewer experiences than phase 1). The cost: fewer gradient updates per unit time (each batch is 2x slower, so fewer updates total). But the quality of each update should be higher.

**Result:** total_reward=914, max_x_dist=314, score=600 (6 kills). Larger batch induces kill-farming, VRAM 173MB. BATCH_SIZE=128 is better. Discarded.

## Exp169: EMA Weights Checkpoint at Best X

**What changed:**
1. Maintain an EMA (Exponential Moving Average) model: after each gradient step, update `ema_model` as `ema_p = 0.995 * ema_p + 0.005 * net_p`. The EMA model shadows the training model with slow decay.
2. Save the EMA model weights at best x_pos (phase 2 only), not the instantaneous training weights.
3. Buffer reset at t=150s unchanged.

**Why:** The training model weights fluctuate with each gradient step — sometimes good, sometimes bad. EMA smooths out these oscillations: the EMA model represents a running average of recent policies, not just the current one. This is similar to Polyak averaging used in TD3 and other modern RL methods. The EMA policy is typically more stable and generalizes better. By saving the EMA model when x_pos is best, we get a smoother version of the best policy. With EMA_DECAY=0.995, the EMA model tracks the last ~200 gradient steps.

**Result:** total_reward=307, max_x_dist=307, score=0. EMA model lags too far behind training model — when training agent reaches x=1234 in phase 2, the EMA model is still at phase-1 level (EMA_DECAY=0.995 is too slow). Discarded.

## Exp170: RIGHT_ONLY Action Set (5 Actions) + Best-X Checkpoint

**What changed:**
1. Changed action space from PRO_MOVEMENT (12 actions) to RIGHT_ONLY (5 actions): NOOP, right, right+A, right+B, right+A+B.
2. Best-x mid-episode checkpoint.

**Why:** Fewer Q-values to estimate = faster convergence. With 12 actions, the network must learn Q-values for actions like left, left+A, etc. which are rarely useful for maximizing rightward progress. With 5 focused right-only actions, every action in the policy is directly relevant to the goal. The DQN output layer goes from 12 to 5 units, slightly reducing computation. More importantly, the agent can't get confused by learning to go left (which sometimes happens with PRO_MOVEMENT, causing backward movement). Note: both train and eval use RIGHT_ONLY since evaluate.py imports make_env from train.py. This ablates whether PRO_MOVEMENT's jump/attack flexibility is worth the larger action space.

**Result:** total_reward=998, max_x_dist=898, score=100. RIGHT_ONLY removes useful actions (jump combos, attack-while-running). PRO_MOVEMENT confirmed superior. Discarded.

## Exp171: GAMMA=0.999 (Very Long Horizon) + Best-X Checkpoint

**What changed:**
1. GAMMA increased from 0.99 to 0.999 — effective horizon ~1000 steps (~50 seconds of game time at skip=3).
2. Best-x mid-episode checkpoint.

**Why:** GAMMA=0.99 has a ~100 step horizon (~5 seconds). GAMMA=0.999 looks 1000 steps ahead (~50 seconds) — essentially the entire level can be within the planning horizon. With such a long horizon, the Q-value for "running right" accounts for all rewards up to the end of the level, not just the next 5 seconds. This extreme foresight might help the agent value actions that lead to flag completion over local optimizations. The risk: Q-values become very large (accumulated over 1000 steps) and training might be less stable. Gradient clipping at 10.0 should help. GAMMA=0.999 is unusual but some environments benefit from very long horizons.

**Result:** total_reward=696, max_x_dist=696. Very long horizon (GAMMA=0.999) causes large Q-value magnitudes that add instability. Worse than best-x alone (1234). Discarded.

## Exp172: dist_mult=4.0 + Best-X Checkpoint

**What changed:**
1. Distance reward multiplier increased from 3.0 to 4.0.
2. Best-x mid-episode checkpoint.

**Why:** The distance reward (x_pos - curr_x) * dist_mult is the primary learning signal. At 3.0, each pixel of rightward progress gives 3 reward units. At 4.0, it gives 4 units. A stronger distance signal might help the agent more decisively prefer rightward actions over neutral/leftward ones, especially in the critical section around x=1234. This hasn't been tested with the best-x checkpoint approach — the 3.0 was tuned for the final model approach, and the checkpoint approach might benefit from a different dist_mult.

**Result:** total_reward=434, max_x_dist=434, score=0. Stronger distance reward (4.0) doesn't help vs 3.0. The higher reward magnitude may add instability. Discarded.

## Exp173: Consistent 3-Episode Checkpoint + Fallback Best-X

**What changed:**
1. After each episode in phase 2, track episode_max_x (max x reached in that episode).
2. Fallback: save model whenever single-episode best x record is set.
3. Primary: if 3 consecutive episodes ALL reach a new best minimum x (min(last_3_episodes) > previous_best), save the model. This overrides the fallback checkpoint.

**Why:** Exp151 (best-x mid-episode) saved the model when the training agent touched a new max x pixel. But that moment might be a fluke — the agent stumbles into x=1234 once but the Q-values don't reliably produce that. A consistent 3-episode checkpoint requires the agent to reach high x RELIABLY (3 in a row). If min(last_3) > best, the model has proven it can navigate to that x consistently. This should produce a more robust checkpoint than a single-episode peak. Includes the mid-episode fallback as backup.

**Result:** total_reward=202, max_x_dist=202, score=0. Consistent 3-episode requirement is too restrictive. The model that reliably reaches x=800 beats the model that reaches x=1234 once. Fallback also underperforms. Discarded.

## Exp174: 64x64 Frames + Best-X Checkpoint

**What changed:**
1. Reduced frame size from 84x84 to 64x64. This changes the conv output from 3136 to 1024 (64*4*4), so the FC layer uses nn.Linear(1024, 512).
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** Exp143 tried 64x64 without the best-x checkpoint and got total_reward=823 (x=723). With the best-x checkpoint, the combination might do better. Smaller frames mean:
- Fewer parameters (1024 vs 3136 conv features)
- Faster inference and more training steps in the time budget
- Slightly less visual detail (risk of missing fine-grained obstacles)
Both train and eval use the same 64x64 frames (eval imports PreprocessFrame from train.py). The model files are smaller and compatible between train/eval.

**Result:** total_reward=922, max_x_dist=722, score=200. 64x64 loses fine-grained visual detail vs 84x84; the checkpoint catches a decent run but the lower resolution limits Q-value precision. Discarded.

## Exp175: Adam Optimizer + Best-X Checkpoint

**What changed:**
1. Replaced RMSprop(lr=2.5e-4, alpha=0.95, eps=0.01) with Adam(lr=2.5e-4).
2. Best-x mid-episode checkpoint (phase 2 only): saves model whenever training agent sets a new x_pos record.

**Why:** RMSprop has been used throughout all experiments. Adam is the most popular optimizer for DQN variants and uses adaptive per-parameter learning rates with momentum (first moment). It may navigate the reward landscape differently and reach further x positions. The lr=2.5e-4 matches what was used with RMSprop. Adam is known to converge faster in some settings. Combined with the best-x checkpoint to capture peak performance moments.

**Result:** total_reward=315, max_x_dist=315, score=0. Adam optimizer significantly underperforms RMSprop for this DQN setup. RMSprop's per-element normalization is better suited to the noisy, non-stationary gradients in RL. Discarded.

## Exp176: EPS_END=0.02 + Best-X Checkpoint

**What changed:**
1. Reduced EPS_END from 0.05 to 0.02 (2% minimum random exploration instead of 5%).
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** Lower epsilon end means the agent exploits the learned policy more aggressively in late training. With 5% random actions, 1 in 20 actions is random even at max exploitation — this can knock the agent off a good path at critical moments. Reducing to 2% gives more deterministic exploitation of the Q-values. Combined with best-x checkpoint to save the peak performance model.

**Result:** total_reward=822, max_x_dist=722, score=100. More exploitation (EPS_END=0.02) doesn't help vs 0.05. The agent needs some randomness to escape local traps. Discarded.

## Exp177: EPS_DECAY=1500 (Faster Decay) + Best-X Checkpoint

**What changed:**
1. Reduced EPS_DECAY from 3000 to 1500 (epsilon reaches near-minimum in half as many steps).
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** With EPS_DECAY=3000, the agent spends many early steps being mostly random. Faster decay means the agent transitions to exploitation sooner, giving more training steps in near-greedy mode. By the time phase 2 starts (t=150s), the agent would be deeper into exploitation mode and may reach further x positions with the best-x checkpoint.

**Result:** total_reward=822, max_x_dist=722, score=100. Faster epsilon decay doesn't help. The agent needs sufficient random exploration in early training to fill the buffer with diverse experiences. Discarded.

## Exp178: Phase-2 LR Decay (2.5e-4 → 5e-5) + Best-X Checkpoint

**What changed:**
1. At buffer reset (t=150s), reduce optimizer LR from 2.5e-4 to 5e-5.
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** Phase 1 uses high LR (2.5e-4) for fast initial learning and exploration. At the midpoint, the policy has developed a base understanding of the environment. Phase 2 should refine this policy with a lower LR to avoid catastrophic forgetting and make finer Q-value adjustments. This two-phase LR schedule mimics learning rate annealing used in many supervised learning tasks. Combined with best-x checkpoint to save peak models.

**Result:** total_reward=711, max_x_dist=711, score=0. LR decay to 5e-5 in phase 2 slows learning too much — the agent stops improving before it reaches further. Discarded.

## Exp179: Wider Conv (64/128/64 channels) + Best-X Checkpoint

**What changed:**
1. Increased conv channel widths from 32/64/64 to 64/128/64. The first layer goes from 32 to 64 filters, the second from 64 to 128 filters. Third layer stays at 64.
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** The standard DQN architecture (32/64/64) was designed for Atari in 2013 with limited GPU memory. Modern GPUs have much more VRAM. Wider conv layers can capture more feature types simultaneously — different enemy patterns, block types, gap signatures. The FC stays the same (3136→512) since the conv output size (7×7×64=3136) is unchanged. More filters may improve the agent's ability to distinguish visually similar but strategically different game states.

**Result:** total_reward=302, max_x_dist=302, score=0. Wider conv (more parameters) slows convergence within the 5-minute budget. The original 32/64/64 architecture is appropriately sized for this task. Discarded.

## Exp180: Remove Score Bonus + Best-X Checkpoint

**What changed:**
1. Removed the score bonus from DistanceReward: `(score - prev_score) * 0.3` was deleted.
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** The score bonus of 0.3 per point rewards killing enemies and collecting coins. This can teach the agent to farm kills instead of advancing rightward. Without the score bonus, the agent has a purer incentive: distance reward (3x per pixel right) + flag bonus (1000) + step penalty (-0.1). This should push the agent to move right without being distracted by enemy encounters.

**Result:** total_reward=314, max_x_dist=314, score=0. Removing score bonus hurts — the score bonus provides useful additional signal for the agent to associate enemy killing (which often requires jumping over gaps/obstacles) with good behavior. Discarded.

## Exp181: Reward Clipping [-1, 1] + Best-X Checkpoint

**What changed:**
1. Added `reward = np.clip(reward, -1.0, 1.0)` in DistanceReward.step(), applied after computing all reward components.
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** The original Atari DQN paper (Mnih et al. 2015) clips rewards to [-1, 1] to stabilize training across games with different reward scales. Our distance reward (3.0 per pixel) and flag reward (1000) create very large reward magnitudes that can destabilize Q-value estimates. Clipping keeps all rewards in the same scale, which can reduce gradient variance and improve learning stability. The flag bonus still becomes 1.0 (positive clip), and death penalties stay at -1.0.

**Result:** total_reward=635, max_x_dist=435, score=200. Reward clipping removes the distinction between small and large rewards — the agent can't tell a 1-pixel move from a flag grab. Distance learning degrades significantly. Discarded.

## Exp182: Episode-End Best-X Checkpoint

**What changed:**
1. Track max x reached during each episode (episode_max_x).
2. At end of each episode in phase 2: if episode_max_x > best_episode_x, save model.
3. No mid-episode step-level checkpointing.

**Why:** The step-level checkpoint (Exp151) saves the model at the exact moment the agent is at x=1234, mid-episode. But this might capture a transient policy state — perhaps the agent stumbled into high x via random actions or a lucky sequence, and the Q-values don't reliably reproduce this. The episode-end checkpoint saves after the full episode has completed and the policy has updated from all experiences in that episode. This should capture more stable, reproducible policies. If an agent reaches x=1234 in an episode and then the policy updates make it even better, we save the post-update policy.

**Result:** total_reward=315, max_x_dist=315, score=0. Episode-end checkpoint underperforms step-level (Exp151=1234). The step-level checkpoint captures the best policy during the episode, while episode-end waits until after additional steps have updated the weights away from the peak state. Discarded.

## Exp183: Epsilon Reset in Phase 2 + Best-X Checkpoint

**What changed:**
1. After buffer reset (t=150s): reset `steps_done = 4011` so epsilon recalculates to ~0.3 (moderate exploration).
   - Formula: 0.05 + 0.95 * exp(-4011/3000) ≈ 0.30
   - Phase 2 then decays normally from eps=0.3 to eps=0.05 over remaining time.
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** Phase 1 explores heavily (eps decays 1.0→~0.05 over ~9000 steps). By the buffer reset point, the agent is nearly fully exploitative. But the buffer was just cleared — the agent should explore with its now-smarter policy to fill it with diverse, high-quality transitions. Resetting epsilon to 0.3 gives moderate exploration: 30% random, 70% policy-guided. The smarter agent explores more purposefully than a purely random agent, potentially discovering paths the original greedy exploration missed.

**Result:** total_reward=535, max_x_dist=435, score=100. Re-exploration in phase 2 hurts: the agent needs more exploitation, not more exploration, in the second half. The smarter policy benefits from greedy use of learned Q-values rather than random re-exploration. Discarded.

## Exp184: Two Buffer Resets (t=100s, t=200s) + Best-X Checkpoint

**What changed:**
1. Changed single buffer reset at t=150s to two resets: t=100s and t=200s (1/3 and 2/3 of TIME_BUDGET).
2. Best-x checkpoint only activates in the final phase (after t=200s).

**Why:** A three-phase schedule:
- Phase 1 (0-100s): Pure exploration, random-then-greedy, fill buffer with diverse experiences
- Phase 2 (100-200s): First reset — smarter policy re-fills buffer with better experiences
- Phase 3 (200-300s): Second reset — now-expert policy fills buffer with high-quality experiences; best-x checkpoint captures peak

The logic: each reset lets the current policy start fresh without contamination from earlier, worse-policy experiences. Two resets means the final buffer is filled only with the best policy's transitions. The checkpoint in phase 3 captures the peak of this final, best phase.

**Result:** total_reward=694, max_x_dist=594, score=100. Two buffer resets reduce training time per phase too much. Each phase only has 100s, leaving insufficient time to learn from each clean buffer. Single reset at t=150 remains better. Discarded.

## Exp185: Soft Buffer Reset (Keep Top 5000) + Best-X Checkpoint

**What changed:**
1. At buffer reset time (t=150s): instead of clearing the buffer entirely, keep the top 5000 highest-reward transitions (sorted by immediate reward). Discard the remaining ~45000 lower-reward transitions.
2. Best-x mid-episode checkpoint (phase 2 only).

**Why:** A hard reset throws away ALL phase-1 experiences, including rare high-reward transitions (reaching high x positions, flag bonuses). These rare experiences are valuable seeds for phase-2 learning. By keeping the top 5000 highest-reward transitions, phase 2 starts with useful "anchor" experiences that demonstrate high-reward behaviors, while still getting fresh phase-2 experiences to fill the remaining 45000 slots. The 5000 carryover is 10% of capacity — substantial enough to be useful, small enough not to dominate phase-2 learning.

**Result:**
