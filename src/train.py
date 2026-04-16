import time
import subprocess
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import shimmy
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import os
import random
from collections import deque
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import TIME_BUDGET, MAX_EPISODE_STEPS, PRO_MOVEMENT
from src.model import PolicyModel

# Distillation + BC hyperparameters
# Strategy: protect early-game (x<600) via distillation from a frozen reference model,
#           teach barrier+beyond (x>=900) via BC from any episode reaching x>=900.
N_WORKERS = 8
N_STEPS = 128
LR = 1e-4               # optimizer LR (policy[-1] only)
MAX_GRAD_NORM = 0.5
GREEDY_CHECK_ROLLOUTS = 8

# Epsilon-greedy exploration for workers
EPS_EXPLORE = 0.40      # 40% random actions — more x>900 episodes than exp138 (was 30%)

# Distillation protects x < BEYOND_THRESHOLD only (was 900 — that blocked x=899 learning)
BEYOND_THRESHOLD = 600

# BC collects from any episode where max_x >= BC_X_THRESHOLD (was: flag_get=True only)
# Workers regularly reach x>900 with EPS=0.40, so BC data is plentiful
BC_X_THRESHOLD = 900

# Distillation: force model to match reference (frozen original) on early-game states (x < BEYOND_THRESHOLD)
DISTILL_COEF = 5.0
DISTILL_BUFFER_MAX = 5000

# Behavioral Cloning from episodes reaching x >= BC_X_THRESHOLD
BC_COEF = 3.0
BC_UPDATES_PER_ROLLOUT = 8
BC_X_WINDOW = 2000      # keep samples from x in [BC_X_THRESHOLD, BC_X_THRESHOLD + 2000]
BC_BATCH_SIZE = 128
BC_BUFFER_MAX = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

MAX_GPU_TEMP = 85


def get_gpu_temp():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            timeout=3
        )
        return int(out.decode().strip())
    except Exception:
        return 0


def make_env(render=False):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = env.env
    env = JoypadSpace(env, PRO_MOVEMENT)
    env = shimmy.GymV21CompatibilityV0(env=env, render_mode='human' if render else None)
    return env


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return obs[:, :, None]


class EnsureChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(list(self.frames), axis=0)


class DistanceReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.curr_x = 0

    def reset(self, **kwargs):
        self.curr_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = info.get('x_pos', 0)
        reward += (x_pos - self.curr_x) * 2.0
        self.curr_x = x_pos
        reward -= 0.1
        if info.get('flag_get', False):
            reward += 1000.0
        return obs, reward, terminated, truncated, info


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        any_flag_get = False
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if info.get('flag_get', False):
                any_flag_get = True
            if terminated or truncated:
                break
        if any_flag_get:
            info['flag_get'] = True
        return obs, total_reward, terminated, truncated, info


def wrap_env(raw_env):
    env = FrameSkip(raw_env, skip=3)
    env = DistanceReward(env)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)
    return env


def greedy_eval_x(model, _unused_eval_env=None):
    """Returns (max_x, final_score) for a greedy (deterministic) episode.

    Creates a fresh environment each call to avoid NES emulator state corruption
    that occurs when reusing an env across multiple greedy evaluations.
    """
    eval_env = wrap_env(make_env(render=False))
    model.eval()
    try:
        state, _ = eval_env.reset()
        max_x = 0
        final_score = 0
        for _ in range(MAX_EPISODE_STEPS):
            st = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                action = model(st).max(1)[1].item()
            state, _, terminated, truncated, info = eval_env.step(action)
            max_x = max(max_x, info.get('x_pos', 0))
            final_score = info.get('score', 0)
            if terminated or truncated:
                break
    finally:
        eval_env.close()
        model.train()
    return max_x, final_score


def train():
    envs = [wrap_env(make_env(render=False)) for _ in range(N_WORKERS)]
    states = [env.reset()[0] for env in envs]

    n_actions = envs[0].action_space.n
    model = PolicyModel(n_actions).to(device)
    # Exp 139: only train policy[-1] (the final Linear 512→n_actions layer).
    # Distillation (x<600) counteracts early-game degradation; BC (x>=900 episodes) trains barrier+beyond.
    optimizer = optim.Adam(model.policy[-1].parameters(), lr=LR, eps=1e-5)

    model_path = "MODELS/model.pt"
    model_saved = False
    if os.path.exists(model_path):
        try:
            saved = torch.load(model_path, map_location=device)
            if any(k.startswith('policy.') for k in saved):
                model.load_state_dict(saved)
                print(f"Warm start: loaded full PPO model from {model_path}")
            else:
                model_dict = model.state_dict()
                conv_weights = {k: v for k, v in saved.items() if k.startswith('conv.')}
                model_dict.update(conv_weights)
                model.load_state_dict(model_dict)
                print(f"Warm start: loaded CNN from DQN checkpoint, fresh policy/value heads")
        except Exception as e:
            print(f"Warm start failed ({e}), starting fresh")

    # Frozen reference model: copy of the loaded model (never updated).
    # Used for distillation loss to preserve early-game (x < BEYOND_THRESHOLD) behavior.
    ref_model = PolicyModel(n_actions).to(device)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    best_greedy_x, best_greedy_score = greedy_eval_x(model)
    best_combined = best_greedy_x + best_greedy_score
    print(f"Baseline: x={best_greedy_x} score={best_greedy_score} combined={best_combined}")

    start_time = time.time()
    rollout_count = 0
    total_ep_rewards = []
    ep_rewards = [0.0] * N_WORKERS
    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    # BC buffer: (state, action) from any episode reaching x >= BC_X_THRESHOLD
    bc_states = deque(maxlen=BC_BUFFER_MAX)
    bc_actions = deque(maxlen=BC_BUFFER_MAX)
    worker_ep_states = [[] for _ in range(N_WORKERS)]
    worker_ep_actions = [[] for _ in range(N_WORKERS)]
    worker_ep_xpos = [[] for _ in range(N_WORKERS)]
    worker_flag_get = [False] * N_WORKERS
    bc_episodes_total = 0

    # Distillation buffer: states from worker steps where x < BEYOND_THRESHOLD.
    # Used to force model to match ref_model on early-game states (prevent early-game degradation).
    distill_states = deque(maxlen=DISTILL_BUFFER_MAX)
    # Track current x_pos per worker (for distill buffer and beyond-barrier routing)
    worker_current_x = [0] * N_WORKERS

    try:
        while time.time() - start_time < TIME_BUDGET:
            flag_get_this_rollout = False

            for step in range(N_STEPS):
                states_t = torch.FloatTensor(
                    np.array(states, dtype=np.float32) / 255.0
                ).to(device)

                with torch.no_grad():
                    logits = model(states_t)
                act_list = []
                for wi in range(N_WORKERS):
                    if random.random() < EPS_EXPLORE:
                        act_list.append(random.randrange(n_actions))
                    else:
                        act_list.append(logits[wi].argmax().item())
                actions = torch.tensor(act_list, device=device)

                next_states = []

                for i in range(N_WORKERS):
                    ns, r, term, trunc, info = envs[i].step(actions[i].item())
                    done = term or trunc
                    ep_rewards[i] += r
                    x_pos = info.get('x_pos', 0)
                    worker_current_x[i] = x_pos

                    # Distillation buffer: keep early-game states (before the barrier)
                    if x_pos < BEYOND_THRESHOLD:
                        distill_states.append((states_t[i].cpu() * 255).byte())

                    # Track episode for BC
                    worker_ep_states[i].append((states_t[i].cpu() * 255).byte())
                    worker_ep_actions[i].append(actions[i].cpu())
                    worker_ep_xpos[i].append(x_pos)
                    if info.get('flag_get', False):
                        worker_flag_get[i] = True

                    ct = info.get('time', 0) + info.get('score', 0)
                    if ct > best_total_reward:
                        best_total_reward = ct
                        best_score = info.get('score', 0)
                        best_time = info.get('time', 0)

                    if done:
                        total_ep_rewards.append(ep_rewards[i])
                        ep_rewards[i] = 0.0
                        # BC from any episode that passed the barrier (not just flag_get)
                        ep_max_x = max(worker_ep_xpos[i]) if worker_ep_xpos[i] else 0
                        if ep_max_x >= BC_X_THRESHOLD:
                            x_lo = BC_X_THRESHOLD
                            x_hi = BC_X_THRESHOLD + BC_X_WINDOW
                            n_added = 0
                            for s, a, xp in zip(worker_ep_states[i], worker_ep_actions[i], worker_ep_xpos[i]):
                                if x_lo <= xp <= x_hi:
                                    bc_states.append(s)
                                    bc_actions.append(a)
                                    n_added += 1
                            if n_added > 0:
                                bc_episodes_total += 1
                                print(f"  BC: ep {bc_episodes_total} added ({n_added}/{len(worker_ep_states[i])} steps, x=[{x_lo},{x_hi}], buf={len(bc_states)})")
                        if worker_flag_get[i]:
                            flag_get_this_rollout = True  # trigger extra greedy check on level completion
                        worker_ep_states[i] = []
                        worker_ep_actions[i] = []
                        worker_ep_xpos[i] = []
                        worker_flag_get[i] = False
                        worker_current_x[i] = 0
                        ns = envs[i].reset()[0]
                    next_states.append(ns)

                states = next_states

            # ── Combined Distillation + BC update ──────────────────────────────────
            # Distillation: model(early_states) must match ref_model(early_states) — prevents early-game degradation.
            # BC: model(late_states) must produce completion actions — teaches beyond-barrier behavior.
            # Only policy[-1] (Linear 512→n_actions) is updated.
            has_distill = len(distill_states) >= BC_BATCH_SIZE
            has_bc = len(bc_states) >= BC_BATCH_SIZE

            if has_distill or has_bc:
                for _ in range(BC_UPDATES_PER_ROLLOUT):
                    total_loss = torch.tensor(0.0, device=device)

                    if has_distill:
                        d_idx = random.sample(range(len(distill_states)), min(BC_BATCH_SIZE, len(distill_states)))
                        d_s = torch.stack([distill_states[i] for i in d_idx]).to(device).float() / 255.0
                        new_logits = model(d_s)
                        with torch.no_grad():
                            ref_logits = ref_model(d_s)
                            ref_probs = torch.softmax(ref_logits, dim=1)
                        new_log_probs = F.log_softmax(new_logits, dim=1)
                        # KL divergence: force model to match ref_model on early-game states
                        distill_loss = -(ref_probs * new_log_probs).sum(dim=1).mean()
                        total_loss = total_loss + DISTILL_COEF * distill_loss

                    if has_bc:
                        bc_idx = random.sample(range(len(bc_states)), BC_BATCH_SIZE)
                        bc_s = torch.stack([bc_states[i] for i in bc_idx]).to(device).float() / 255.0
                        bc_a = torch.stack([bc_actions[i] for i in bc_idx]).to(device)
                        bc_logits = model(bc_s)
                        bc_loss = nn.CrossEntropyLoss()(bc_logits, bc_a)
                        total_loss = total_loss + BC_COEF * bc_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.policy[-1].parameters(), MAX_GRAD_NORM)
                    optimizer.step()

            rollout_count += 1

            if flag_get_this_rollout:
                gx, gs = greedy_eval_x(model)
                combined = gx + gs
                if combined > best_combined:
                    best_combined = combined
                    best_greedy_x, best_greedy_score = gx, gs
                    model_saved = True
                    os.makedirs("MODELS", exist_ok=True)
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                    print(f"  FLAG GET checkpoint: greedy x={gx} score={gs} combined={combined} (saved)")
                else:
                    print(f"  FLAG GET (rollout): greedy x={gx} score={gs} combined={combined} (best={best_combined})")

            if rollout_count % GREEDY_CHECK_ROLLOUTS == 0:
                gx, gs = greedy_eval_x(model)
                combined = gx + gs
                if combined > best_combined:
                    best_combined = combined
                    best_greedy_x, best_greedy_score = gx, gs
                    model_saved = True
                    os.makedirs("MODELS", exist_ok=True)
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                    print(f"  Greedy checkpoint: x={gx} score={gs} combined={combined} (saved)")
                else:
                    print(f"  Greedy check: x={gx} score={gs} combined={combined} (best={best_combined})")

            if total_ep_rewards and rollout_count % 4 == 0:
                gpu_temp = get_gpu_temp()
                mean_r = np.mean(total_ep_rewards[-N_WORKERS:])
                env_steps = rollout_count * N_STEPS * N_WORKERS
                print(f"Rollout {rollout_count:>4} | Steps {env_steps:>7} | MeanReward: {mean_r:>7.1f} | GPU: {gpu_temp}C")
                if gpu_temp >= MAX_GPU_TEMP:
                    print(f"GPU {gpu_temp}C >= {MAX_GPU_TEMP}C -- stopping.")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        for env in envs:
            env.close()
        os.makedirs("MODELS", exist_ok=True)
        if not model_saved:
            final_gx, final_gs = greedy_eval_x(model)
            final_combined = final_gx + final_gs
            if final_combined >= best_combined:
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                print(f"Final save: x={final_gx} score={final_gs} combined={final_combined}")
            else:
                print(f"Final model degraded (x={final_gx} combined={final_combined} < best={best_combined}), skipping save")

        training_seconds = time.time() - start_time
        print(f"training_seconds: {training_seconds:.1f}")
        print(f"total_seconds: {training_seconds:.1f}")
        print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0:.1f}")

        seconds_to_finish = 400 - best_time if best_time > 0 else 999
        print(f"train_best_reward: {best_total_reward:.1f}")
        print(f"train_seconds_to_finish: {seconds_to_finish:.1f}")
        print(f"train_best_score: {best_score}")

        if total_ep_rewards:
            print(f"mean_episode_reward: {np.mean(total_ep_rewards):.1f}")


if __name__ == "__main__":
    train()
