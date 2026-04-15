import time
import subprocess
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import shimmy
import gymnasium as gym
import torch
import torch.nn as nn
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

# PPO hyperparameters
N_WORKERS = 8
N_STEPS = 128           # env steps per worker before each PPO update
N_EPOCHS = 4            # PPO gradient epochs per rollout
MINI_BATCH_SIZE = 256   # mini-batch size within each epoch
LR = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
ENT_COEF = 0.01         # entropy bonus coefficient (encourages exploration)
VF_COEF = 0.5           # value loss coefficient
MAX_GRAD_NORM = 0.5
GREEDY_CHECK_ROLLOUTS = 8   # run greedy eval every N rollouts

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
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def wrap_env(raw_env):
    env = FrameSkip(raw_env, skip=3)
    env = DistanceReward(env)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)
    return env


def greedy_eval_x(model, eval_env):
    """Returns (max_x, final_score) for a greedy (deterministic) episode."""
    model.eval()
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
    model.train()
    return max_x, final_score


def train():
    envs = [wrap_env(make_env(render=False)) for _ in range(N_WORKERS)]
    eval_env = wrap_env(make_env(render=False))
    states = [env.reset()[0] for env in envs]

    n_actions = envs[0].action_space.n
    model = PolicyModel(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # Warm start: load CNN weights from the saved DQN checkpoint.
    # The DQN used 'advantage.*' for action scores and 'value.*' for state value.
    # We map those to PPO's 'policy.*' and 'value_head.*'. Only the CNN layers
    # transfer cleanly (same architecture). The FC policy head gets fresh orthogonal
    # init (set in model.py), so we only copy conv.* from the checkpoint.
    model_path = "MODELS/model.pt"
    model_saved = False
    if os.path.exists(model_path):
        try:
            saved = torch.load(model_path, map_location=device)
            if any(k.startswith('policy.') for k in saved):
                # PPO format checkpoint: load all weights and continue training
                model.load_state_dict(saved)
                print(f"Warm start: loaded full PPO model from {model_path}")
            else:
                # DQN format checkpoint: copy only conv layers (FC heads have wrong scale)
                model_dict = model.state_dict()
                conv_weights = {k: v for k, v in saved.items() if k.startswith('conv.')}
                model_dict.update(conv_weights)
                model.load_state_dict(model_dict)
                print(f"Warm start: loaded CNN from DQN checkpoint, fresh policy/value heads")
            # NOTE: do NOT set model_saved=True here — loading != saving a PPO model.
            # The finally block must always write a valid PPO-format model.pt.
        except Exception as e:
            print(f"Warm start failed ({e}), starting fresh")

    best_greedy_x, best_greedy_score = greedy_eval_x(model, eval_env)
    best_combined = best_greedy_x + best_greedy_score
    print(f"Baseline: x={best_greedy_x} score={best_greedy_score} combined={best_combined}")

    start_time = time.time()
    rollout_count = 0
    total_ep_rewards = []
    ep_rewards = [0.0] * N_WORKERS
    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    try:
        while time.time() - start_time < TIME_BUDGET:
            # ── Collect rollout ──────────────────────────────────────────────────
            mb_states = []
            mb_actions = []
            mb_log_probs = []
            mb_values = []
            mb_rewards = []
            mb_dones = []

            for step in range(N_STEPS):
                states_t = torch.FloatTensor(
                    np.array(states, dtype=np.float32) / 255.0
                ).to(device)

                with torch.no_grad():
                    logits, values = model.full_forward(states_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                mb_states.append(states_t)
                mb_actions.append(actions)
                mb_log_probs.append(log_probs)
                mb_values.append(values.squeeze(-1))

                step_rewards = []
                step_dones = []
                next_states = []

                for i in range(N_WORKERS):
                    ns, r, term, trunc, info = envs[i].step(actions[i].item())
                    done = term or trunc
                    ep_rewards[i] += r
                    step_rewards.append(r)
                    step_dones.append(float(done))

                    ct = info.get('time', 0) + info.get('score', 0)
                    if ct > best_total_reward:
                        best_total_reward = ct
                        best_score = info.get('score', 0)
                        best_time = info.get('time', 0)

                    if done:
                        total_ep_rewards.append(ep_rewards[i])
                        ep_rewards[i] = 0.0
                        # When a worker completes the level, immediately checkpoint —
                        # the stochastic policy is at its peak; capture it for greedy eval.
                        if info.get('flag_get', False):
                            gx, gs = greedy_eval_x(model, eval_env)
                            combined = gx + gs
                            if combined > best_combined:
                                best_combined = combined
                                best_greedy_x, best_greedy_score = gx, gs
                                model_saved = True
                                os.makedirs("MODELS", exist_ok=True)
                                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                                print(f"  FLAG GET checkpoint: greedy x={gx} score={gs} combined={combined} (saved)")
                            else:
                                print(f"  FLAG GET (worker {i}): greedy x={gx} score={gs} combined={combined} (best={best_combined})")
                        ns = envs[i].reset()[0]
                    next_states.append(ns)

                mb_rewards.append(torch.FloatTensor(step_rewards).to(device))
                mb_dones.append(torch.FloatTensor(step_dones).to(device))
                states = next_states

            # ── Compute GAE returns ──────────────────────────────────────────────
            states_t = torch.FloatTensor(
                np.array(states, dtype=np.float32) / 255.0
            ).to(device)
            with torch.no_grad():
                _, last_values = model.full_forward(states_t)
                last_values = last_values.squeeze(-1)

            returns_list = []
            gae = torch.zeros(N_WORKERS, device=device)
            for t in reversed(range(N_STEPS)):
                next_val = last_values if t == N_STEPS - 1 else mb_values[t + 1]
                delta = mb_rewards[t] + GAMMA * next_val * (1.0 - mb_dones[t]) - mb_values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - mb_dones[t]) * gae
                returns_list.insert(0, gae + mb_values[t])

            # Flatten to (N_STEPS * N_WORKERS,)
            all_states = torch.cat(mb_states, dim=0)
            all_actions = torch.cat(mb_actions, dim=0)
            all_log_probs = torch.cat(mb_log_probs, dim=0).detach()
            all_values = torch.cat(mb_values, dim=0).detach()
            all_returns = torch.stack(returns_list, dim=0).view(-1).detach()
            all_advantages = all_returns - all_values
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

            # ── PPO update ───────────────────────────────────────────────────────
            batch_size = N_STEPS * N_WORKERS
            indices = np.arange(batch_size)

            for _ in range(N_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, batch_size, MINI_BATCH_SIZE):
                    mb_idx = indices[start:start + MINI_BATCH_SIZE]
                    s = all_states[mb_idx]
                    a = all_actions[mb_idx]
                    old_lp = all_log_probs[mb_idx]
                    adv = all_advantages[mb_idx]
                    ret = all_returns[mb_idx]

                    logits, values = model.full_forward(s)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_lp = dist.log_prob(a)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(values.squeeze(-1), ret)
                    loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

            rollout_count += 1

            # ── Greedy checkpoint ────────────────────────────────────────────────
            if rollout_count % GREEDY_CHECK_ROLLOUTS == 0:
                gx, gs = greedy_eval_x(model, eval_env)
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
        eval_env.close()
        os.makedirs("MODELS", exist_ok=True)
        if not model_saved:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")

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
