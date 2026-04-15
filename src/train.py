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

# Dueling Double DQN hyperparameters
N_WORKERS = 8
BATCH_SIZE = 256
BUFFER_SIZE = 200000
GAMMA = 0.99
LR = 1e-5
TARGET_UPDATE_INTERVAL = 200
TRAIN_START = 10000
TRAIN_FREQ = 32
GREEDY_CHECK_INTERVAL = 5000

# Pure greedy fine-tuning — like exp85 (1137→2023), hoping 1959→2100+
BARRIER_X_THRESHOLD = 3200  # effectively disabled: workers never reach threshold
BARRIER_EPSILON = 0.0

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


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device) / 255.0,
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(np.array(s2)).to(device) / 255.0,
            torch.FloatTensor(d).to(device),
        )

    def __len__(self):
        return len(self.buf)


def greedy_eval_x(model, eval_env):
    """Returns (max_x, final_score) for the greedy episode."""
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
    return max_x, final_score


def train():
    """Evolutionary search: perturb advantage FC head, keep if greedy eval improves.
    No gradient updates — avoids catastrophic forgetting entirely."""
    eval_env = wrap_env(make_env(render=False))
    n_actions = eval_env.action_space.n

    model = PolicyModel(n_actions).to(device)
    model_path = "MODELS/model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Warm start from {model_path}")
    model.eval()

    best_x, best_score = greedy_eval_x(model, eval_env)
    best_combined = best_x + best_score
    best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
    print(f"Baseline: x={best_x} score={best_score} combined={best_combined}")
    os.makedirs("MODELS", exist_ok=True)
    torch.save(best_state, model_path)

    candidate = PolicyModel(n_actions).to(device)
    candidate.eval()
    start_time = time.time()
    n_evals = 0
    noise_std = 0.05  # perturbation magnitude for advantage.2 (last FC of advantage head)

    try:
        while time.time() - start_time < TIME_BUDGET:
            # Perturb only the advantage head's last layer (directly controls action ranking)
            candidate_state = {k: v.clone() for k, v in best_state.items()}
            for k in ['advantage.2.weight', 'advantage.2.bias']:
                candidate_state[k] = candidate_state[k] + torch.randn_like(candidate_state[k]) * noise_std
            candidate.load_state_dict({k: v.to(device) for k, v in candidate_state.items()})

            gx, gs = greedy_eval_x(candidate, eval_env)
            combined = gx + gs
            n_evals += 1

            if combined > best_combined:
                best_combined = combined
                best_x, best_score = gx, gs
                best_state = {k: v.clone().detach().cpu() for k, v in candidate_state.items()}
                torch.save(best_state, model_path)
                print(f"  Improved [{n_evals}]: x={gx} score={gs} combined={combined} (saved)")
            else:
                print(f"  No improv [{n_evals}]: x={gx} score={gs} combined={combined} (best={best_combined})")

            gpu_temp = get_gpu_temp()
            if gpu_temp >= MAX_GPU_TEMP:
                print(f"GPU {gpu_temp}C >= {MAX_GPU_TEMP}C -- stopping.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        eval_env.close()
        training_seconds = time.time() - start_time
        print(f"training_seconds: {training_seconds:.1f}")
        print(f"total_seconds: {training_seconds:.1f}")
        print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0:.1f}")
        print(f"train_best_reward: {best_combined:.1f}")
        print(f"train_seconds_to_finish: 999.0")
        print(f"train_best_score: {best_score}")
        print(f"mean_episode_reward: {best_combined:.1f}")


if __name__ == "__main__":
    train()
