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
GREEDY_CHECK_INTERVAL = 2000  # 2.5x more frequent: catch more lucky checkpoints

# Exp-91 style, from x=2023 backup: original selective filter, no bonus, eps=1.0 at x=2022
BARRIER_X_THRESHOLD = 2022
BARRIER_EPSILON = 1.0

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
    envs = [wrap_env(make_env(render=False)) for _ in range(N_WORKERS)]
    eval_env = wrap_env(make_env(render=False))
    states = [env.reset()[0] for env in envs]

    n_actions = envs[0].action_space.n
    model = PolicyModel(n_actions).to(device)
    target = PolicyModel(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model_path = "MODELS/model.pt"
    model_saved = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model_saved = True
            print(f"Warm start from {model_path}")
        except Exception as e:
            print(f"Warm start failed ({e}), starting fresh")
    target.load_state_dict(model.state_dict())
    target.eval()

    buffer = ReplayBuffer(BUFFER_SIZE)
    start_time = time.time()

    total_ep_rewards = []
    ep_reward = [0.0] * N_WORKERS
    update_count = 0
    env_steps = 0
    worker_x = [0.0] * N_WORKERS
    if model_saved:
        best_greedy_x, best_greedy_score = greedy_eval_x(model, eval_env)
        print(f"Warm start baseline: greedy x={best_greedy_x} score={best_greedy_score}")
    else:
        best_greedy_x = 0
        best_greedy_score = 0

    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    try:
        while time.time() - start_time < TIME_BUDGET:
            for i in range(N_WORKERS):
                epsilon = BARRIER_EPSILON if worker_x[i] >= BARRIER_X_THRESHOLD else 0.0
                if random.random() < epsilon:
                    action = random.randrange(n_actions)
                else:
                    st = torch.FloatTensor(states[i]).unsqueeze(0).to(device) / 255.0
                    with torch.no_grad():
                        action = model(st).max(1)[1].item()

                next_state, reward, terminated, truncated, info = envs[i].step(action)
                done = terminated or truncated
                ep_reward[i] += reward
                new_x = info.get('x_pos', 0)
                # Exp-91 selective filter: always add pre-barrier; at barrier, only add if crossed
                at_barrier = (worker_x[i] >= BARRIER_X_THRESHOLD)
                if not at_barrier or new_x > BARRIER_X_THRESHOLD:
                    buffer.add(states[i], action, reward, next_state, float(done))
                worker_x[i] = new_x

                current_time = info.get('time', 0)
                current_score = info.get('score', 0)
                ct = current_time + current_score
                if ct > best_total_reward:
                    best_total_reward = ct
                    best_score = current_score
                    best_time = current_time

                if done:
                    total_ep_rewards.append(ep_reward[i])
                    ep_reward[i] = 0.0
                    worker_x[i] = 0.0
                    states[i] = envs[i].reset()[0]
                else:
                    states[i] = next_state

            env_steps += N_WORKERS

            if len(buffer) >= TRAIN_START and env_steps % TRAIN_FREQ == 0:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)
                with torch.no_grad():
                    next_actions = model(s2).argmax(1)
                    next_q = target(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = r + GAMMA * next_q * (1 - d)
                current_q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.MSELoss()(current_q, target_q)

                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    update_count += 1

                if update_count % TARGET_UPDATE_INTERVAL == 0:
                    target.load_state_dict(model.state_dict())

            if env_steps % GREEDY_CHECK_INTERVAL == 0 and len(buffer) >= TRAIN_START:
                gx, gs = greedy_eval_x(model, eval_env)
                combined = gx + gs
                best_combined = best_greedy_x + best_greedy_score
                if combined > best_combined:
                    best_greedy_x, best_greedy_score = gx, gs
                    model_saved = True
                    os.makedirs("MODELS", exist_ok=True)
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                    print(f"  Greedy checkpoint: x={gx} score={gs} combined={combined} (saved)")
                else:
                    print(f"  Greedy check: x={gx} score={gs} combined={combined} (best={best_combined})")

            if total_ep_rewards and env_steps % (GREEDY_CHECK_INTERVAL // 2) == 0:
                gpu_temp = get_gpu_temp()
                mean_r = np.mean(total_ep_rewards[-N_WORKERS:])
                print(f"Steps {env_steps:>7} | MeanReward: {mean_r:>7.1f} | Updates: {update_count} | Buf: {len(buffer)} | GPU: {gpu_temp}C")
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
