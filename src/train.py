import time
import random
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
import sys
import warnings
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import TIME_BUDGET, MAX_EPISODE_STEPS, PRO_MOVEMENT
from src.model import PolicyModel

# Frozen-feature DQN hyperparameters (exp148)
# Strategy: keep conv+policy[:2] frozen (good features from exp142 x=898 model),
#           warm-start policy[-1] Q-head, train with standard DQN.
#           Frozen features prevent catastrophic forgetting of early-game behavior.
#           Only the Q-head (policy[-1]) updates via TD-learning.
MEMORY_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-5              # very conservative — preserve warm-start behavior
TARGET_UPDATE = 2000   # training steps between target network updates
TRAIN_FREQ = 4         # train every 4 env steps
EPS_START = 0.20       # moderate exploration — good warm start doesn't need full random
EPS_END = 0.02
EPS_DECAY = 40000      # decay over 40k env steps
TRAIN_START = 2000     # start training after buffer has this many transitions
GREEDY_CHECK_STEPS = 3000  # check greedy every 3000 training steps

BARRIER_X = 899
BARRIER_BONUS = 500.0

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
        self.frames = []
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(self.frames, axis=0)


class DistanceReward(gym.Wrapper):
    """Distance-based reward with barrier bonus for first crossing x=BARRIER_X."""
    def __init__(self, env, barrier_bonus=0.0):
        super().__init__(env)
        self.curr_x = 0
        self.barrier_crossed = False
        self.barrier_bonus = barrier_bonus

    def reset(self, **kwargs):
        self.curr_x = 0
        self.barrier_crossed = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = info.get('x_pos', 0)
        reward += (x_pos - self.curr_x) * 2.0
        self.curr_x = x_pos
        reward -= 0.1
        if self.barrier_bonus > 0 and x_pos > BARRIER_X and not self.barrier_crossed:
            reward += self.barrier_bonus
            self.barrier_crossed = True
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


def wrap_env(raw_env, barrier_bonus=0.0):
    env = FrameSkip(raw_env, skip=3)
    env = DistanceReward(env, barrier_bonus=barrier_bonus)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)
    return env


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def greedy_eval_x(model, _unused=None):
    """Returns (max_x, final_score) for a greedy (deterministic) episode."""
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
    env = wrap_env(make_env(render=False), barrier_bonus=BARRIER_BONUS)
    n_actions = env.action_space.n
    model = PolicyModel(n_actions).to(device)

    # Freeze conv + policy[:2] (proven feature extractor from exp142).
    # Only policy[-1] Q-head trains. beyond_head stays zero (no-op).
    for p in model.conv.parameters():
        p.requires_grad = False
    for p in model.policy[:2].parameters():
        p.requires_grad = False
    for p in model.beyond_head.parameters():
        p.requires_grad = False
    for p in model.value_head.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.policy[-1].parameters(), lr=LR)

    model_path = "MODELS/model.pt"
    model_saved = False
    if os.path.exists(model_path):
        try:
            saved = torch.load(model_path, map_location=device)
            if any(k.startswith('policy.') for k in saved):
                model.load_state_dict(saved, strict=False)
                print(f"Warm start: loaded PPO model from {model_path} (frozen features + warm Q-head)")
            else:
                model_dict = model.state_dict()
                conv_weights = {k: v for k, v in saved.items() if k.startswith('conv.')}
                model_dict.update(conv_weights)
                model.load_state_dict(model_dict)
                print(f"Warm start: loaded CNN only, fresh Q-head")
        except Exception as e:
            print(f"Warm start failed ({e}), starting fresh")

    # Target network: copy entire model, but only policy[-1] will be updated
    target_model = PolicyModel(n_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    best_greedy_x, best_greedy_score = greedy_eval_x(model)
    best_combined = best_greedy_x + best_greedy_score
    print(f"Baseline: x={best_greedy_x} score={best_greedy_score} combined={best_combined}")

    buffer = ReplayBuffer(MEMORY_SIZE)
    start_time = time.time()
    env_steps = 0
    train_steps = 0
    total_ep_rewards = []
    ep_reward = 0.0
    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    state, _ = env.reset()

    try:
        while time.time() - start_time < TIME_BUDGET:
            # Epsilon-greedy
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-env_steps / EPS_DECAY)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                st = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    action = model(st).max(1)[1].item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            env_steps += 1

            ct = info.get('time', 0) + info.get('score', 0)
            if ct > best_total_reward:
                best_total_reward = ct
                best_score = info.get('score', 0)
                best_time = info.get('time', 0)

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            if done:
                total_ep_rewards.append(ep_reward)
                ep_reward = 0.0
                state, _ = env.reset()

            # ── DQN update ────────────────────────────────────────────────────────
            if len(buffer) >= TRAIN_START and env_steps % TRAIN_FREQ == 0:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(BATCH_SIZE)

                states_t = torch.FloatTensor(states_b / 255.0).to(device)
                next_states_t = torch.FloatTensor(next_states_b / 255.0).to(device)
                actions_t = torch.LongTensor(actions_b).to(device)
                rewards_t = torch.FloatTensor(rewards_b).to(device)
                dones_t = torch.FloatTensor(dones_b).to(device)

                with torch.no_grad():
                    next_q = target_model(next_states_t).max(1)[0]
                    target_q = rewards_t + GAMMA * next_q * (1 - dones_t)

                current_q = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.policy[-1].parameters(), 10.0)
                optimizer.step()
                train_steps += 1

                if train_steps % TARGET_UPDATE == 0:
                    target_model.policy[-1].load_state_dict(model.policy[-1].state_dict())

                if train_steps % GREEDY_CHECK_STEPS == 0:
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

            if env_steps % 5000 == 0 and total_ep_rewards:
                gpu_temp = get_gpu_temp()
                eps_now = EPS_END + (EPS_START - EPS_END) * np.exp(-env_steps / EPS_DECAY)
                mean_r = np.mean(total_ep_rewards[-20:])
                print(f"EnvStep {env_steps:>7} | TrainStep {train_steps:>6} | EPS {eps_now:.3f} | MeanR: {mean_r:>7.1f} | GPU: {gpu_temp}C")
                if gpu_temp >= MAX_GPU_TEMP:
                    print(f"GPU {gpu_temp}C >= {MAX_GPU_TEMP}C -- stopping.")
                    break

    except KeyboardInterrupt:
        pass
    finally:
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
