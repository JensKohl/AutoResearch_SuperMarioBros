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

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 50000
TARGET_UPDATE = 1000
MEMORY_SIZE = 50000
LR = 1e-4
RENDER = True
N_STEP = 2  # N-step returns: R = r_t + γ·r_{t+1} + ... , bootstrap with γ^N
LEARN_START = 1000  # wait until replay buffer has this many transitions before training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

MAX_GPU_TEMP = 85  # °C — stop training gracefully if exceeded

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
    env = env.env  # unwrap gym 0.26 TimeLimit which expects 5-tuple but base env returns 4-tuple
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


# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # Input: 4 stacked grayscale frames (4x84x84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # Kaiming init for ReLU conv/fc layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# --- Training Loop ---
def train():
    env = make_env(render=RENDER)
    env = FrameSkip(env, skip=3)
    env = DistanceReward(env)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayBuffer(MEMORY_SIZE)
    steps_done = 0

    # N-step transition buffer: holds the last N (s,a,r) tuples
    nstep_buffer = deque(maxlen=N_STEP)

    def push_nstep(next_state, done):
        # Aggregate the oldest transition's N-step return and push to replay
        R = 0.0
        for i, (_, _, r_i) in enumerate(nstep_buffer):
            R += (GAMMA ** i) * r_i
        s0, a0, _ = nstep_buffer[0]
        memory.push(s0, a0, R, next_state, done)

    start_time = time.time()
    total_rewards = []

    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0
    # Track best snapshot by eval-aligned proxy: score + max_x_dist per episode.
    best_snapshot_metric = -float('inf')
    best_snapshot_state = None

    try:
        while time.time() - start_time < TIME_BUDGET:
            state, info = env.reset()
            nstep_buffer.clear()
            episode_reward = 0
            episode_max_x = 0
            episode_last_score = 0

            for t in range(MAX_EPISODE_STEPS):
                eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
                if random.random() > eps_threshold:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                        action = policy_net(state_tensor).max(1)[1].view(1, 1).item()
                else:
                    action = env.action_space.sample()
                steps_done += 1

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                nstep_buffer.append((state, action, reward))
                if len(nstep_buffer) == N_STEP:
                    push_nstep(next_state, done)
                state = next_state

                current_time = info.get('time', 0)
                current_score = info.get('score', 0)
                current_total = current_time + current_score
                if current_total > best_total_reward:
                    best_total_reward = current_total
                    best_score = current_score
                    best_time = current_time

                # Track per-episode peak position and final score for snapshot selection
                episode_max_x = max(episode_max_x, info.get('x_pos', 0))
                episode_last_score = current_score

                if len(memory) >= LEARN_START:
                    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                    states = torch.FloatTensor(states).to(device) / 255.0
                    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    next_states = torch.FloatTensor(next_states).to(device) / 255.0
                    dones = torch.FloatTensor(dones).to(device)

                    q_values = policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        # Double DQN: policy net picks action, target net evaluates it.
                        next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
                        # N-step: reward is already R = sum γ^i r_i, bootstrap with γ^N
                        target_q_values = rewards + ((GAMMA ** N_STEP) * next_q_values * (1 - dones))

                    loss = nn.SmoothL1Loss()(q_values.squeeze(), target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    optimizer.step()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if done or (time.time() - start_time >= TIME_BUDGET):
                    # Flush the remaining partial n-step windows at episode end
                    while len(nstep_buffer) > 0:
                        nstep_buffer.popleft()
                        if len(nstep_buffer) > 0:
                            # Compute n-step return over whatever's left
                            R = 0.0
                            for i, (_, _, r_i) in enumerate(nstep_buffer):
                                R += (GAMMA ** i) * r_i
                            s0, a0, _ = nstep_buffer[0]
                            memory.push(s0, a0, R, next_state, done)
                    break

            total_rewards.append(episode_reward)
            # Snapshot the model after any episode that beats our best
            # eval-aligned proxy (score + max_x_dist), matching evaluate.py.
            episode_metric = episode_last_score + episode_max_x
            if episode_metric > best_snapshot_metric:
                best_snapshot_metric = episode_metric
                best_snapshot_state = {k: v.detach().cpu().clone() for k, v in policy_net.state_dict().items()}
            gpu_temp = get_gpu_temp()
            print(f"Episode {len(total_rewards):>3} | Reward: {episode_reward:>6.1f} | Epsilon: {eps_threshold:>5.3f} | Steps: {steps_done} | GPU: {gpu_temp}°C")
            if gpu_temp >= MAX_GPU_TEMP:
                print(f"GPU temperature {gpu_temp}°C >= {MAX_GPU_TEMP}°C limit — stopping training to cool down.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        os.makedirs("MODELS", exist_ok=True)
        # Prefer the best snapshot if any episode improved on the running best.
        save_state = best_snapshot_state if best_snapshot_state is not None else policy_net.state_dict()
        torch.save(save_state, "MODELS/model.pt")
        print(f"saved_snapshot_metric: {best_snapshot_metric:.1f}")

        training_seconds = time.time() - start_time
        print(f"training_seconds: {training_seconds:.1f}")
        print(f"total_seconds: {training_seconds:.1f}")
        print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0:.1f}")

        seconds_to_finish = 400 - best_time if best_time > 0 else 999
        print(f"train_best_reward: {best_total_reward:.1f}")
        print(f"train_seconds_to_finish: {seconds_to_finish:.1f}")
        print(f"train_best_score: {best_score}")

        if len(total_rewards):
            print(f"mean_episode_reward: {np.mean(total_rewards):.1f}")


if __name__ == "__main__":
    train()
