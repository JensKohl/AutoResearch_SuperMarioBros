import time
import subprocess
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import shimmy
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import cv2
import os
from collections import deque
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import TIME_BUDGET, MAX_EPISODE_STEPS, PRO_MOVEMENT
from src.model import PolicyModel

# Hyperparameters
GAMMA = 0.99
N_STEPS = 20        # rollout length before each update
VALUE_COEF = 0.5    # weight of critic loss
ENTROPY_COEF = 0.01 # entropy bonus to encourage exploration
LR = 1e-4
RENDER = True

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


# --- A2C Training Loop ---
def train():
    env = make_env(render=RENDER)
    env = FrameSkip(env, skip=3)
    env = DistanceReward(env)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    policy_net = PolicyModel(n_actions).to(device)
    # A2C uses RMSprop with standard settings (alpha=0.99, eps=1e-5)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LR, alpha=0.99, eps=1e-5)

    state, _ = env.reset()
    start_time = time.time()
    episode_reward = 0
    episode_count = 0
    episode_rewards = []
    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    try:
        while time.time() - start_time < TIME_BUDGET:
            log_probs_buf, values_buf, rewards_buf, dones_buf, entropies_buf = [], [], [], [], []

            for _ in range(N_STEPS):
                if time.time() - start_time >= TIME_BUDGET:
                    break
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                probs, value = policy_net.forward_ac(state_t)
                dist = Categorical(probs)
                action = dist.sample()

                next_state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                episode_reward += reward

                current_time = info.get('time', 0)
                current_score = info.get('score', 0)
                if current_time + current_score > best_total_reward:
                    best_total_reward = current_time + current_score
                    best_score = current_score
                    best_time = current_time

                log_probs_buf.append(dist.log_prob(action))
                values_buf.append(value.squeeze())
                rewards_buf.append(reward)
                dones_buf.append(float(done))
                entropies_buf.append(dist.entropy())

                state = next_state
                if done:
                    episode_rewards.append(episode_reward)
                    episode_count += 1
                    episode_reward = 0
                    state, _ = env.reset()
                    gpu_temp = get_gpu_temp()
                    print(f"Episode {episode_count:>3} | Reward: {episode_rewards[-1]:>6.1f} | GPU: {gpu_temp}°C")
                    if gpu_temp >= MAX_GPU_TEMP:
                        print(f"GPU {gpu_temp}°C >= {MAX_GPU_TEMP}°C — stopping.")
                        raise KeyboardInterrupt

            if not log_probs_buf:
                break

            # Bootstrap value for last state
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                _, next_value = policy_net.forward_ac(state_t)
            R = next_value.squeeze().detach()

            # Compute n-step returns
            returns = []
            for r, d in zip(reversed(rewards_buf), reversed(dones_buf)):
                R = r + GAMMA * R * (1.0 - d)
                returns.insert(0, R)

            returns_t = torch.stack(returns).detach()
            values_t = torch.stack(values_buf)
            log_probs_t = torch.stack(log_probs_buf)
            entropies_t = torch.stack(entropies_buf)

            advantages = returns_t - values_t.detach()
            actor_loss = -(log_probs_t * advantages).mean()
            critic_loss = (returns_t - values_t).pow(2).mean()
            entropy_loss = entropies_t.mean()

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            optimizer.step()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        os.makedirs("MODELS", exist_ok=True)
        torch.save(policy_net.state_dict(), "MODELS/model.pt")

        training_seconds = time.time() - start_time
        print(f"training_seconds: {training_seconds:.1f}")
        print(f"total_seconds: {training_seconds:.1f}")
        print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0:.1f}")

        seconds_to_finish = 400 - best_time if best_time > 0 else 999
        print(f"train_best_reward: {best_total_reward:.1f}")
        print(f"train_seconds_to_finish: {seconds_to_finish:.1f}")
        print(f"train_best_score: {best_score}")

        if episode_rewards:
            print(f"mean_episode_reward: {np.mean(episode_rewards):.1f}")


if __name__ == "__main__":
    train()
