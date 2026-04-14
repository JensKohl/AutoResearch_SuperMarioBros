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

# PPO Hyperparameters
N_STEPS = 256       # steps per rollout
CLIP_EPS = 0.1      # PPO clip range
K_EPOCHS = 4        # update epochs per rollout
MINI_BATCH = 64     # mini-batch size for PPO updates
ENTROPY_COEF = 0.01 # entropy bonus coefficient
VALUE_COEF = 0.5    # value loss coefficient
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 2.5e-4
RENDER = True

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


def compute_gae(rewards, values, dones, next_value):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def train():
    env = make_env(render=RENDER)
    env = FrameSkip(env, skip=3)
    env = DistanceReward(env)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    model = PolicyModel(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()
    total_rewards = []
    update_count = 0

    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    state, info = env.reset()

    try:
        while time.time() - start_time < TIME_BUDGET:
            # --- Collect rollout ---
            states_buf, actions_buf, log_probs_buf = [], [], []
            rewards_buf, values_buf, dones_buf = [], [], []
            episode_reward = 0.0
            episode_done = False

            for _ in range(N_STEPS):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    action, log_prob, value = model.act(state_t)

                next_state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                episode_reward += reward

                states_buf.append(state)
                actions_buf.append(action.item())
                log_probs_buf.append(log_prob.item())
                rewards_buf.append(reward)
                values_buf.append(value.item())
                dones_buf.append(float(done))

                current_time = info.get('time', 0)
                current_score = info.get('score', 0)
                current_total = current_time + current_score
                if current_total > best_total_reward:
                    best_total_reward = current_total
                    best_score = current_score
                    best_time = current_time

                state = next_state
                if done:
                    total_rewards.append(episode_reward)
                    gpu_temp = get_gpu_temp()
                    print(f"Episode {len(total_rewards):>3} | Reward: {episode_reward:>6.1f} | Updates: {update_count} | GPU: {gpu_temp}°C")
                    if gpu_temp >= MAX_GPU_TEMP:
                        print(f"GPU temp {gpu_temp}°C >= {MAX_GPU_TEMP}°C — stopping.")
                        raise KeyboardInterrupt
                    episode_reward = 0.0
                    state, info = env.reset()

                if time.time() - start_time >= TIME_BUDGET:
                    break

            # Bootstrap value for last state
            with torch.no_grad():
                last_state_t = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                _, _, next_value = model.act(last_state_t)
                next_value = next_value.item() * (1 - dones_buf[-1])

            advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, next_value)

            # Convert to tensors
            states_t = torch.FloatTensor(np.array(states_buf)).to(device) / 255.0
            actions_t = torch.LongTensor(actions_buf).to(device)
            old_log_probs_t = torch.FloatTensor(log_probs_buf).to(device)
            advantages_t = torch.FloatTensor(advantages).to(device)
            returns_t = torch.FloatTensor(returns).to(device)

            # Normalize advantages
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # --- PPO update ---
            indices = np.arange(N_STEPS)
            for _ in range(K_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, N_STEPS, MINI_BATCH):
                    mb = indices[start:start + MINI_BATCH]
                    mb_states = states_t[mb]
                    mb_actions = actions_t[mb]
                    mb_old_lp = old_log_probs_t[mb]
                    mb_adv = advantages_t[mb]
                    mb_ret = returns_t[mb]

                    log_probs, values, entropy = model.evaluate(mb_states, mb_actions)

                    ratio = torch.exp(log_probs - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(values, mb_ret)
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                update_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        os.makedirs("MODELS", exist_ok=True)
        torch.save(model.state_dict(), "MODELS/model.pt")

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
