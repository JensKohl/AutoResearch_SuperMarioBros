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
from collections import deque
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import TIME_BUDGET, MAX_EPISODE_STEPS, PRO_MOVEMENT

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.005  # very small — encourage policy to peak at correct actions
LR = 1e-4
MAX_GRAD_NORM = 0.5
N_STEPS = 128    # smaller rollout → more frequent updates → faster convergence
N_EPOCHS = 4     # PPO update epochs per rollout
MINI_BATCH = 64  # minibatch size
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


# --- DQN stub: required by evaluate.py which imports and instantiates DQN ---
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


# --- Actor-Critic Network for PPO training ---
class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # Shared conv backbone (same as DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.policy = nn.Linear(512, n_actions)  # action logits
        self.value = nn.Linear(512, 1)           # state value

        # Kaiming init for conv, orthogonal for FC/output heads
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, x):
        features = self.conv(x).view(x.size(0), -1)
        h = self.fc(features)
        return self.policy(h), self.value(h)

    def act(self, x):
        """Sample action and return (action, log_prob, value)."""
        logits, value = self.forward(x)
        logits = torch.clamp(logits, -10.0, 10.0)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        """For PPO update: returns log_probs, values, entropy."""
        logits, values = self.forward(states)
        logits = torch.clamp(logits, -10.0, 10.0)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_v = next_value
        else:
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages


# --- Training Loop ---
def train():
    env = make_env(render=RENDER)
    env = FrameSkip(env, skip=3)
    env = DistanceReward(env)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    net = ActorCritic(n_actions).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    start_time = time.time()
    total_rewards = []
    episode_reward = 0
    episode_max_x = 0

    best_max_x = 0
    best_snapshot_state = None
    best_snapshot_metric = -float('inf')

    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    best_greedy_x = 0           # greedy probe best max_x
    best_greedy_snapshot = None # separate from stochastic snapshot
    n_rollouts = 0              # count rollouts for probe scheduling

    state, info = env.reset()
    steps_done = 0

    try:
        while time.time() - start_time < TIME_BUDGET:
            # ---- Collect N_STEPS rollout ----
            rollout_states = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_values = []
            rollout_rewards = []
            rollout_dones = []

            for _ in range(N_STEPS):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    action, log_prob, value = net.act(state_tensor)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                rollout_states.append(state)
                rollout_actions.append(action)
                rollout_log_probs.append(log_prob.item())
                rollout_values.append(value.item())
                rollout_rewards.append(reward)
                rollout_dones.append(float(done))

                episode_reward += reward
                episode_max_x = max(episode_max_x, info.get('x_pos', 0))
                steps_done += 1

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
                    episode_metric = episode_max_x
                    if episode_metric > best_snapshot_metric:
                        best_snapshot_metric = episode_metric
                        best_snapshot_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    gpu_temp = get_gpu_temp()
                    print(f"Episode {len(total_rewards):>3} | Reward: {episode_reward:>6.1f} | MaxX: {episode_max_x} | Steps: {steps_done} | GPU: {gpu_temp}°C")
                    if gpu_temp >= MAX_GPU_TEMP:
                        print(f"GPU temperature {gpu_temp}°C >= {MAX_GPU_TEMP}°C limit — stopping.")
                        raise KeyboardInterrupt
                    episode_reward = 0
                    episode_max_x = 0
                    state, info = env.reset()

                if time.time() - start_time >= TIME_BUDGET:
                    break

            # Skip PPO update if time budget exceeded
            if time.time() - start_time >= TIME_BUDGET:
                break

            # ---- Compute GAE ----
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                _, next_value = net.forward(state_tensor)
                next_value = next_value.squeeze().item()

            rewards_t = torch.FloatTensor(rollout_rewards).to(device)
            values_t = torch.FloatTensor(rollout_values).to(device)
            dones_t = torch.FloatTensor(rollout_dones).to(device)

            advantages = compute_gae(rewards_t, values_t, dones_t, next_value, GAMMA, GAE_LAMBDA)
            returns = advantages + values_t
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---- PPO Update ----
            states_arr = np.array(rollout_states, dtype=np.float32) / 255.0
            states_t = torch.FloatTensor(states_arr).to(device)
            actions_t = torch.LongTensor(rollout_actions).to(device)
            old_log_probs_t = torch.FloatTensor(rollout_log_probs).to(device)

            indices = np.arange(N_STEPS)
            for _ in range(N_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, N_STEPS, MINI_BATCH):
                    mb_idx = indices[start:start + MINI_BATCH]
                    mb_states = states_t[mb_idx]
                    mb_actions = actions_t[mb_idx]
                    mb_old_log_probs = old_log_probs_t[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_returns = returns[mb_idx]

                    log_probs, values, entropy = net.evaluate(mb_states, mb_actions)
                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = VALUE_COEF * F.mse_loss(values, mb_returns.detach())
                    entropy_loss = -ENTROPY_COEF * entropy.mean()

                    loss = policy_loss + value_loss + entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

            # ---- Greedy probe every 5 rollouts ----
            n_rollouts += 1
            if n_rollouts % 5 == 0:
                net.eval()
                g_state, _ = env.reset()
                g_max_x = 0
                with torch.no_grad():
                    for _ in range(600):
                        g_tensor = torch.FloatTensor(g_state).unsqueeze(0).to(device) / 255.0
                        logits, _ = net.forward(g_tensor)
                        logits = torch.clamp(logits, -10.0, 10.0)
                        g_action = logits.argmax(dim=1).item()
                        g_state, _, g_term, g_trunc, g_info = env.step(g_action)
                        g_max_x = max(g_max_x, g_info.get('x_pos', 0))
                        if g_term or g_trunc:
                            break
                net.train()
                state, info = env.reset()  # reset training state after probe
                if g_max_x > best_greedy_x:
                    best_greedy_x = g_max_x
                    best_greedy_snapshot = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    print(f"  [greedy probe] new best greedy_x={g_max_x}")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        os.makedirs("MODELS", exist_ok=True)
        # Prefer greedy probe snapshot (directly optimizes eval metric);
        # fall back to stochastic snapshot if no greedy probe ran.
        if best_greedy_snapshot is not None:
            ac_state = best_greedy_snapshot
            print(f"Using greedy snapshot (greedy_x={best_greedy_x}, stochastic_x={best_snapshot_metric:.0f})")
        else:
            ac_state = best_snapshot_state if best_snapshot_state is not None else {k: v.cpu() for k, v in net.state_dict().items()}
            print(f"Using stochastic snapshot (max_x={best_snapshot_metric:.0f})")
        dqn_for_eval = DQN(n_actions)
        dqn_for_eval.conv.load_state_dict(
            {k.replace('conv.', ''): v for k, v in ac_state.items() if k.startswith('conv.')})
        # ActorCritic fc[0] → DQN fc[0], ActorCritic policy → DQN fc[2]
        dqn_for_eval.fc[0].weight.data.copy_(ac_state['fc.0.weight'])
        dqn_for_eval.fc[0].bias.data.copy_(ac_state['fc.0.bias'])
        dqn_for_eval.fc[2].weight.data.copy_(ac_state['policy.weight'])
        dqn_for_eval.fc[2].bias.data.copy_(ac_state['policy.bias'])
        torch.save(dqn_for_eval.state_dict(), "MODELS/model.pt")
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
