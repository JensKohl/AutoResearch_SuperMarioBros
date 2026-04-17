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
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import TIME_BUDGET, MAX_EPISODE_STEPS, PRO_MOVEMENT
from src.model import PolicyModel

# PPO T=0.3 + LR=3e-6 continue from x=1519 (exp186)
# exp185 gave NEW PPO BEST: x=1519 total_reward=1719. Continue same config to push further.
N_WORKERS = 8
N_STEPS = 128
LR = 3e-6
MAX_GRAD_NORM = 0.5

# PPO
CLIP_EPS = 0.10
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 1
MINI_BATCH = 256
GREEDY_CHECK_ROLLOUTS = 2

SAMPLE_TEMP = 0.3      # near-greedy — safer workers

BARRIER_X = 899
BARRIER_BONUS = 750.0

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


def greedy_eval_x(model, _unused_eval_env=None):
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
    envs = [wrap_env(make_env(render=False), barrier_bonus=BARRIER_BONUS) for _ in range(N_WORKERS)]
    states = [env.reset()[0] for env in envs]

    n_actions = envs[0].action_space.n
    model = PolicyModel(n_actions).to(device)

    # Freeze conv + policy[:2] + beyond_head. Train policy[-1] + value_head.
    for p in model.conv.parameters():
        p.requires_grad = False
    for p in model.policy[:2].parameters():
        p.requires_grad = False
    for p in model.beyond_head.parameters():
        p.requires_grad = False
    trainable = list(model.policy[-1].parameters()) + list(model.value_head.parameters())
    optimizer = optim.Adam(trainable, lr=LR, eps=1e-5)

    model_path = "MODELS/model.pt"
    model_saved = False
    if os.path.exists(model_path):
        try:
            saved = torch.load(model_path, map_location=device)
            model.load_state_dict(saved, strict=False)
            print(f"Warm start: loaded model from {model_path}")
        except Exception as e:
            print(f"Load failed ({e}), starting fresh")

    # Reinitialize value_head: prevents stale over-estimates of x=899 states
    # causing negative advantages that push the policy away from x=899.
    for m in model.value_head:
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    print("Value head reinitialized (fresh advantage estimates)")

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

    try:
        while time.time() - start_time < TIME_BUDGET:
            # ── Collect rollout ──────────────────────────────────────────────────
            states_buf = []
            actions_buf = []
            log_probs_buf = []
            rewards_buf = []
            dones_buf = []
            values_buf = []

            for step in range(N_STEPS):
                states_t = torch.FloatTensor(
                    np.array(states, dtype=np.float32) / 255.0
                ).to(device)

                with torch.no_grad():
                    logits, values = model.full_forward(states_t)
                    probs = torch.softmax(logits / SAMPLE_TEMP, dim=1)
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                states_buf.append(states_t.cpu())
                actions_buf.append(actions.cpu())
                log_probs_buf.append(log_probs.cpu())
                values_buf.append(values.squeeze(1).cpu())

                next_states = []
                rewards_step = []
                dones_step = []

                for i in range(N_WORKERS):
                    ns, r, term, trunc, info = envs[i].step(actions[i].item())
                    done = term or trunc
                    ep_rewards[i] += r
                    rewards_step.append(r)
                    dones_step.append(float(done))

                    ct = info.get('time', 0) + info.get('score', 0)
                    if ct > best_total_reward:
                        best_total_reward = ct
                        best_score = info.get('score', 0)
                        best_time = info.get('time', 0)

                    if done:
                        total_ep_rewards.append(ep_rewards[i])
                        ep_rewards[i] = 0.0
                        ns = envs[i].reset()[0]
                    next_states.append(ns)

                rewards_buf.append(torch.tensor(rewards_step, dtype=torch.float32))
                dones_buf.append(torch.tensor(dones_step, dtype=torch.float32))
                states = next_states

            # ── GAE ──────────────────────────────────────────────────────────────
            with torch.no_grad():
                last_states_t = torch.FloatTensor(
                    np.array(states, dtype=np.float32) / 255.0
                ).to(device)
                _, last_vals = model.full_forward(last_states_t)
                last_vals = last_vals.squeeze(1).cpu()

            advantages = torch.zeros(N_STEPS, N_WORKERS)
            gae = torch.zeros(N_WORKERS)
            for t in reversed(range(N_STEPS)):
                next_v = last_vals if t == N_STEPS - 1 else values_buf[t + 1]
                delta = rewards_buf[t] + GAE_GAMMA * next_v * (1 - dones_buf[t]) - values_buf[t]
                gae = delta + GAE_GAMMA * GAE_LAMBDA * (1 - dones_buf[t]) * gae
                advantages[t] = gae
            returns = advantages + torch.stack(values_buf, dim=0)

            S = torch.stack(states_buf).view(-1, *states_buf[0].shape[1:])
            A = torch.stack(actions_buf).view(-1)
            LP = torch.stack(log_probs_buf).view(-1)
            ADV = advantages.view(-1)
            RET = returns.view(-1)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

            # ── PPO update ───────────────────────────────────────────────────────
            n = len(S)
            for _ in range(PPO_EPOCHS):
                idx = torch.randperm(n)
                for start in range(0, n, MINI_BATCH):
                    mb = idx[start:start + MINI_BATCH]
                    mb_s = S[mb].to(device)
                    mb_a = A[mb].to(device)
                    mb_lp = LP[mb].to(device)
                    mb_adv = ADV[mb].to(device)
                    mb_ret = RET[mb].to(device)

                    logits, vals = model.full_forward(mb_s)
                    dist = torch.distributions.Categorical(torch.softmax(logits / SAMPLE_TEMP, dim=1))
                    new_lp = dist.log_prob(mb_a)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - mb_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = VALUE_COEF * F.mse_loss(vals.squeeze(1), mb_ret)
                    loss = policy_loss + value_loss - ENTROPY_COEF * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                    optimizer.step()

            rollout_count += 1

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
            if final_combined > best_combined:  # strictly better — avoid saving on equal baseline
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                print(f"Final save: x={final_gx} score={final_gs} combined={final_combined}")
            else:
                print(f"No improvement (x={final_gx} combined={final_combined} <= best={best_combined}), skipping save")

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
