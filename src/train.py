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
N_WORKERS = 8       # parallel render=False environments
N_STEPS = 128       # steps per worker per rollout
CLIP_EPS = 0.2
K_EPOCHS = 4
MINI_BATCH = 256    # mini-batch size (out of N_WORKERS*N_STEPS=1024 per rollout)
VALUE_COEF = 0.5
GAE_LAMBDA = 0.95
GAMMA = 0.99
LR = 1e-4

# Two-phase entropy: stochastic phase uses 0.01, greedy phase uses 0.0
ENTROPY_COEF_STOCHASTIC = 0.01
ENTROPY_COEF_GREEDY = 0.0

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


def compute_gae(rewards, values, dones, next_value):
    """Compute GAE advantages and returns for one worker's trajectory."""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


GREEDY_CHECK_INTERVAL = 10  # run greedy eval every N rollouts


def greedy_eval(model, eval_env, n_episodes=5):
    """Run N greedy (argmax) episodes; return (best_x, flag_get_any)."""
    best_x = 0
    flag_get_any = False
    for _ in range(n_episodes):
        state, info = eval_env.reset()
        ep_max_x = 0
        ep_flag = False
        for _ in range(MAX_EPISODE_STEPS):
            st = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                action = model(st).max(1)[1].item()
            state, _, terminated, truncated, info = eval_env.step(action)
            ep_max_x = max(ep_max_x, info.get('x_pos', 0))
            if info.get('flag_get', False):
                ep_flag = True
                break
            if terminated or truncated:
                break
        best_x = max(best_x, ep_max_x)
        if ep_flag:
            flag_get_any = True
            break
    return best_x, flag_get_any


def collect_action(model, state_tensor, greedy_mode):
    """Sample or argmax action; always return (action, log_prob, value).

    In greedy_mode we use argmax for the action but still compute the log_prob
    under the current distribution. This lets PPO update toward making the
    greedy action even more probable (deterministic fine-tuning).
    """
    from torch.distributions import Categorical
    with torch.no_grad():
        f = model._features(state_tensor)
        logits = model.actor(f)
        value = model.critic(f).squeeze(-1)
        dist = Categorical(logits=logits)
        if greedy_mode:
            action = logits.max(1)[1]
        else:
            action = dist.sample()
        lp = dist.log_prob(action)
    return action, lp, value


def train():
    envs = [wrap_env(make_env(render=False)) for _ in range(N_WORKERS)]
    eval_env = wrap_env(make_env(render=False))
    states = [env.reset()[0] for env in envs]

    n_actions = envs[0].action_space.n
    model = PolicyModel(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()
    total_ep_rewards = []
    update_count = 0
    best_greedy_x = 0
    model_saved = False
    greedy_mode = False  # Two-phase: start stochastic, switch to greedy after flag_get

    best_total_reward = -float('inf')
    best_score = 0
    best_time = 0

    try:
        while time.time() - start_time < TIME_BUDGET:
            # ---- Collect rollout from all workers ----
            w_states  = [[] for _ in range(N_WORKERS)]
            w_actions = [[] for _ in range(N_WORKERS)]
            w_lp      = [[] for _ in range(N_WORKERS)]
            w_values  = [[] for _ in range(N_WORKERS)]
            w_rewards = [[] for _ in range(N_WORKERS)]
            w_dones   = [[] for _ in range(N_WORKERS)]
            w_ep_r    = [0.0] * N_WORKERS

            for step in range(N_STEPS):
                if time.time() - start_time >= TIME_BUDGET:
                    break
                for i in range(N_WORKERS):
                    st = torch.FloatTensor(states[i]).unsqueeze(0).to(device) / 255.0
                    action, lp, value = collect_action(model, st, greedy_mode)

                    next_state, reward, terminated, truncated, info = envs[i].step(action.item())
                    done = terminated or truncated
                    w_ep_r[i] += reward

                    # Phase switch: stochastic flag_get -> enter greedy fine-tuning phase
                    if info.get('flag_get', False) and not greedy_mode:
                        greedy_mode = True
                        os.makedirs("MODELS", exist_ok=True)
                        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                        model_saved = True
                        print(f"  *** Phase switch to GREEDY MODE at update {update_count} (stochastic flag_get!) ***")

                    w_states[i].append(states[i])
                    w_actions[i].append(action.item())
                    w_lp[i].append(lp.item())
                    w_values[i].append(value.item())
                    w_rewards[i].append(reward)
                    w_dones[i].append(float(done))

                    current_time = info.get('time', 0)
                    current_score = info.get('score', 0)
                    ct = current_time + current_score
                    if ct > best_total_reward:
                        best_total_reward = ct
                        best_score = current_score
                        best_time = current_time

                    if done:
                        total_ep_rewards.append(w_ep_r[i])
                        w_ep_r[i] = 0.0
                        states[i] = envs[i].reset()[0]
                    else:
                        states[i] = next_state

            if time.time() - start_time >= TIME_BUDGET:
                break

            # ---- Compute advantages for each worker ----
            all_states, all_actions, all_old_lp, all_adv, all_ret = [], [], [], [], []
            for i in range(N_WORKERS):
                if not w_rewards[i]:
                    continue
                nst = torch.FloatTensor(states[i]).unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    _, _, nv = model.act(nst)
                    next_val = nv.item() * (1 - w_dones[i][-1])
                adv, ret = compute_gae(w_rewards[i], w_values[i], w_dones[i], next_val)
                all_states.extend(w_states[i])
                all_actions.extend(w_actions[i])
                all_old_lp.extend(w_lp[i])
                all_adv.extend(adv)
                all_ret.extend(ret)

            if not all_states:
                continue

            entropy_coef = ENTROPY_COEF_GREEDY if greedy_mode else ENTROPY_COEF_STOCHASTIC

            N_total = len(all_states)
            states_t   = torch.FloatTensor(np.array(all_states)).to(device) / 255.0
            actions_t  = torch.LongTensor(all_actions).to(device)
            old_lp_t   = torch.FloatTensor(all_old_lp).to(device)
            adv_t      = torch.FloatTensor(all_adv).to(device)
            ret_t      = torch.FloatTensor(all_ret).to(device)

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

            # ---- PPO update ----
            indices = np.arange(N_total)
            for _ in range(K_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, N_total, MINI_BATCH):
                    mb = indices[start:start + MINI_BATCH]
                    lp, vals, entropy = model.evaluate(states_t[mb], actions_t[mb])
                    ratio = torch.exp(lp - old_lp_t[mb])
                    a = adv_t[mb]
                    surr = torch.min(ratio * a,
                                     torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * a)
                    actor_loss  = -surr.mean()
                    critic_loss = nn.MSELoss()(vals, ret_t[mb])
                    loss = actor_loss + VALUE_COEF * critic_loss - entropy_coef * entropy.mean()

                    if torch.isnan(loss):
                        continue
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            update_count += 1

            # Greedy checkpoint: 5 episodes, save if flag or new best x
            if update_count % GREEDY_CHECK_INTERVAL == 0:
                gx, gflag = greedy_eval(model, eval_env, n_episodes=5)
                if gflag:
                    best_greedy_x = gx
                    model_saved = True
                    os.makedirs("MODELS", exist_ok=True)
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                    print(f"  *** Greedy FLAG! x_dist={gx} (saved) ***")
                elif gx > best_greedy_x:
                    best_greedy_x = gx
                    model_saved = True
                    os.makedirs("MODELS", exist_ok=True)
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "MODELS/model.pt")
                    print(f"  Greedy checkpoint: x_dist={best_greedy_x} (saved) [{'greedy' if greedy_mode else 'stochastic'} phase]")
                else:
                    print(f"  Greedy check: x_dist={gx} (best={best_greedy_x}) [{'greedy' if greedy_mode else 'stochastic'} phase]")

            if total_ep_rewards:
                gpu_temp = get_gpu_temp()
                mean_r = np.mean(total_ep_rewards[-N_WORKERS:])
                phase = "G" if greedy_mode else "S"
                print(f"Update {update_count:>4} [{phase}] | MeanReward: {mean_r:>7.1f} | Episodes: {len(total_ep_rewards)} | GPU: {gpu_temp}C")
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
