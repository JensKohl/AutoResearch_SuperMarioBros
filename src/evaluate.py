import time
import subprocess
import argparse
import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import MAX_EPISODE_STEPS
from src.train import PreprocessFrame, EnsureChannelFirst, FrameStack, FrameSkip, DQN, device, make_env


def evaluate(render=False):
    env = make_env(render=render)
    env = FrameSkip(env, skip=3)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    model = DQN(n_actions).to(device)

    model_path = os.path.join("MODELS", "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    model.eval()
    state, info = env.reset()

    total_reward = 0
    frames_survived = 0
    max_x_dist = 0
    flag_get = False

    with torch.no_grad():
        for _ in range(MAX_EPISODE_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            action = model(state_tensor).max(1)[1].view(1, 1).item()

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames_survived += 1
            max_x_dist = max(max_x_dist, info.get('x_pos', 0))
            flag_get = info.get('flag_get', False)

            if render:
                time.sleep(0.01)

            if terminated or truncated or flag_get:
                break

    seconds_survived = frames_survived / 60.0
    env.close()

    if flag_get:
        combined_metric = 10000 + (1000 - seconds_survived) + max_x_dist
    else:
        combined_metric = max_x_dist

    print(f"Evaluation finished.")
    print(f"flag_get: {flag_get}")
    print(f"max_x_dist: {max_x_dist}")
    print(f"seconds_to_finish: {seconds_survived if flag_get else 0.0:.1f}")
    print(f"combined_metric: {combined_metric:.1f}")

    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        commit = "unknown"

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results.tsv")
    line = (
        f"{commit}\t{total_reward:.1f}\t{seconds_survived if flag_get else 0.0:.1f}\t"
        f"{combined_metric:.1f}\t{flag_get}\t{max_x_dist}\tpending\tauto-evaluation\n"
    )

    with open(results_path, "a") as f:
        f.write(line)
    print(f"Results appended to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the gameplay')
    args = parser.parse_args()
    evaluate(render=args.render)
