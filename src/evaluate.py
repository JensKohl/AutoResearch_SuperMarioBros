import time
import argparse
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import MAX_EPISODE_STEPS
from src.train import PreprocessFrame, EnsureChannelFirst, FrameStack, FrameSkip, device, make_env
from src.model import PolicyModel

def evaluate(render=False):
    env = make_env(render=render)
    env = FrameSkip(env, skip=3)
    env = PreprocessFrame(env)
    env = EnsureChannelFirst(env)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n

    net = PolicyModel(n_actions).to(device)
    model_path = os.path.join("MODELS", "model.pt")
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    net.eval()
    state, info = env.reset()

    frames_survived = 0
    max_x_dist = 0
    flag_get = False
    score = 0

    with torch.no_grad():
        for _ in range(MAX_EPISODE_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            action = net(state_tensor).max(1)[1].view(1, 1).item()

            state, reward, terminated, truncated, info = env.step(action)
            frames_survived += 1
            max_x_dist = max(max_x_dist, info.get('x_pos', 0))
            flag_get = info.get('flag_get', False)
            score = info.get('score', 0)

            if render:
                time.sleep(0.01)

            if terminated or truncated or flag_get:
                break

    seconds_survived = frames_survived / 60.0
    seconds_to_finish = seconds_survived if flag_get else 0.0
    env.close()

    if flag_get:
        total_reward = 10000 + (400 - seconds_to_finish) + score + max_x_dist
    else:
        total_reward = score + max_x_dist

    print(f"Evaluation finished.")
    print(f"flag_get: {flag_get}")
    print(f"max_x_dist: {max_x_dist}")
    print(f"score: {score}")
    print(f"seconds_to_finish: {seconds_to_finish:.1f}")
    print(f"total_reward: {total_reward:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the gameplay')
    args = parser.parse_args()
    evaluate(render=args.render)