import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    """PPO Actor-Critic network: shared CNN + separate policy (actor) and value (critic) heads.

    Architecture matches the previous Dueling DQN CNN so we can warm-start the convolutional
    layers from the existing x=1959 model checkpoint. The FC heads are re-initialized.

    evaluate.py compatibility: forward() returns action logits; argmax == greedy action,
    exactly like a DQN's Q-values.
    """
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        # Small init for policy output so initial action distribution is near-uniform
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy[-1].bias)

    def forward(self, x):
        """Returns action logits — evaluate.py uses .max(1)[1] for greedy action."""
        f = self.conv(x).view(x.size(0), -1)
        return self.policy(f)

    def full_forward(self, x):
        """Returns (logits, value) used during PPO training."""
        f = self.conv(x).view(x.size(0), -1)
        return self.policy(f), self.value_head(f)
