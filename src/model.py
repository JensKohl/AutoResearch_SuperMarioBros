import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyModel(nn.Module):
    """Actor-Critic CNN for PPO: shared conv backbone, separate actor/critic heads."""
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
        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _features(self, x):
        return self.conv(x).view(x.size(0), -1)

    def act(self, x):
        """Sample action and return (action, log_prob, value)."""
        f = self._features(x)
        dist = Categorical(logits=self.actor(f))
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(f).squeeze(-1)

    def evaluate(self, x, actions):
        """Evaluate actions for PPO update. Returns (log_probs, values, entropy)."""
        f = self._features(x)
        dist = Categorical(logits=self.actor(f))
        return dist.log_prob(actions), self.critic(f).squeeze(-1), dist.entropy()

    def forward(self, x):
        """Greedy action logits for evaluate.py compatibility."""
        return self.actor(self._features(x))
