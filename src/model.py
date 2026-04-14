import torch.nn as nn
import torch.nn.functional as F


class PolicyModel(nn.Module):
    """A2C Actor-Critic model with shared conv backbone.

    forward() returns action probs only — compatible with evaluate.py.
    forward_ac() returns (probs, value) — used during training.
    """
    def __init__(self, n_actions):
        super(PolicyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def _features(self, x):
        x = self.conv(x)
        return F.relu(self.fc(x.view(x.size(0), -1)))

    def forward(self, x):
        """Returns action probs — used by evaluate.py (argmax = greedy action)."""
        return F.softmax(self.actor(self._features(x)), dim=-1)

    def forward_ac(self, x):
        """Returns (probs, value) — used during A2C training."""
        feat = self._features(x)
        return F.softmax(self.actor(feat), dim=-1), self.critic(feat)
