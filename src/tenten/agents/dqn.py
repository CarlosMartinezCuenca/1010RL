import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, obs_size, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class ReplayBuffer:
    def __init__(self, capacity, obs_size, n_actions):
        self.capacity = capacity
        self.size = 0
        self.pos = 0

        # Las observaciones son 0/1: uint8 ocupa 4 veces menos que float32
        self.obs = np.zeros((capacity, obs_size), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.uint8)
        self.next_masks = np.zeros((capacity, n_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, next_mask, done):
        i = self.pos

        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_masks[i] = next_mask
        self.dones[i] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32),
            torch.as_tensor(self.next_masks[idx]),
            torch.as_tensor(self.dones[idx]),
        )


class DQNAgent:
    def __init__(
        self,
        obs_size,
        n_actions,
        hidden=256,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        seed=0,
    ):
        torch.manual_seed(seed)

        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.batch_size = batch_size

        self.q = QNetwork(obs_size, n_actions, hidden)
        self.q_target = QNetwork(obs_size, n_actions, hidden)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_size, n_actions)

    def choose(self, obs, mask, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))

        with torch.no_grad():
            q_values = self.q(torch.as_tensor(obs, dtype=torch.float32)).numpy()

        q_values[~mask] = -np.inf

        return int(np.argmax(q_values))

    def act(self, env, info):
        return self.choose(env._obs(), info["action_mask"], epsilon=0.0)

    def remember(self, obs, action, reward, next_obs, next_mask, done):
        self.buffer.add(obs, action, reward, next_obs, next_mask, done)

    def learn(self):
        obs, actions, rewards, next_obs, next_masks, dones = self.buffer.sample(
            self.batch_size, self.rng
        )

        q_sa = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q(next_obs).masked_fill(~next_masks, -1e9)
            best_next = next_q_online.argmax(dim=1, keepdim=True)

            next_q = self.q_target(next_obs).gather(1, best_next).squeeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
        self.update_target()
