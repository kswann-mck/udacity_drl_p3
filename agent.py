"""
This script is the agent implementation of a Deep Deterministic Policy Gradient agent on the Unity Reacher environment. The base model, agent
and training function were taken from the solution here:
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum.

It was modified to include parameters to make training with different hyperparamter options easier.
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
        state_size=33,
        action_size=4,
        eps_start=1.0, 
        eps_end=0.01, 
        eps_decay=0.995,
        buffer_size=1e6,
        batch_size=256,
        update_every=20,
        update_times=1,
        actor_fc1_units=400,
        actor_fc2_units=300,
        critic_fc1_units=256,
        critic_fc2_units=128,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=1e-3,
        weight_decay=0,
        noise_theta=0.15,
        noise_sigma=0.20,
        random_seed=2
    ):
        """Initialize an Agent object.
        
        Parameters
        ----------
        state_size: the size of the state representation vector.
        action_size: the size of the action space.
        eps_start: float, the starting value of epsilon used to decrease the noise over time
        eps_end: float, the minimum value of epsilon
        eps_decay: float, the rate at which epsilon decays with subsequent timesteps
        buffer_size: the size of the replay experince buffer
        batch_size: int, the batch sized used for gradient descent during the learning phase
        update_every: int, the interval of episodes at which the learning step occurs
        update_times: the number of times to update at each update
        actor_fc1_units: int, the number of neurons in the first fully connected layer of the actor neural network
        actor_fc2_units: int, the number of neurons in the second fully connected layer of the actor neural network
        critic_fc1_units: int, the number of neurons in the first fully connected layer of the critic neural network
        critic_fc2_units: int, the number of neurons in the second fully connected layer of the critic neural network
        actor_lr: float, the learning rate for gradient descent of the actor network
        critic_lr: float, the learning rate for gradient descent of the critic network
        gamma: float, the reward discount factor used in updates
        tau: float, the interpolation parameter for the soft update
        weight_decay: the weight decay rate for the adam optimizer used on the critic network
        noise_theta: the theta term on the Ornstein-Uhlenbeck process used to add noise during training
        noise_sigma: the sigma term on the Ornstein-Uhlenbeck process used to add noise during training
        random_seed: the random seed used for consistency
        """
        self.state_size = state_size
        self.action_size = action_size
        self.eps_start = eps_start
        self.eps = self.eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.buffer_size=int(buffer_size)
        self.batch_size=int(batch_size)
        self.update_every = update_every
        self.update_times = update_times
        self.gamma = gamma
        self.tau = tau
        self.t_step = 0
        self.noise_mu = 0
        
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, actor_fc1_units, actor_fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, actor_fc1_units, actor_fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, critic_fc1_units, critic_fc2_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, critic_fc1_units, critic_fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, self.noise_mu, noise_theta, noise_sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, random_seed, buffer_size, batch_size)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and self.t_step == 0:
            for i in range(self.update_times):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            # decay noise over time
            action += np.max([self.eps, self.eps_end]) * self.noise.sample()
            self.epsilon = self.eps*self.eps_decay
        
        return np.clip(action, -1, 1)

    def reset(self):
        #self.eps = self.eps_start
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, seed, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        print(f"Replay Buffer Size: {buffer_size}, Batch Size: {batch_size}")
        self.action_size = action_size
        self.memory = deque(maxlen=int(buffer_size))  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)