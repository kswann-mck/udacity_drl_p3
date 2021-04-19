"""
This script is used to train a Deep Deterministic Policy Gradient agent on the Unity Reacher environment. The base model, agent
and training function were taken from the solution here:
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum.

It was modified to include parameters to make training with different hyperparamter options easier, and to work with
the Unity environment, which is slightly different than the OpenAI Gym API.
"""


from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import json
from ddpg_agent import Agent

def train_ddpg(
    env,
    brain_name,
    num_agents=1,
    name="",
    break_early=False,
    solved_threshold=31.0,
    n_episodes=2000, 
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
    random_seed=22
    ):
    """
    This function uses the ddpg_agent, and the actor and critic neural networks, to train an agent using a specific set of hyperparameters
    to solve the Unity Reacher environment.
    
    Parameters
    ----------
    env: UnityEnvironment, the Reacher environment.
    brain_name: The name of the brain in the unity environemts.
    num_agents: The number of agents training simultaniously.
    name: string, the name to associate with the model checkpoint.
    break_early: bool, if the operation should cease when the solved threshold is reached.
    solved_threshold: float,  the point at which the task is considered solved.
    n_episodes: int, the maximum number of episodes to train for.
    max_t: int, the maximum number of timesteps per episode.
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

    Returns
    -------
    scores: List[float], the list of score values for each episode.
    """
    
    params = {key: val for key, val in locals().items() if key != "env"}
    # print the set of parameters used in this call
    print(json.dumps(params, indent=2, default=str), end="\r")
    
    # initialize agent
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        eps_start=eps_start, 
        eps_end=eps_end, 
        eps_decay=eps_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        update_every=update_every,
        update_times=update_times,
        actor_fc1_units=actor_fc1_units,
        actor_fc2_units=actor_fc2_units,
        critic_fc1_units=critic_fc1_units,
        critic_fc2_units=critic_fc2_units,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        weight_decay=weight_decay,
        noise_theta=noise_theta,
        noise_sigma=noise_sigma,
        random_seed=random_seed
    )
    
    writer = SummaryWriter()


    scores_deque = deque(maxlen=100)
    scores = []
    episode_durations = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        states = env_info.vector_observations
        agent_scores = np.zeros(num_agents)
        start = time.time()
        while True:
            actions = agent.act(state=states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
            
            states = next_states
            agent_scores += env_info.rewards
            if np.any(dones):
                break 
        duration = time.time() - start
        episode_durations.append(duration)
        mean_accross_agents = np.mean(agent_scores)
        scores_deque.append(mean_accross_agents)
        scores.append(mean_accross_agents)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), f'checkpoint_actor{name}.pth')
        torch.save(agent.critic_local.state_dict(), f'checkpoint_critic{name}.pth')
        writer.add_scalar('Score', mean_accross_agents, i_episode)
        writer.add_scalar('Average_Score', np.mean(scores_deque), i_episode)
        writer.add_scalar('Episode Duration', duration, i_episode)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
        if np.mean(scores_deque) >= solved_threshold:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {round(np.mean(scores_deque), 2)}')
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor{name}.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic{name}.pth')
            if break_early == True:
                break

    return scores