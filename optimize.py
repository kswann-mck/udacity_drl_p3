from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from train import train_ddpg
import json
import numpy as np

N_EPISODES = 100

# define the search space for bayesian optimization
SPACE = [
    #Real(0, 0.5, "uniform", name='eps_end'),
    #Real(1e-5, 1e0, "log-uniform", name='eps_decay'),
    Categorical([0.30], name='eps_end'),
    Categorical([0.70], name='eps_decay'),
    Categorical([1e6], name="buffer_size"),
    Categorical([256], name="batch_size"),
    Categorical([20], name="update_every"),
    Categorical([10], name="update_times"),
    Categorical([128], name="actor_fc1_units"),
    Categorical([128], name="actor_fc2_units"),
    Categorical([128], name="critic_fc1_units"),
    Categorical([128], name="critic_fc2_units"),
    Categorical([1e-3], name='actor_lr'),
    Categorical([1e-4], name='critic_lr'),
#     Real(1e-5, 1e-3, "log-uniform", name='actor_lr'),
#     Real(1e-5, 1e-3, "log-uniform", name='critic_lr'),
    Categorical([0.95], name="gamma"),
    #Real(0.9, 0.99, "uniform", name="gamma"),
    #Real(1e-2, 1e-1, "log-uniform", name="tau"),
    Categorical([5e-2], name="tau"),
    Categorical([0], name="weight_decay"),
    #Real(0, 1e-3, "uniform", name="weight_decay"),
    #Real(0.01, 0.05, "log-uniform", name="noise_theta"),
    #Real(0.01, 0.05, "log-uniform", name="noise_sigma"),
    Categorical([0.015], name="noise_theta"),
    Categorical([0.02], name="noise_sigma")
]


def find_optimal_hyperparameters(env, brain_name, num_agents=1, episodes_per_batch=10, n_calls=50, space=SPACE):
    """
    Given an environment and unity brain_name, conduct a search for optimal hyperparameters, returning the optimal parameters
    and the raw output from the optimization process:
    
    Parameters
    ----------
    env: UnityEnvironment, the banana environment.
    brain_name: string, the name to associate with the model checkpoint

    Returns
    -------
    (params: Dict, the list of parameters used for the optimal score, res_gp: the raw output from the optimizer)
    
    """
    
    @use_named_args(space)
    def objective(**params):
        """The objective function to minimuze with Gaussian Process Regression"""
        scores = train_ddpg(env=env, brain_name=brain_name, num_agents=num_agents, n_episodes=episodes_per_batch, break_early=False, **params)
        return -np.mean(scores[-100:])
    
    """Find and return optimal hyperparameter values for the agent."""
    res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=0)
    
    # show the best score achieved with optimal hyperparameters
    print(f"Best score on {N_EPISODES} episodes: {-res_gp.fun}")

    # extract the parameters used for the best score, and (parameters we didn't change)
    params = {
        'break_early': True,
        'n_episodes': 2000,
        'eps_start': 1,
        'random_seed': 22,
        'eps_end': int(res_gp.x[0]),
        'eps_decay': int(res_gp.x[1]),
        'buffer_size': int(res_gp.x[2]),
        'batch_size': int(res_gp.x[3]),
        'update_every': int(res_gp.x[4]),
        'update_times': int(res_gp.x[5]),
        'actor_fc1_units': int(res_gp.x[6]),
        'actor_fc2_units': int(res_gp.x[7]),
        'critic_fc1_units': int(res_gp.x[8]),
        'critic_fc1_units': int(res_gp.x[9]),
        'actor_lr': int(res_gp.x[10]),
        'critic_lr': int(res_gp.x[11]),
        'gamma': int(res_gp.x[12]),
        'tau': int(res_gp.x[13]),
        'weight_decay': int(res_gp.x[14]),
        'noise_theta': int(res_gp.x[15]),
        'noise_sigma': int(res_gp.x[16]),   
    }
    
    print(json.dumps(params, indent=2))

    return params, res_gp