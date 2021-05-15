# Project 3: Collaboration and Competition
## Udacity Deep Reinforcement Learning Nano-Degree
by Kitson Swann
submitted: 2021-05-14

This repository contains a solution to the third project of Udacity's Deep Reinforcement Learning Nano-degree.
The instructions for the project are [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

## Tennis Environment

![playing.gif](playing.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

### States

The state space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 

### Actions

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

# Rewards

The task is episodic. The task is considered solved when the maximum of the total score for the two agents is +0.5 (over 100 consecutive episodes).

After each episode, we add up the rewards that each agent received (without discounting, to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Project Installation

This project requires:

- Unity
- Python 3.6

1. Clone this repository `https://github.com/kswann-mck/udacity_drl_p3.git`
2. [Install Unity](https://unity3d.com/get-unity/download)
3. Create a folder `env` inside the cloned repo, and download the zip file containing the environment 
   [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip) and put 
   the `Tennis.app` file in the `env` folder.
4. Create a conda environment with the requirements by doing the 
   following `conda env create --name udacity_drl_p3 --file=environment.yml python=3.6`
5. Activate the environment: `conda activate udacity_drl_p3`
6. Run jupyter lab with `jupyterlab`
7. Open the [Report.ipynb](Report.ipynb) file in jupyter lab.

## Project Structure

- [model.py](model.py) - defines the structure of the neural networks for the actor and critic using PyTorch
- [agent.py](agent.py) - defines the agent implementation
- [train.py](train.py) - defines the training loop to run a training sequence with a given set of hyperparameters
- [optimize.py](optimize.py) - defines an optimization routine to search for optimal hyperparameters
- [Report.md](Report.md) - defines the solution steps, algorithm and outputs from different training and optimiztion runs in markdown
- [Report.ipynb](Report.ipynb) - defines the solution steps, algorithm and outputs from different training and optimiztion runs
- [checkpoint_actor_base.pth](checkpoint_actor_base.pth) - the saved model weights for tha actor network from the solution that solved the task
- [checkpoint_critic_base.pth](checkpoint_actor_base.pth) - the saved model weights for tha critic network from the solution that solved the task
- [playing.gif](playing.gif) - a gif of the trained optimal agent playing
- [environment.yml](environment.yml) - the conda environment file for reproducing the experiment

## Instructions

After you have successfully installed units and set up the python 3.6 environment and the conda environment 
requirements. If you wish, you can re-run the experiment by re-running all the cells in 
the [Report.ipynb](Report.ipynb).


