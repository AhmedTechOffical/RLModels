import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import numpy as np
import time

dqn_path = os.path.join("Training","Saved_Models","DQN_Assault")

#env = gym.make("Assault-v0",render_mode = "human")
env = make_atari_env('Assault-v0')
env = VecFrameStack(env, n_stack=1)


model = DQN.load(dqn_path,env)

obs = env.reset()

print(evaluate_policy(model,env,n_eval_episodes=5,render=True))