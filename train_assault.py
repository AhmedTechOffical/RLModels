import gym
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import numpy as np
import time

#env = gym.make("Assault-v0",render_mode = "human")
env = make_atari_env('Assault-v0', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=1)


obs = env.reset()

log_path = os.path.join("Training","Logs")
dqn_path = os.path.join("Training","Saved_Models","DQN_Assault")

model = DQN(CnnPolicy, env, verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=500_000,tb_log_name="first run")
model.save(dqn_path)



episdoes = 5

#This should work but instead its giving me an error on env.step(action) line. Idky
'''
for episode in range(1,episdoes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_obs, reward, done, info = env.step(action)
        score+=reward
    
    print("Episdoe: {} Score: {}".format(episode, score))
    '''