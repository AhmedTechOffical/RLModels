import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import numpy as np
import time

env = gym.make("Breakout-v0", render_mode = "human")

episodes = 5

env = make_atari_env('Breakout-v0', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

log_path = os.path.join("Training","Logs")
a2c_path = os.path.join("Training","Saved_Models","A2C_Breakout")
model = A2C("CnnPolicy",env, verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=500_000)
model.save(a2c_path)







''' 
env = gym.make("Breakout-v0", render_mode = "human")
env = make_atari_env('Breakout-v0', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=1)

obs = env.reset()

for episode in range(1, episodes+1):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        env.render(mode = "rgb_array")
        print(observation)
        action = env.action_space.sample()
        obs,reward,done,info = env.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode,score))


env.close()

'''

