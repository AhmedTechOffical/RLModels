import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import numpy as np
import time

env = make_atari_env("Breakout-v0")
env = VecFrameStack(env, n_stack=4)

a2c_path = os.path.join("Training","Saved_Models","A2C_Breakout")

model = A2C.load(a2c_path)

obs = env.reset()

episodes = 5

for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render(mode="rgb_array")
        action,_states = model.predict(obs)
        n_obs, reward, done, info = env.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))

env.close()

'''
for episodes in range(1,160):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render(mode="rgb_array")
        action,_states = model.predict(obs)
        obs,rewards,done,info = env.step(action)
        score+= rewards
    print("Episode: {} Reward {}".format(episodes,rewards))
'''
'''for episodes in range(1,6):
    action,_states = model.predict(obs)
    obs,rewards,done,info = env.step(action)
    env.render(mode="rgb_array")'''

    




