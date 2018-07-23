import os

import gym

from Agent import Agent

gym_name = 'CartPole-v1'
env = gym.make(gym_name)

observation_space = env.observation_space.shape[0]
for model_name in os.listdir('model'):
    with Agent(model_name) as agent:
        observation = env.reset()
        t = 0
        while env.render():
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            t += 1
            if done:
                print(f"Episode {model_name} finished after {t} timesteps.")
                break
