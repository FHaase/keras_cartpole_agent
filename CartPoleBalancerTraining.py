import gym
from tensorflow import keras

from Agent import TrainingAgent

gym_name = 'CartPole-v1'
env = gym.make(gym_name)


def dense_model(environment, learning_rate=0.01):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    model = keras.Sequential()
    model.add(Dense(24, input_dim=environment.observation_space.shape[0], activation='linear'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(environment.action_space.n, activation='linear'))

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


network_models = [('Dense', dense_model)]
for model_name, model_generator in network_models:
    name = f"{gym_name}-{model_name}"
    TrainingAgent.register_model(name, model_generator(env), overwrite_existing=True)

    with TrainingAgent(name, env.action_space.sample) as agent:
        done = False
        for i_episode in range(1, 2001):
            observation = env.reset()
            for t in range(1, 500):
                action = agent.act(observation)
                next_observation, reward, done, _ = env.step(action)
                agent.remember(observation, action, reward if not done else -10, next_observation, done)
                observation = next_observation

                if done:
                    break
            print(f"Episode {i_episode} finished after {t} timesteps")
            if not done:
                break
            agent.replay(64)
