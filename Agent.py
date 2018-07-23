import os
import random
from collections import deque

import numpy as np
from tensorflow.keras.models import load_model, save_model


class Agent:
    def __init__(self, model_name, tag='model'):
        self.model_name = model_name
        self.tag = tag
        self.state_space = None
        self.action_space = None
        self.model = None

    def act(self, state):
        options = self.model.predict(np.reshape(state, self.state_space))
        return np.argmax(options[0])

    def __enter__(self):
        self.model = load_model(f"model/{self.model_name}/{self.tag}")
        self.state_space = self.state_space = list(self.model.input_shape)
        self.state_space[0] = 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TrainingAgent(Agent):
    @staticmethod
    def register_model(model_name, model, tag='model', overwrite_existing=False):
        dir_name = f"model/{model_name}"

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
            save_model(model, f"model/{model_name}/{tag}")
        elif overwrite_existing:
            save_model(model, f"model/{model_name}/{tag}")

    def __init__(self, model_name, invoke_random, tag='model', memory_size=1000):
        super().__init__(model_name, tag=tag)
        self.invoke_random = invoke_random

        self.memory = deque(maxlen=memory_size)

        self.epsilon_decay = .95
        self.epsilon_min = .01
        self.epsilon = 1.0
        self.gamma = .95

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture('model')

    def capture(self, tag):
        save_model(self.model, f"model/{self.model_name}/{tag}")

    def act(self, state):
        if self.epsilon >= random.random():
            return self.invoke_random()
        return super().act(state)

    def replay(self, batch_size):
        for (state, action, reward, next_state, done) in random.sample(self.memory, min(len(self.memory), batch_size)):
            target = self.model.predict(state)
            target[0][action] = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            self.model.fit(state, target, epochs=1, verbose=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.reshape(state, self.state_space), action, reward,
                            np.reshape(next_state, self.state_space), done))
