import math
import numpy as np
import tensorflow as tf
import random #랜덤으로 공급. 사람이 투입하는 값이 아님.
import pandas_datareader as data_reader
from collections import deque
from tqdm import tqdm


class AiTrainer:
    def __init__(self):
        pass

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=0.01))
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size +1, len(self.memory)):
            batch.append(self.memory[i])

