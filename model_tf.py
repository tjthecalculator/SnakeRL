import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, ReLU, Softmax,Input
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from typing import Tuple
from copy import deepcopy

class Linear_QNet(Model):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Linear_QNet, self).__init__()
        self.inputs  = Input(shape=input_size)
        self.linear1 = Dense(hidden_size)
        self.linear2 = Dense(output_size)
        self.relu    = ReLU()
        self.softmax = Softmax()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.linear2(x)
        return self.softmax(x)
    
class Trainer:

    def __init__(self, model: Model, lr: float, gamma: float) -> None:
        self.lr        = lr
        self.gamma     = gamma
        self.model     = model
        self.optimizer = Adam(learning_rate=self.lr)
        self.loss      = MeanSquaredError()

    def train_step(self, state: np.ndarray, action: Tuple[int, int, int], reward: int, next_state: np.ndarray, done: bool) -> None:
        state      = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action     = tf.convert_to_tensor(action, dtype=tf.float32)
        reward     = tf.convert_to_tensor(reward, dtype=tf.float32)

        if len(state.shape) == 1:
            state      = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            action     = tf.expand_dims(action, 0)
            reward     = tf.expand_dims(reward, 0)
            done       = (done,)

        with tf.GradientTape() as tape:
            pred   = self.model(state)
            target = deepcopy(pred).numpy()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    next_state = tf.reshape(next_state[idx], (1, 11))
                    Q_new      = reward[idx] + self.gamma*tf.reduce_max(self.model(next_state))
                target[idx][tf.argmax(action[idx]).numpy().astype(int)] = Q_new
                target = tf.convert_to_tensor(target, dtype=tf.float32)
            loss = self.loss(target, pred)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))