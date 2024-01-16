import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, ELU, Input
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from typing import Tuple
from copy import copy

class Linear_QNet(Model):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Linear_QNet, self).__init__()
        self.inputs  = Input(shape=input_size)
        self.linear1 = Dense(hidden_size)
        self.linear2 = Dense(hidden_size)
        self.relu    = ELU()
        self.outputs = Dense(output_size)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.outputs(x)
        return x
    
class Trainer:

    def __init__(self, model: Model, lr: float, gamma: float) -> None:
        self.lr        = lr
        self.gamma     = gamma
        self.model     = model
        self.optimizer = Adam(learning_rate=self.lr)
        self.loss      = MeanSquaredError()

    #@tf.function(reduce_retracing=True)
    def train_step(self, state: np.ndarray, action: Tuple[int, int, int], reward: int, next_state: np.ndarray, done: bool) -> None:
        state      = tf.convert_to_tensor(state, dtype=tf.int32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.int32)
        action     = tf.convert_to_tensor(action, dtype=tf.int32)
        reward     = tf.convert_to_tensor(reward, dtype=tf.float32)

        if len(state.shape) == 1:
            state      = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            action     = tf.expand_dims(action, 0)
            reward     = tf.expand_dims(reward, 0)
            done       = (done,)

        with tf.GradientTape() as tape:
            pred   = self.model(state)
            target = copy(pred)
            target = target.numpy()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma*tf.reduce_max(self.model(tf.expand_dims(next_state[idx], 0)))
                target[idx][tf.argmax(action[idx])] = Q_new
            loss = self.loss(target, pred)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))