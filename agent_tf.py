import random
import numpy as np 
import tensorflow as tf 
from snake import SnakeAI, Point, Direction
from model_tf import Linear_QNet, Trainer
from collections import deque
from typing import Tuple

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR         = 0.001

class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma   = 0.9
        self.memory  = deque(maxlen=MAX_MEMORY)
        self.model   = Linear_QNet(11, 256, 3)
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeAI) -> np.ndarray:
        head    = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l   = game.direction == Direction.LEFT
        dir_r   = game.direction == Direction.RIGHT
        dir_u   = game.direction == Direction.UP
        dir_d   = game.direction == Direction.DOWN

        state   = [(dir_r and game.is_collision(point_r)) or
                   (dir_l and game.is_collision(point_l)) or
                   (dir_u and game.is_collision(point_u)) or
                   (dir_d and game.is_collision(point_d)),
                   
                   (dir_u and game.is_collision(point_r)) or
                   (dir_d and game.is_collision(point_l)) or
                   (dir_l and game.is_collision(point_u)) or
                   (dir_r and game.is_collision(point_d)),
                   
                   (dir_d and game.is_collision(point_r)) or
                   (dir_u and game.is_collision(point_l)) or
                   (dir_r and game.is_collision(point_u)) or
                   (dir_l and game.is_collision(point_d)),
                   
                   dir_l, dir_r, dir_u, dir_d,
                   
                   game.food.x < game.head.x,
                   game.food.x > game.head.x,
                   game.food.y < game.head.y,
                   game.food.y > game.head.y]
        
        return np.array(state, dtype=np.int32)
    
    def remember(self, state: np.ndarray, action: Tuple[int, int, int], reward: int, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: Tuple[int, int, int], reward: int, next_state: np.ndarray, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> Tuple[int, int, int]:
        self.epsilon = 80 - self.n_games
        final_move   = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move             = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0           = tf.convert_to_tensor(state, dtype=tf.float32)
            state0           = tf.reshape(state0, (1, 11))
            prediction       = self.model(state0)
            move             = tf.argmax(prediction).numpy().astype(int)
            if np.array_equal(move, [0, 0, 0]):
                move = random.randint(0, 2)
            final_move[move] = 1
        return final_move
    
def train() -> None:

    record = 0
    agent  = Agent()
    game   = SnakeAI()
    while True:
        state_old  = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new  = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    train()