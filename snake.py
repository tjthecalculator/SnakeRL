import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Tuple

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED      = 40

class SnakeAI:

    def __init__(self, w: int = 640, h : int = 480) -> None:
        self.w       = w
        self.h       = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SnakeAI')
        self.clock   = pygame.time.Clock()
        self.reset()

    def reset(self) -> None:
        self.direction       = Direction.RIGHT
        self.head            = Point(self.w/2, self.h/2)
        self.snake           = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        self.score           = 0
        self.food            = None
        self.frame_iteration = 0

    def _place_food(self) -> None:
        x         = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y         = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action: Tuple[int, int, int]) -> Tuple[int, bool, int]:
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)
        reward    = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward    = -10
            return reward, game_over, self.score
        
        if self.head == self.food:
            self.score += 1
            reward      = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def is_collision(self, pt: Tuple[int, int] = None) -> bool:
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
    
    def _update_ui(self) -> None:
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()

    def _move(self, action: np.ndarray) -> None:
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx       = clockwise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1)%4
            new_dir  = clockwise[next_idx]
        else:
            next_idx = (idx - 1)%4
            new_dir  = clockwise[next_idx]

        self.direction = new_dir
        x              = self.head.x
        y              = self.head.y
        match self.direction:
            case Direction.RIGHT:
                x += BLOCK_SIZE
            case Direction.LEFT:
                x -= BLOCK_SIZE
            case Direction.DOWN:
                y += BLOCK_SIZE
            case Direction.UP:
                y -= BLOCK_SIZE

        self.head = Point(x, y)