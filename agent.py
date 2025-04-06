import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 80
LR = 0.00001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(256, 3)
        if load_model:
            self.model.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        block_size = game.get_block_size()

        head_x_norm = head.x / game.w
        head_y_norm = head.y / game.h
        food_x_norm = game.food.x / game.w
        food_y_norm = game.food.y / game.h

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        point_l = Point(head.x - block_size, head.y)
        point_r = Point(head.x + block_size, head.y)
        point_u = Point(head.x, head.y - block_size)
        point_d = Point(head.x, head.y + block_size)

        danger_straight = (dir_r and game.is_collision(point_r)) or \
                          (dir_l and game.is_collision(point_l)) or \
                          (dir_u and game.is_collision(point_u)) or \
                          (dir_d and game.is_collision(point_d))
        
        danger_right = (dir_u and game.is_collision(point_r)) or \
                       (dir_d and game.is_collision(point_l)) or \
                       (dir_l and game.is_collision(point_u)) or \
                       (dir_r and game.is_collision(point_d))
    
        danger_left = (dir_d and game.is_collision(point_r)) or \
                      (dir_u and game.is_collision(point_l)) or \
                      (dir_r and game.is_collision(point_u)) or \
                      (dir_l and game.is_collision(point_d))
        
        columns = 32
        rows = 24
        grid = np.zeros((rows, columns))

        for i, segment in enumerate(game.snake):
            grid_x = min(int((segment.x / game.w) * columns), columns-1)
            grid_y = min(int((segment.y / game.h) * rows), rows-1)

            if i == 0:
                grid[grid_y, grid_x] = 1.0
            else:
                grid[grid_y, grid_x] = 0.5

        food_grid_x = min(int((game.food.x / game.w) * columns), columns-1)
        food_grid_y = min(int((game.food.y / game.h) * rows), rows-1)
        grid[food_grid_y, food_grid_x] = 0.8

        flattened_grid = grid.flatten()

        base_state = [
            # Head position
            head_x_norm,
            head_y_norm,
            
            # Food position
            food_x_norm,
            food_y_norm,
            
            # Dangers
            danger_straight,
            danger_right,
            danger_left,
            
            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            len(game.snake) / 100.0,

            head_x_norm,
            1.0 - head_x_norm,
            head_y_norm,
            1.0 - head_y_norm,
        ]

        state = np.concatenate((base_state, flattened_grid))

        return state.astype(float)

    def remember(self, state, hidden_state, action, reward, next_state, done):
        self.memory.append((state, hidden_state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, hidden_states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, hidden_states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, hidden_state, action, reward, next_state, done):
        self.trainer.train_step(state, hidden_state, action, reward, next_state, done)

    def get_action(self, state, hidden_state, training=True):
        #  random moves: tradeoff exploration / exploitation
        self.epsilon = max(minimum_epsilon, maximum_epsilon - self.n_games) if start_with_high_epsilon else minimum_epsilon
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon and training:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction, hidden_state = self.model(state0, hidden_state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move, hidden_state
    
    def init_zero_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.model.hidden_size, requires_grad=False)

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_recent_mean_scores = []
    total_score = 0
    best_recent_mean_scores = 0
    record = 0


    agent = Agent()
    game = SnakeGameAI(speed=game_speed)
    hidden = agent.init_zero_hidden()

    while True:
        # get old state
        state = agent.get_state(game)

        # get move
        final_move, hidden = agent.get_action(state, hidden, train_model)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if train_model:
            # train short memory
            agent.train_short_memory(state, hidden, final_move, reward, state_new, done)

            # remember
            agent.remember(state, hidden, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            if train_model:
                agent.train_long_memory()

            hidden = agent.init_zero_hidden()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / max(agent.n_games, 1)
            plot_mean_scores.append(mean_score)
            recent_mean_scores = sum(plot_scores[-recent_mean_amount:]) / min(recent_mean_amount, len(plot_scores))
            plot_recent_mean_scores.append(recent_mean_scores)

            if(recent_mean_scores > best_recent_mean_scores and agent.n_games >= minimum_games_before_save):
                best_recent_mean_scores = recent_mean_scores
                if(train_model):
                    agent.model.save()

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record, 'Best Recent Mean Scores:', best_recent_mean_scores)

            if(plot_results):
                plot(plot_scores, plot_mean_scores, plot_recent_mean_scores)
            


if __name__ == '__main__':
    load_model = False
    train_model = True
    start_with_high_epsilon = True
    plot_results = True
    minimum_epsilon = 5
    maximum_epsilon = 100
    recent_mean_amount = 100
    minimum_games_before_save = 30
    game_speed = 100000000 # 20 or 100000000 recommended
    train()