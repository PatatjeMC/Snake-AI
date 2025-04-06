import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 80
LR = 0.00005

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(256, 4)
        if load_model:
            self.model.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):       
        columns = 32
        rows = 24
        grid = np.zeros((rows, columns))

        for i, segment in enumerate(game.snake):
            grid_x = min(int((segment.x / game.w) * columns), columns-1)
            grid_y = min(int((segment.y / game.h) * rows), rows-1)

            value = (len(game.snake) - i) / len(game.snake)

            grid[grid_y, grid_x] = value

        food_grid_x = min(int((game.food.x / game.w) * columns), columns-1)
        food_grid_y = min(int((game.food.y / game.h) * rows), rows-1)
        grid[food_grid_y, food_grid_x] = 2.0

        flattened_grid = grid.flatten()

        return flattened_grid.astype(float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, training=True):
        #  random moves: tradeoff exploration / exploitation
        self.epsilon = max(minimum_epsilon, maximum_epsilon - self.n_games) if start_with_high_epsilon else minimum_epsilon
        final_move = [0,0,0,0]
        if random.randint(0, 200) < self.epsilon and training:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_recent_mean_scores = []
    total_score = 0
    best_recent_mean_scores = 0
    record = 0


    agent = Agent()
    game = SnakeGameAI(speed=game_speed)

    while True:
        # get old state
        state = agent.get_state(game)

        # get move
        final_move = agent.get_action(state, train_model)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if train_model:
            # train short memory
            agent.train_short_memory(state, final_move, reward, state_new, done)

            # remember
            agent.remember(state, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            if train_model:
                agent.train_long_memory()

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