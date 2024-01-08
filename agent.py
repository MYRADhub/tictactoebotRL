import torch
import random
import numpy as np
from collections import deque
from game import TicTacToeAI
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(10, 256, 9)
        self.model.load_state_dict(torch.load('models/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = np.zeros(9)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 8)
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
    total_score = 0
    record = 0
    total_count = 0
    count = 0
    wins = 0
    agent = Agent()
    game = TicTacToeAI()
    print("training...")
    while True:
        # play a first random move to start the game if AI is O
        # if count % 2 == 1:
        game.play_random_start()

        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, winner = game.play_step(final_move)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # current_player = count % 2 + 1
            count += 1
            if winner == 2:
                wins += 1
            # train long memory, plot result
            agent.n_games += 1
            game.reset()
            agent.train_long_memory()
            if count == 20:
                score = wins
                total_count += 1
                wins = 0
                count = 0

                if score > record:
                    record = score
                    agent.model.save()

                print('Match', total_count, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / total_count
                print('Mean Score:', mean_score)
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
