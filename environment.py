from game import TicTacToeAI
from gym import Env
import numpy as np

from gym.spaces import Discrete, Box


class TicTacToeEnv(Env):
    def __init__(self, second=False, self_play=False, dir='models/a2c_tictactoe_v2'):
        self.action_space = Discrete(9)
        self.observation_space = Box(low=0, high=2, shape=(10,), dtype=np.int32)
        self.game = TicTacToeAI()
        self.player = second
        self.self_play = self_play
        self.dir = dir
        if self.player:
            if self.self_play:
                self.game.play_self_start()
            else:
                self.game.play_random_start()
        self.state = self.game.get_state()

    def step(self, action):
        move = np.zeros(9)
        move[action] = 1
        # print('Action:', action)
        # print('Move:', move)
        # print('Current player:', self.game.current_player)
        # print('State:', self.state)
        # print('Board:', self.game.board)
        if self.self_play:
            reward, done, winner = self.game.play_step_self_play(move, self.dir)
        else:
            reward, done, winner = self.game.play_step(move)
        self.state = self.game.get_state()
        info = {}
        return self.state, reward, done, info

    def play_human(self, move):
        self.game.play_step_human(move)
        self.state = self.game.get_state()

    def game_over(self, get_winner=False):
        if get_winner:
            return self.game.game_over, self.game.winner
        else:
            return self.game.game_over

    def reset(self):
        self.game.reset()
        if self.player:
            if self.self_play:
                self.game.play_self_start()
            else:
                self.game.play_random_start()
        self.state = self.game.get_state()
        return self.state

    def render(self):
        print(self.game.board)
        print('-------------------')


if __name__ == '__main__':
    env = TicTacToeEnv()
    episodes = 20
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
