import numpy as np
from stable_baselines3 import A2C
from environment import TicTacToeEnv


def train(times=100000, save=True):
    env = TicTacToeEnv(second=True, self_play=False)
    # model = A2C('MlpPolicy', env, verbose=1)
    model = A2C.load('models/a2c_tictactoe_v2', env=env, verbose=1)
    model.learn(total_timesteps=times)
    if save:
        model.save('models/a2c_tictactoe_v2')
    del model


def self_train(times=100000):
    for i in range(10):
        # train for X
        path_load = 'models/a2c_tictactoe_v' + str(i+2)
        path_save = 'models/a2c_tictactoe_v' + str(i+3)
        env = TicTacToeEnv(second=False, self_play=True, dir=path_load)
        model = A2C.load(path_load, env=env, verbose=1)
        model.learn(total_timesteps=times/10)
        model.save(path_save)
        del model
        del env
        # train for O
        env = TicTacToeEnv(second=True, self_play=True, dir=path_save)
        model = A2C.load(path_save, env=env, verbose=1)
        model.learn(total_timesteps=times / 10)
        model.save(path_save)
        del model
        del env


def play(turn):
    env = TicTacToeEnv()
    model = A2C.load('models/a2c_tictactoe_v10')
    obs = env.reset()
    done = False
    if turn == 'O':
        # get human input
        env.render()
        row = int(input('Enter the row (0-2): '))
        col = int(input('Enter the column (0-2): '))
        move = np.zeros(9)
        move[row * 3 + col] = 1
        env.play_human(move)
    while not done:
        obs = env.state
        action, _states = model.predict(obs)
        move = np.zeros(9)
        move[action] = 1
        env.play_human(move)
        # get human input
        env.render()
        if env.game_over():
            print('Game over, AI wins')
            break
        row = int(input('Enter the row (0-2): '))
        col = int(input('Enter the column (0-2): '))
        move = np.zeros(9)
        move[row * 3 + col] = 1
        env.play_human(move)
        if env.game_over():
            env.render()
            print('Game over, human wins')
            break


if __name__ == '__main__':
    # self_train()
    # train(times=50000, save=True)
    play('X')
