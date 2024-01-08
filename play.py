from stable_baselines3 import A2C
import numpy as np
from environment import TicTacToeEnv


def play_computer():
    player1 = A2C.load('models/a2c_tictactoe_v10')
    player2 = A2C.load('models/a2c_tictactoe_v10')
    env.reset()
    done = False
    while not done:
        # player1 turn
        obs = env.state
        action, _states = player1.predict(obs)
        move = np.zeros(9)
        move[action] = 1
        env.play_human(move)
        # player2 turn
        env.render()
        if env.game_over():
            print('Game over, Player1 wins')
            break
        obs = env.state
        action, _states = player2.predict(obs)
        move = np.zeros(9)
        move[action] = 1
        env.play_human(move)
        result = env.game_over(get_winner=True)
        if result[0]:
            env.render()
            print('Game over,' + str(result[1]) + ' wins')
            break


if __name__ == '__main__':
    env = TicTacToeEnv()
    for i in range(5):
        play_computer()
