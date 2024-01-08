import numpy as np
import random
from selfplay import SelfAgent


class TicTacToeAI:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.reward_win_X = 20
        self.reward_win_O = -20
        self.reward_draw = -5

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def get_state(self):
        state = np.array(self.board, dtype=int).flatten()
        state = np.append(state, self.current_player)

        return state

    def get_opponent_action(self):
        available_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    available_moves.append((i, j))
        row, col = random.choice(available_moves)
        return row, col

    def play_random_start(self):
        row, col = self.get_opponent_action()
        self.board[row][col] = self.current_player
        self.current_player = 2 if self.current_player == 1 else 1

    def play_self_start(self, dir='models/a2c_tictactoe_v2'):
        self_agent = SelfAgent(dir)
        action = self_agent.get_action(self.get_state())
        move = np.zeros(9)
        move[action] = 1
        row, col = np.where(move.reshape(3, 3) == 1)
        row = row[0]
        col = col[0]
        self.board[row][col] = self.current_player
        self.current_player = 2 if self.current_player == 1 else 1


    def play_step(self, move):
        # move is a numpy array of size 9 with 1 at the position of the move
        row, col = np.where(move.reshape(3, 3) == 1)
        row = row[0]
        col = col[0]
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            if self._check_winner(self.current_player):
                self.game_over = True
                self.winner = self.current_player
                reward = self.reward_win_X
            elif self._check_draw():
                self.game_over = True
                reward = self.reward_draw
            else:
                self.current_player = 2 if self.current_player == 1 else 1
                row, col = self.get_opponent_action()
                self.board[row][col] = self.current_player
                if self._check_winner(self.current_player):
                    self.game_over = True
                    self.winner = self.current_player
                    reward = self.reward_win_O
                elif self._check_draw():
                    self.game_over = True
                    reward = self.reward_draw
                else:
                    self.current_player = 2 if self.current_player == 1 else 1
                    reward = 0
            return reward, self.game_over, self.winner
        else:
            if self.current_player == 1:
                return -100, True, 2
            else:
                return -100, True, 1

    def play_step_human(self, move):
        # move is a numpy array of size 9 with 1 at the position of the move
        row, col = np.where(move.reshape(3, 3) == 1)
        row = row[0]
        col = col[0]
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            if self._check_winner(self.current_player):
                self.game_over = True
                self.winner = self.current_player
                reward = self.reward_win_X
            elif self._check_draw():
                self.game_over = True
                reward = self.reward_draw
            else:
                self.current_player = 2 if self.current_player == 1 else 1
                reward = 0
            return reward, self.game_over, self.winner
        else:
            return -100, True, 2

    def _check_winner(self, player):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == player:
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == player:
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] == player:
            return True
        return False

    def _check_draw(self):
        for row in self.board:
            if 0 in row:
                return False
        return True

    def play_step_self_play(self, move, dir='models/a2c_tictactoe_v2'):
        # move is a numpy array of size 9 with 1 at the position of the move
        self_agent = SelfAgent(dir)
        row, col = np.where(move.reshape(3, 3) == 1)
        row = row[0]
        col = col[0]
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            if self._check_winner(self.current_player):
                self.game_over = True
                self.winner = self.current_player
                reward = self.reward_win_X
            elif self._check_draw():
                self.game_over = True
                reward = self.reward_draw
            else:
                self.current_player = 2 if self.current_player == 1 else 1
                action = self_agent.get_action(self.get_state())
                move = np.zeros(9)
                move[action] = 1
                row, col = np.where(move.reshape(3, 3) == 1)
                row = row[0]
                col = col[0]
                self.board[row][col] = self.current_player
                # print(self.board)
                if self._check_winner(self.current_player):
                    self.game_over = True
                    self.winner = self.current_player
                    reward = self.reward_win_O
                elif self._check_draw():
                    self.game_over = True
                    reward = self.reward_draw
                else:
                    self.current_player = 2 if self.current_player == 1 else 1
                    reward = 0
            return reward, self.game_over, self.winner
        else:
            if self.current_player == 1:
                return -100, True, 2
            else:
                return -100, True, 1
