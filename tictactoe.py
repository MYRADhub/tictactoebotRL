import agent
import torch
import numpy as np
import train

# Create the tic-tac-toe board
board = np.array([[' ', ' ', ' '],
                  [' ', ' ', ' '],
                  [' ', ' ', ' ']])


# Function to print the board
def print_board():
    print('---------')
    for row in board:
        print('|', end='')
        for cell in row:
            print(cell, end='|')
        print('\n---------')


# Function to check if the game is over
def is_game_over():
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] != ' ':
            return True

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return True

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return True
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return True

    # Check if the board is full
    if ' ' not in board:
        return True

    return False


# Function to make a move
def make_move(player):
    if player == 'O':
        board_state = np.zeros(9)
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    board_state[i * 3 + j] = 1
                elif board[i][j] == 'O':
                    board_state[i * 3 + j] = 2

        board_state = np.append(board_state, 2)
        move = agent.get_action(board_state)
        move = np.where(move.reshape(3, 3) == 1)
        row = move[0][0]
        col = move[1][0]
        print(f'Agent chose row {row} and column {col}')
        board[row][col] = player
    else:
        while True:
            row = int(input('Enter the row (0-2): '))
            col = int(input('Enter the column (0-2): '))

            if row < 0 or row > 2 or col < 0 or col > 2 or board[row][col] != ' ':
                print('Invalid move. Try again.')
            else:
                board[row][col] = player
                break


def make_move_by_number(move_number, player):
    row = move_number // 3
    col = move_number % 3

    if row < 0 or row > 2 or col < 0 or col > 2 or board[row][col] != ' ':
        print('Invalid move. Try again.')
    else:
        board[row][col] = player


# Main game loop
def play_game():
    print('Welcome to Tic-Tac-Toe!')
    print_board()

    current_player = 'X'
    while not is_game_over():
        print(f"Player {current_player}'s turn:")
        make_move(current_player)
        print_board()
        current_player = 'O' if current_player == 'X' else 'X'

    print('Game over!')


if __name__ == '__main__':
    agent = agent.Agent()
    agent.model.load_state_dict(torch.load('models/model.pth'))
    agent.model.eval()

    play_game()
