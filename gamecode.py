# game code below, switch players to X and Os

board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]


game = True

print("Are you player 1 or 2?.")

try:
    answer = str(input())
except ValueError:
    print("Please enter either '1' or '2'")
        

if answer == "1":
    player = 1
    opp = 2
    round = 1
elif answer == "2":
    player = 2
    opp = 1
    round = 0


def show_board():
    for row in board:
        print(row)


while game is True:

    print(f"Select a column. Player {player}:")
    show_board()

    try:
        column_num = int(input()) - 1
    except ValueError:
        print("Please enter a valid column number, 1-7")
        continue

    if column_num in range(0,7):
        round += 1

    if column_num not in range(0, 7):
        print("Please enter a valid column number, 1-7")
        continue

    choices = []

    for row in range(0, 6):
        if board[row][column_num] == 0:
            choices.append(row)

    if not choices:
        print("That column is full.")
        continue

    board[choices[-1]][column_num] = player


    if round % 2 == 0:
        player = 2
        opp = 1
    else:
        player = 1
        opp = 2


    x_winCounter = 0
    y_winCounter = 0

    x_nums_tracker = [0]
    y_nums_tracker = [0]

    # horizontal wins
    for row in board:
        x_winCounter = 0
        x_nums_tracker = [0]

        for nums in row:
            if nums == x_nums_tracker[-1] and nums != 0:
                x_winCounter += 1

                if x_winCounter == 4:
                    print(f"Player {opp} wins!")
                    game = False
                    show_board()
                    break
            else:
                x_winCounter = 1
                x_nums_tracker.append(nums)



    # vertical wins
    for y in range(0, 6):
        if board[y][column_num] == y_nums_tracker[-1] and board[y][column_num] != 0:
            y_winCounter += 1

            if y_winCounter == 4:
                print(f"Player {opp} wins!")
                game = False
                show_board()
                break
        else:
            y_winCounter = 1
            y_nums_tracker.append(board[y][column_num])

    # diagonal, bottom left to top right
    for y in range(3, 6):
        for x in range(0, 4):
            if board[y][x] != 0:
                if (board[y][x] == board[y-1][x+1] and
                    board[y][x] == board[y-2][x+2] and
                    board[y][x] == board[y-3][x+3]):

                    print(f"Player {opp} wins!")
                    game = False
                    show_board()

    # bottom right, top left
    for y in range(3, 6):
        for x in range(3, 7):
            if board[y][x] != 0:
                if (board[y][x] == board[y-1][x-1] and
                    board[y][x] == board[y-2][x-2] and
                    board[y][x] == board[y-3][x-3]):

                    print(f"Player {opp} wins!")
                    game = False
                    show_board()

    # top left, bottom right
    for y in range(0, 3):
        for x in range(0, 4):
            if board[y][x] != 0:
                if (board[y][x] == board[y+1][x+1] and
                    board[y][x] == board[y+2][x+2] and
                    board[y][x] == board[y+3][x+3]):

                    print(f"Player {opp} wins!")
                    game = False
                    show_board()

    # top right, bottom left
    for y in range(0, 3):
        for x in range(3, 7):
            if board[y][x] != 0:
                if (board[y][x] == board[y+1][x-1] and
                    board[y][x] == board[y+2][x-2] and
                    board[y][x] == board[y+3][x-3]):

                    print(f"Player {opp} wins!")
                    game = False
                    show_board()