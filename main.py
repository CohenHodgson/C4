board = [

    [0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0]

]

game = True

print("Are you player 1? Answer Y or N.")

answer = input()

player = 0 # intialization

if answer == "Y":
    player = 1
    opp = 2
else:
    player = 2
    opp = 1

round = 1

while game is True:

    print(f"Select a column. Player {player}:")

    for row in board:
        print(row)

    column_num = int(input()) - 1

    if column_num not in list(range(0,7)):
        print ("Please enter a valid column number, 1-7")

    choices = []
    row_list = range(0,6)



    for row in list(row_list): # for every row in selected column (1 to 7)
        if board[row][column_num] == 0: # check if it is 0
            choices.append(row) # if it is, append it to choices array
    board[choices[-1]][column_num] = player # alter board so that the lowest row number is put into column_num

    round+=1

    if round % 2 == 0:
        player = 2
        opp = 1
    else:
        player = 1
        opp = 2

    winCounter = 0

    nums_tracker = [3] # assume empty?

    for row in board: # horizontal wins
        for nums in row:
            if nums == nums_tracker[-1] and nums > 0: # if number = last number tried
                winCounter+=1 
                if winCounter == 4:
                    print("You Win!")
                    game = False
            else: # if number is different from last number tried
                winCounter = 1 # set to 0
                nums_tracker.append(nums) # put diff number into last tried list

    for y in range(0,7): # internal, so use 0-6
        if board[y][column_num] == nums_tracker[-1] and nums > 0:
            winCounter+=1
            if winCounter == 4:
                print("You Win!")
                game = False
            else:
                winCounter = 1
                nums_tracker.append(nums)

    for row in board:
        print(row)
