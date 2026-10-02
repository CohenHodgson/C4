import os # for file save
import torch
from torch import nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import math
import time



device = "cpu"


class NeuralNetwork(nn.Module):
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48,7),
        )
        
model = NeuralNetwork().to(device)
print(model)

transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity): 
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 50_000
TAU = 0.005
LR = 1e-4

n_actions = 7

policy_net = NeuralNetwork().to(device)
target_net = NeuralNetwork().to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad=True)

criterion = nn.SmoothL1Loss()

memory = ReplayMemory(200_000)

steps_done = 0


def select_action(state, legal_actions): 

    global steps_done

    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. *steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:

        with torch.no_grad(): 

            q_values = policy_net(state)[0]

            for column in range(n_actions):
                if column not in legal_actions:
                    q_values[column] = float("-inf")

            return q_values.argmax().view(1,1)

    else:

        return torch.tensor([[random.choice(legal_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_q_values = target_net(non_final_next_states)

        legal_mask = non_final_next_states[:, 0, :] == 0

        next_q_values[~legal_mask] = float("-inf")

        next_state_values[non_final_mask] = next_q_values.max(1).values

        expected_action_values = reward_batch - (GAMMA * next_state_values)

    loss = criterion(
        state_action_values,
        expected_action_values.unsqueeze(1)
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), clip_value = 1.0)

    optimizer.step()


rows = 6
columns = 7

win_length = 4

directions = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "diag_neg": (1, 1),
    "diag_pos": (1, -1) 
}


def in_bounds(y, x):
    return 0 <= y < rows and 0 <= x < columns

def count_line(board, y, x, dy, dx, player): 
    count = 1
    for direc in (1, -1):
        next_y = y + direc * dy
        next_x = x + direc * dx
        while in_bounds(next_y, next_x) and board[next_y][next_x] == player:
            count += 1
            next_y += direc * dy
            next_x += direc * dx
    return count


def check_horizontal(board, y, x, player):
    return count_line(board, y, x, *directions["horizontal"], player) >= win_length


def check_vertical(board, y, x, player):
    return count_line(board, y, x, *directions["vertical"], player) >= win_length


def check_diag_neg(board, y, x, player):
    return count_line(board, y, x, *directions["diag_neg"], player) >= win_length


def check_diag_pos(board, y, x, player):
    return count_line(board, y, x, *directions["diag_pos"], player) >= win_length


win_check = (
    check_horizontal,
    check_vertical,
    check_diag_neg,
    check_diag_pos,
)


def check_win(board, last_move):
    """Return the winning player (1 or 2) or 0 if nobody won."""
    y, x = last_move
    player = board[y][x]
    if player == 0:
        return 0

    for check in win_check:
        if check(board, y, x, player):
            return player

    return 0




def get_legal_actions(board): 

    legal_actions = []
        
    for column in range (7):
        if board[0][column] == 0:
            legal_actions.append(column)
    return legal_actions    


def make_move(board, column, player):
    for row in range(5, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            return row

    return None


def get_state(board, player):
    opponent = 2 if player == 1 else 1

    state = [
        [
            1 if cell == player
            else -1 if cell == opponent
            else 0
            for cell in row
        ]
        for row in board
    ]
    return torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

def show_board():
    for row in board:
        print(row)

def train_ai(num_games):
    global board

    for episode in range(num_games): 

        board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]

        player = 1
        game_over = False
        prev = None 

        while game_over == False: 

            legal_actions = get_legal_actions(board)

            if not legal_actions:
                break

            state = get_state(board, player)

            moves_played = 42 - sum(row.count(0) for row in board)
            if moves_played < 4 and random.random() < 0.5:
                column = random.choice(legal_actions)
                action = torch.tensor([[column]], device=device, dtype=torch.long)

            action = select_action(state, legal_actions)
            column = action.item()

            row = make_move(board, column, player)


            if row is None:
                break
            winner = check_win(board, (row, column))

            if winner == player:

                reward = torch.tensor([1.0], device=device)
                memory.push(state, action, None, reward)

                if prev is not None:
                    memory.push(prev[0], prev[1], None,
                                
                                torch.tensor([-1.0], device=device))
                optimize_model() 
                optimize_model()
                optimize_model()
                optimize_model()
                game_over = True

            elif not get_legal_actions(board):

                reward = torch.tensor([0.0], device=device)
                memory.push(state, action, None, reward)

                if prev is not None:
                    memory.push(prev[0], prev[1], None,
                                torch.tensor([0.0], device=device))
                optimize_model()
                optimize_model()
                optimize_model()
                optimize_model()
                game_over = True

            else:
                if prev is not None:
                    memory.push(
                        prev[0],                                          
                        prev[1],
                        state,
                        torch.tensor([0.0], device=device)
                    )
                    optimize_model()
                    optimize_model()
                    optimize_model()
                    optimize_model()
                prev = (state, action) 

            player = 2 if player == 1 else 1

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                policy_net_state_dict[key] * TAU
                + target_net_state_dict[key] * (1 - TAU)
            )

        target_net.load_state_dict(target_net_state_dict)

        if (episode + 1) % 100 == 0:
            print(f"Finished {episode + 1}/{num_games} games")


game = True

print("Do you want to fight another player, or an AI?")

enemy = input()

if enemy == "Player":

    print("Are you player 1 or 2?")
    answer = input()

    if answer == "1":
        player = 1
    elif answer == "2":
        player = 2
    else:
        print("Invalid player.")
        game = False

elif enemy == "AI":
    print("Do you wish to train a new AI, or use a pre-trained one? (enter: train/use). Warning: unless you have changed the path, this will overwrite saved model.")
    choice = input()
    if choice == "train":
        
        start = time.time()

        train_ai(10_000) 

        torch.save(policy_net.state_dict(), "connect4_model.pth")  # save trained model

        print("Model saved. In connect4/path") 

        end = time.time()

        print(str(end - start) + " seconds")
        board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0] ]


    elif choice == "use":
        if os.path.exists("connect4_model.pth"):

            policy_net.load_state_dict(torch.load("10k_good.pth", map_location = torch.device(device))) 

        else:

            print("Please enter 'train' or 'use'")

            game = False
    ai = 2
    player = 1

else:
    print("Please enter either 'AI' or 'Player'")
    game = False


while game:

    print(f"Select a column. Player {player}:")
    show_board()

    if enemy == "AI" and player == ai:

        state = get_state(board, player)

        legal_actions = get_legal_actions(board)

        with torch.no_grad():
            q_values = policy_net(state)[0]

            for column in range(7):
                if column not in legal_actions:
                    q_values[column] = float("-inf")

            column_num = q_values.argmax().item()

        print(f"AI chooses column {column_num + 1}")

    else:

        try:
            column_num = int(input()) - 1

        except ValueError:
            print("Please enter a valid column number, 1-7")
            continue

    if column_num not in range(7):
        print("Please enter a valid column number, 1-7")
        continue

    choices = []

    for row in range(6):
        if board[row][column_num] == 0:
            choices.append(row)

    if not choices:
        print("That column is full.")
        continue

    row = choices[-1]
    board[row][column_num] = player
    winner = check_win(board, (row, column_num))

    if winner != 0:
        show_board()
        print(f"Player {winner} wins!")
        game = False
        continue

    if not get_legal_actions(board):
        show_board()
        print("Draw!")
        game = False
        continue

    if player == 1:
        player = 2
    else:
        player = 1