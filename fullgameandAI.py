import torch
from torch import nn
import torch.optim as optim
from collections import namedtuple
from collections import deque
import random
import math
import itertools


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # will be gpu on pc
print(f"Using {device} device")

learning_rate = 1e-3 # CHANGE ME
batch_size = 32 # CHANGE ME
epochs = 5 # CHANGE ME


class NeuralNetwork(nn.Module):
    def forward(self, x): # data goes through, x = linearized board, self = network
        x = self.flatten(x) # x = result of flatten
        logits = self.linear_relu_stack(x)
        return logits # Q values

    def __init__(self):
        super().__init__() # run intialization from parent (super -> nn.module in this case)
        self.flatten = nn.Flatten() # turn the 2d into 1d, 6x7=42.
        self.linear_relu_stack = nn.Sequential( # puts layers into order, input -> A -> B -> output
            nn.Linear(42, 128), # stack of linear and ReLU neural layers
            nn.ReLU(), # activation func, prunes (replaces with 0) negative nums.
            nn.Linear(128, 48), # nn.Linear is fully connected layer, 42 inputs -> linear layer -> 7 moves/outputs
            nn.ReLU(),
            nn.Linear(48, 7),
        )


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 6, 7, device=device) # one board, 6 rows, 7 columns
logits = model(X) # call model on x
y_pred = logits.argmax(1) # find the index of the greatest score
print(f"Predicted class: {y_pred}") # prints what the model thinks is best.

transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))


'''
here, transition is the container for state, action, next_state and reward.

state = board before move
action = move ai makes
next_state = board after move
reward = what happens because of move
'''

class ReplayMemory(object): # uhh, the memory.
    def __init__(self, capacity):
        '''
        aside from acting on behalf of the replayer memory,
        replay memory is reffered to as an object,
        call replaymemory with number and it'll reference the number to func
        '''
        self.memory = deque([], maxlen=capacity) # deque has special commands letting it remove and add things easier from right and left sides.
        '''
        deque[] makes it start with empty. maxlen=capacity means,
        do not let memory grow beyond set capacity, e.g., 10,000 games.
        '''

    def push(self, *args):
        self.memory.append(transition(*args))
        '''
        pass its arguements and use self to access memory.
        '''

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        '''
        random(pop, k) k is batch size
        '''

    def __len__(self): # gives length of object
        return len(self.memory)



board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

board_tensor = torch.tensor(board)


float_tensor_board = board_tensor.float() # model inputs and outputs float, flattened board is long/int

print(float_tensor_board)
print(float_tensor_board.shape)
print(float_tensor_board.dtype)

float_tensor_board = float_tensor_board.unsqueeze(0) # increase dimensionality

output = model(float_tensor_board)

print(output)
print(output.shape)

weight = torch.randn(42, 7, requires_grad=True) # for each possible choice we need 7 weights, each neuron will weigh each place?
bias = torch.randn(7) # 7 output neurons

float_tensor_board = float_tensor_board.flatten(start_dim=1, end_dim=2) # flatten board to 2d
trained_data = torch.matmul(float_tensor_board, weight) + bias # z=Wx+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(trained_data, output)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
Q is defined as Q(s,a), where s is state, and a is action,
so board state and column_num

arg max Q(s,a) means, give me the action with highest q value

Qπ(s,a)=r+γQπ(s′,π(s′))
means pi is the policy, the ais strategy
r is reward from said action
gamma is how much future rewards matter.
s' is the state after move
pi(s') is the plan of action after move.
all together, it reads the q value, as defined by policy, is equal to the
reward plus how much future rewards matter times
Q policy which is dependant on the future moves/weightings (s,a vs s',a')
'''

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

n_actions = 7 # number of possible actions in env

n_observations = 42 # places to check

policy_net = NeuralNetwork().to(device) # network training
target_net = NeuralNetwork().to(device) # delayed copy for Q

target_net.load_state_dict(policy_net.state_dict()) # copies policy net to target after both have been sent to device?fff

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


class Connect4Env:

    def reset(self):
        self.board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        return self.board

    def step(self, action):

        column = action

        if column < 0 or column >= 7:
            return self.board, -1, True, False, {}

        row = None

        for y in range(5, -1, -1):
            if self.board[y][column] == 0:
                row = y
                break

        if row is None:
            return self.board, -1, False, False, {}

        self.board[row][column] = 1

        if self.check_win(1):
            return self.board, 1, True, False, {}

        if all(self.board[0][x] != 0 for x in range(7)):
            return self.board, 0, True, False, {}

        opponent_columns = [
            x for x in range(7)
            if self.board[0][x] == 0
        ]

        if opponent_columns:
            opponent_column = random.choice(opponent_columns)

            for y in range(5, -1, -1):
                if self.board[y][opponent_column] == 0:
                    self.board[y][opponent_column] = 2
                    break

            if self.check_win(2):
                return self.board, -1, True, False, {}

        if all(self.board[0][x] != 0 for x in range(7)):
            return self.board, 0, True, False, {}

        return self.board, 0, False, False, {}

    def check_win(self, player):

        for y in range(6):
            for x in range(4):
                if all(self.board[y][x + i] == player for i in range(4)):
                    return True

        for y in range(3):
            for x in range(7):
                if all(self.board[y + i][x] == player for i in range(4)):
                    return True

        for y in range(3):
            for x in range(4):
                if all(self.board[y + i][x + i] == player for i in range(4)):
                    return True

        for y in range(3, 6):
            for x in range(4):
                if all(self.board[y - i][x + i] == player for i in range(4)):
                    return True

        return False

    @property
    def action_space(self):
        return self

    def sample(self):
        valid_actions = [
            x for x in range(7)
            if self.board[0][x] == 0
        ]

        return random.choice(valid_actions)


env = Connect4Env()


def select_action(state):
    global steps_done

    sample = random.random() # random decision for exploration NOT exploit

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1. * steps_done / EPS_DECAY
    )

    '''
    epsilon is the probability of choosing random action. decreases with each gen.
    math: ϵ=ϵend​+(ϵstart​−ϵend​)e−steps/decay -> 
    code: EPS_END + (EPS_START - EPS_END) * math.exp(-1. *steps_done / EPS_DECAY)
    ^ I'm not sure if that's right though
    '''

    steps_done += 1

    if sample > eps_threshold: # live update explore and exploit, so it exploits more as it gets better.
        with torch.no_grad(): # disables gradient calc, also a decorator
            return policy_net(state).max(1).indices.view(1, 1) # max q value, find index of that, shape result into tensor
    else:
        return torch.tensor(
            [[env.action_space.sample()]],
            device=device,
            dtype=torch.long
        )

    '''
    if epsilon is bigger, than return a random choice from n_actions = 7, with device, as a tensor
    '''


def optimize_model():

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(
            map(
                lambda s: s is not None,
                batch.next_state
            )
        ),
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
        next_state_values[non_final_mask] = target_net(
            non_final_next_states
        ).max(1).values

    expected_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(
        state_action_values,
        expected_action_values.unsqueeze(1)
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_value_(
        policy_net.parameters(),
        100
    )

    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50


for i_episode in range(num_episodes):

    observation = env.reset()

    state = torch.tensor(
        observation,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    for t in itertools.count():

        action = select_action(state)

        observation, reward, terminated, truncated, _ = env.step(
            action.item()
        )

        reward = torch.tensor(
            [reward],
            device=device,
            dtype=torch.float32
        )

        if terminated or truncated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

        memory.push(
            state,
            action,
            next_state,
            reward
        )

        state = next_state

        if terminated or truncated:
            break

        optimize_model()

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = (
            TAU * policy_net_state_dict[key]
            + (1 - TAU) * target_net_state_dict[key]
        )

    target_net.load_state_dict(target_net_state_dict)



# game code below, switch players to X and Os

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