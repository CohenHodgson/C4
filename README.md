# Connect Four AI

A self-play Connect Four AI written in Python 3.12 with PyTorch.
The AI is a small DQN (Deep Q-Network) trained by playing against
itself. You can train a new model from scratch or play against a
pre-trained one.

View "noCom.py" for the un-commented code.

## Requirements

- Python 3.12+
- PyTorch (CPU or GPU)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch
```

## Run

```bash
python3 main.py
```

The terminal will prompt you to:

1. Choose an opponent: `Player` or `AI`.
2. If `AI`, choose `train` (start from scratch) or `use`
   (load `connect4_model.pth`).
3. Play by entering a column number 1–7 when prompted.

## Training

Training uses self-play: the same network plays both sides, and
wins are rewarded back to the network. Progress is printed in
batches of 100 games.

To change the number of training games, edit the `train_ai(num_games)`
call near the bottom of `main.py`. The model is saved to
`connect4_model.pth` after training finishes.

Note: the `train` option overwrites whatever file is in `connect4_model.pth`. 
If you want to keep an existing model, save it elsewhere or change the file name,
before training.

Training on CPU takes roughly 1–2 hours for 100,000 games. GPU
(MPS on macOS, CUDA on Linux/Windows) is selected automatically
when available.

## Playing against a pre-trained model

A pre-trained `connect4_model.pth` is in the project folder.
You can choose `AI` → `use` at the prompt to load it.

If you train your own model, it will overwrite this
file unless you change the save path or rename the
existing file first.

## Tuning

Search the source for `CHANGE ME` to find tunable constants:

- `learning_rate`, `batch_size` — near the top of the file
- `BATCH_SIZE`, `GAMMA`, `EPS_START`, `EPS_END`, `EPS_DECAY`, `TAU`, `LR` — the DQN hyperparameters
- `ReplayMemory(100_000)` — replay buffer size
- `train_ai(num_games)` — number of training games

If training feels too slow, try lowering `EPS_DECAY` or reducing
`train_ai(num_games)`.

## Acknowledgments

Thank you to the folks who made the PyTorch Tutorial.

This project follows two official PyTorch tutorials:

- [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
  for the general PyTorch workflow: tensors, model definition and optimization.
- [Reinforcement Learning (DQN) Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
  for the DQN architecture: replay memory, target network and the optimization loop (optimize_model).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE)
for the full text.

Portions of this project are adapted from the official PyTorch tutorials,
which are licensed under BSD 3-Clause. Full notice is in the LICENSE file.
