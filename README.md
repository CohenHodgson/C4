This is a connect 4 AI project, it was made using PyTorch and Python 3.12.3.

You may either download and install the pre-trained AI model, or
create your own using the program values.

if you don't have much knowledge on PyTorch/Neural Nets as a whole,
than feel free to use Control F to find the values with the comments:
"# CHANGE ME", which will allow you to model your own AI.

You need to install PyTorch in the directory:

pip install torch

then you may run the project using:

~/directory/to/program/folder/python3 fullgameandAI.py

From there, the terminal will print the approximate progress of how long it is 
taking to train the AI (in batches of 100 games).

It is training by fighting itself (self play, s.t., the strategy that wins 
is rewarded to the neural net driving both fights) 
however many times you decide to make it,
you can change this via the "train_ai(number of games)" (line 387) function.

By default, it is 1000, as that takes ~10 minutes. But, that is also because
I am using a laptop with a CPU. The device that is used to train the AI
is determined by the 10th line:

"device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # will be gpu on pc"

