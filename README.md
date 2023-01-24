# Doom-RL

This is a Reinforcement Learning agent that plays the famous first person shooter video game DOOM. It is based on the TD-Learning algorithm DQN and its variations such as Double DQN, Dueling DQN, n-Step DQN and PER DQN. This project was tested using a setup with the following configuration:

* CPU &emsp;i7-11800H
* RAM &emsp;16GB
* GPU &emsp;GeForce RTX 3060 (6GB VRAM)

Project scripts were tested on a Ubuntu 20.04 LTS system using Python 3.8, PyTorch (CUDA 11.6) and the ViZDoom environment. Specific versions of Python packages are listed bellow:

* numpy..................(1.23.1)
* opencv-python....(4.7.0)
* torch.....................(1.13.1+cu116)
* torchaudio...........(0.13.1+cu116)
* torchvision...........(0.14.1+cu116)
* tqdm.....................(4.64.1)
* vizdoom................(1.1.14)
