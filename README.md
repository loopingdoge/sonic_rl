# Sonic RL

## Usage

```sh
usage: train.py [-h] [--timesteps TIMESTEPS] [--game GAME] [--level LEVEL]
                [--num-processes NUM_PROCESSES] [--save-dir SAVE_DIR]
                [--logs-dir LOGS_DIR] [--full-set FULL_SET]
                [--load-model LOAD_MODEL]

Sonic's reinforcement learning

optional arguments:
  -h, --help            show this help message and exit
  --timesteps TIMESTEPS
                        number of frames to train (default: 1e6)
  --game GAME           game to train on (default: SonicTheHedgehog-Genesis)
  --level LEVEL         lebel to train on (default: GreenHillZone.Act1)
  --num-processes NUM_PROCESSES
                        how many training CPU processes to use (default: 4)
  --save-dir SAVE_DIR   directory to save agent checkpoints (default:
                        ./models/)
  --logs-dir LOGS_DIR   directory to save tensorboard logs (default: ./logs/)
  --full-set FULL_SET   train on the full test set
  --load-model LOAD_MODEL
                        path of the model to load

```

## Setup

```
pip install tensorflow-gpu==1.14
pip install stable_baselines[mpi]
pip install gym-retro

git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"


git clone https://github.com/openai/baselines.git
pip install -e "baselines"
```

Import your games using:
```
python -m retro.import.sega_classics
```

## Visualize

```
pip install tensorboard
tensorboard --logdir path/to/log/foler
```