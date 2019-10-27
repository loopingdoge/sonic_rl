import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow.python.util.deprecation as deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False

    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy

from sonic_util import make_env
from args import get_args
from levels import train_set, test_set

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward

    # Print stats every 100 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(logs_path), "timesteps")
        if len(x) > 0:
            mean_reward = np.mean(y[-10:])
            print(x[-1], "timesteps")
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    best_mean_reward, mean_reward
                )
            )

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals["self"].save(logs_path + "best_model.pkl")
    n_steps += 1
    return True


def main():
    args = get_args()

    num_cpu = args.num_processes
    train_timesteps = args.timesteps
    game = args.game
    level = args.level
    model_save_path = args.save_dir + level + "/"
    logs_path = args.logs_dir + level + "/"
    is_full_set = args.full_set
    load_model = args.load_model

    print("\n\n===============================================================")
    print("Num CPU:\t\t", num_cpu)
    print("Train timesteps:\t", train_timesteps)
    print("Model save path:\t", model_save_path)
    print("Logs path:\t\t", logs_path)
    if not is_full_set:
        print("Game:\t\t\t", game)
        print("Level:\t\t\t", level)
    else:
        print("Testing full set")
    if load_model:
        print("Loading model:\t\t", load_model)
    else:
        print("Creating new model")
    print("===============================================================\n\n")

    envs = [
        make_env(game=game, level=level, rank=i, log_dir=logs_path)
        for i in range(num_cpu)
    ]

    if num_cpu > 1:
        env = SubprocVecEnv(envs)
    else:
        env = DummyVecEnv(envs)

    print("\n\n")
    model = None
    if load_model:
        print("Loading...")
        model = PPO2.load(load_model, env=env, tensorboard_log=logs_path)
    else:
        print("New model...")
        model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=logs_path)

    model.learn(total_timesteps=train_timesteps, callback=callback)

    model.save(model_save_path)


if __name__ == "__main__":
    main()
