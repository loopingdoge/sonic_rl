import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    import tensorflow.python.util.deprecation as deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines import PPO2, A2C
from stable_baselines.results_plotter import load_results, ts2xy

from sonic_util import make_env
from args import get_args
from levels import small_train_set, train_set, test_set

best_mean_reward, n_steps = -np.inf, 0
logs_path = ""


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


def adjust_logs_folder_path(logs_dir, id=-1):
    if id == -1:
        # First call, id is not set
        if os.path.exists(logs_dir):
            return adjust_logs_folder_path(logs_dir, id=0)
        else:
            return logs_dir
    else:
        # The folder aleady exists, we need to increment the id until we find a free one
        if os.path.exists(logs_dir[:-1] + "_" + str(id)):
            return adjust_logs_folder_path(logs_dir, id=id + 1)
        else:
            return logs_dir[:-1] + "_" + str(id) + "/"


def save_plot(x_values, y_values, title, save_path):
    fig, ax = plt.subplots()
    if len(x_values) > 0:
        ax.set(xlabel="Timesteps", ylabel="Score", title=title)
        ax.grid()

        poly = np.polyfit(x_values, y_values, 5)
        poly_y = np.poly1d(poly)(x_values)
        plt.plot(x_values, poly_y)

        fig.savefig(save_path)


def main():
    global logs_path

    args = get_args()

    train_id = args.train_id
    num_cpu = args.num_processes
    train_timesteps = args.timesteps
    game = args.game
    level = args.level
    model_save_path = args.save_dir + train_id + ".pkl"
    logs_path = adjust_logs_folder_path(args.logs_dir + train_id + "/")
    is_joint = args.joint
    load_model_path = args.load_model
    algo_name = args.algo
    policy_name = args.policy

    print("\n\n===============================================================")
    print("Num CPU:\t\t", num_cpu)
    print("Train timesteps:\t", train_timesteps)
    print("Model save path:\t", model_save_path)
    print("Logs path:\t\t", logs_path)
    if not is_joint:
        print("Game:\t\t\t", game)
        print("Level:\t\t\t", level)
    else:
        print("Joint Training")
    if load_model_path:
        print("Loading model:\t\t", load_model_path)
    else:
        print("Creating new model")
    print("===============================================================\n\n")

    envs = []
    if is_joint:
        for i, (game, level) in enumerate(small_train_set):
            envs.append(make_env(game=game, level=level, rank=i, log_dir=logs_path))
    else:
        envs = [
            make_env(game=game, level=level, rank=i, log_dir=logs_path)
            for i in range(num_cpu)
        ]

    if num_cpu > 1:
        env = SubprocVecEnv(envs)
    else:
        env = DummyVecEnv(envs)

    print("\n\n")

    algo = None
    if algo_name == "ppo2":
        algo = PPO2
    elif algo_name == "a2c":
        algo = A2C

    policy = None
    nminibatches = 4
    if policy_name == "cnn":
        policy = CnnPolicy
    elif policy_name == "cnnlstm":
        if is_joint:
            nminibatches = 5
        policy = CnnLstmPolicy

    model = None
    if load_model_path:
        print("Loading a model...")
        model = algo.load(load_model_path, env=env, tensorboard_log=logs_path)
    else:
        print("Creating a new model...")
        model = algo(
            policy, env, nminibatches=nminibatches, verbose=1, tensorboard_log=logs_path
        )

    print(f"Starting training for {train_timesteps} timesteps")
    model.learn(total_timesteps=train_timesteps, callback=callback)
    print("Training finished!")
    
    model.save(model_save_path)
    print("Model saved in:\t", model_save_path)

    timestep_values, score_values = ts2xy(load_results(logs_path), "timesteps")
    plot_path = os.path.join(logs_path, f"{level}.png")
    print("Saving the plot in: " + plot_path)
    save_plot(
        timestep_values,
        score_values,
        title=level,
        save_path=plot_path,
    )


if __name__ == "__main__":
    main()
