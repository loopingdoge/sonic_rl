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

from scipy.interpolate import make_interp_spline, BSpline

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines import PPO2, A2C
from stable_baselines.results_plotter import load_results, ts2xy

from sonic_util import make_env
from args import get_args
from levels import small_train_set, train_set, test_set
from utils import save_plot, check_subfolder_availability

best_mean_reward, n_steps = -np.inf, 0
global_logs_path = ""


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
        x, y = ts2xy(load_results(global_logs_path), "timesteps")
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
                _locals["self"].save(os.path.join(global_logs_path, "best_model.pkl"))
    n_steps += 1
    return True


def train(
    train_id,
    game,
    level,
    num_processes,
    num_timesteps,
    algo_name,
    policy_name,
    is_joint,
    model_save_path,
    logs_path,
    load_model_path=None,
):
    global global_logs_path
    global_logs_path = logs_path
    envs = []
    if is_joint:
        envs = [
            make_env(game=game, level=level, rank=i, log_dir=logs_path, seed=i * 100)
            for i, (game, level) in enumerate(small_train_set)
        ]
    else:
        envs = [
            make_env(game=game, level=level, rank=i, log_dir=logs_path, seed=i * 100)
            for i in range(num_processes)
        ]

    env = VecFrameStack(SubprocVecEnv(envs), 4)

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

    print(f"Starting training for {num_timesteps} timesteps")
    model.learn(total_timesteps=num_timesteps, callback=callback)
    print("Training finished!")

    if model_save_path:
        model.save(model_save_path)
        print("Model saved in:\t", model_save_path)

    timestep_values, score_values = ts2xy(load_results(logs_path), "timesteps")
    score_values = score_values * 100
    
    plot_path = os.path.join(logs_path, f"{level}.png")
    print("Saving the plot in: " + plot_path)
    save_plot(timestep_values, score_values, title=level, save_path=plot_path)

    env.close()


def main():
    global logs_path

    args = get_args()

    train_id = args.train_id
    num_processes = args.num_processes
    num_timesteps = args.timesteps
    game = args.game
    level = args.level
    model_save_path = args.save_dir + train_id + ".pkl"
    logs_path = os.path.join(
        args.logs_dir, check_subfolder_availability(args.logs_dir, train_id)
    )
    is_joint = args.joint
    load_model_path = args.load_model
    algo_name = args.algo
    policy_name = args.policy

    print("\n\n===============================================================")
    print("Num processes:\t\t", num_processes)
    print("Train timesteps:\t", num_timesteps)
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

    train(
        train_id=train_id,
        game=game,
        level=level,
        num_processes=num_processes,
        num_timesteps=num_timesteps,
        algo_name=algo_name,
        policy_name=policy_name,
        is_joint=is_joint,
        model_save_path=model_save_path,
        load_model_path=load_model_path,
        logs_path=logs_path,
    )


if __name__ == "__main__":
    main()
