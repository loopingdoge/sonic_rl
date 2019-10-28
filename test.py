import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse

import warnings

import numpy as np

from stable_baselines.results_plotter import load_results, ts2xy

from levels import test_set
from train import train
from utils import check_subfolder_availability


def mcd_id(model_save_path, logs_path, test_id):
    """
    Serve per trovare un id che vada bene sia dentro logs/ che dentro models/
    # In alcuni casi molto particolari fa cose strane (ma sicure), ci riguardero'
    """
    test_id_1 = check_subfolder_availability(model_save_path, test_id)
    test_id_2 = check_subfolder_availability(logs_path, test_id)

    if test_id_1 == test_id and test_id_2 == test_id:
        return test_id
    else:
        id_1 = int(test_id_1[test_id_1.rfind("_") + 1 :]) if "_" in test_id_1 else 0
        id_2 = int(test_id_2[test_id_2.rfind("_") + 1 :]) if "_" in test_id_2 else 0
        mcid = f"{test_id}_{max([id_1, id_2])}"
        return mcd_id(model_save_path, logs_path, mcid)


def test(
    test_id, load_model_path, model_save_basedir, logs_dir, timesteps, algo, policy
):
    scores = []

    for (game, level) in test_set:
        model_save_path = os.path.join(model_save_dir, f"{level}.pkl")

        logs_path = os.path.join(logs_dir, level)

        train(
            train_id=test_id,
            game=game,
            level=level,
            num_processes=4,
            num_timesteps=timesteps,
            algo_name=algo,
            policy_name=policy,
            is_joint=False,
            model_save_path=model_save_path,
            logs_path=logs_path,
            load_model_path=load_model_path,
        )

        _, score_values = ts2xy(load_results(logs_path), "timesteps")
        mean_score = round(score_values.mean() * 100, 2)
        scores.append(mean_score)

        print("Mean Score: ", mean_score)
        with open(os.path.join(logs_path, "score.txt"), "a") as f:
            f.write(f"{mean_score}\n")

    final_score = round(np.array(scores).mean(), 2)
    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sonic's reinforcement learning testing suite"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(1e6),
        help="number of frames to train (default: 1e6)",
    )
    parser.add_argument(
        "--save-dir",
        default="./models/",
        help="directory to save agent checkpoints (default: ./models/)",
    )
    parser.add_argument(
        "--logs-dir",
        default="./logs/",
        help="directory to save tensorboard logs (default: ./logs/)",
    )
    parser.add_argument("--load-model", help="path of the model to load")
    parser.add_argument("--algo", default="ppo2", help="algorithm to use: a2c | ppo2")
    parser.add_argument(
        "--policy", default="cnn", help="algorithm to use: cnn | cnnlstm"
    )
    parser.add_argument("test_id", help="test id (used for the logs' name)")

    args = parser.parse_args()

    # Find a unique ID
    new_test_id = mcd_id(args.save_dir, args.logs_dir, args.test_id)
    logs_dir = os.path.join(args.logs_dir, new_test_id)
    model_save_dir = os.path.join(args.save_dir, new_test_id)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    score = test(
        args.test_id,
        args.load_model,
        model_save_dir,
        logs_dir,
        args.timesteps,
        args.algo,
        args.policy,
    )
    print("\n\nFinal Score: ", score)
    with open(os.path.join(logs_dir, "final_score.txt"), "a") as f:
        f.write(f"{score}\n")

