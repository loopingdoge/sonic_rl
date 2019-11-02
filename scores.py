import os
import argparse
import numpy as np

from typing import List
from stable_baselines.results_plotter import load_results, ts2xy

from utils import subdirs_list

scores_filename = "scores.csv"


def score(logs_dir: str) -> float:
    _, score_values = ts2xy(load_results(logs_dir), "timesteps")
    mean_score = round(score_values.mean() * 100, 2)
    return mean_score


def final_score(mean_scores: List[float]) -> float:
    final_score = round(np.array(mean_scores).mean(), 2)
    return final_score


def log_scores(logs_base_dir: str):
    """
    Creates a scores.csv file inside the logs_base_dir folder
    """
    dirs = subdirs_list(logs_base_dir)
    scores = list(map(lambda d: score(os.path.join(logs_base_dir, d)), dirs))
    dir_score_tuple = zip(dirs, scores)
    f_score = final_score(scores)

    with open(os.path.join(logs_base_dir, scores_filename), "w") as f:
        for (d, mean_score) in dir_score_tuple:
            f.write(f"{d},{mean_score}\n")
        f.write(f"Final score,{f_score}")


def main():
    parser = argparse.ArgumentParser(description="Calculate the scores from the logs")

    parser.add_argument(
        "logs_path",
        type=str,
        help="The base folder which contains the level's log folders",
    )

    args = parser.parse_args()

    log_scores(args.logs_path)
    print(
        f"\nCreated {scores_filename} ({os.path.join(args.logs_path, scores_filename)})"
    )


if __name__ == "__main__":
    main()
