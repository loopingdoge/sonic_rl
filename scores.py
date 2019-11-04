import os
import argparse
import numpy as np

from typing import List
from stable_baselines.results_plotter import load_results, ts2xy

from utils import subdirs_list

scores_filename = "scores.csv"


def level_final_score(logs_dir: str) -> int:
    _, score_values = ts2xy(load_results(logs_dir), "timesteps")
    f_score = int(score_values[-1] * 100)
    return f_score
    

def level_score(logs_dir: str) -> float:
    _, score_values = ts2xy(load_results(logs_dir), "timesteps")
    mean_score = int(score_values.mean() * 100)
    return mean_score


def mean_score(mean_scores: List[float]) -> float:
    mean_score = int(np.array(mean_scores).mean())
    return mean_score


def log_scores(logs_base_dir: str):
    """
    Creates a scores.csv file inside the logs_base_dir folder
    """
    dirs = subdirs_list(logs_base_dir)
    scores = list(map(lambda d: level_score(os.path.join(logs_base_dir, d)), dirs))
    final_scores = list(map(lambda d: level_final_score(os.path.join(logs_base_dir, d)), dirs))
    dir_score_tuple = zip(dirs, scores, final_scores)
    score = mean_score(scores)
    f_mean_score = mean_score(final_scores)

    with open(os.path.join(logs_base_dir, scores_filename), "w") as f:
        f.write("Level, Mean Score, Final Score\n")
        for (d, m_score, f_score) in dir_score_tuple:
            f.write(f"{d},{m_score},{f_score}\n")
        f.write(f"Final score,{score},{f_mean_score}")


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
