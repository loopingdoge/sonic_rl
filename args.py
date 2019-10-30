import argparse


def add_trailing_slash(path):
    if path[-1] != "/":
        return path + "/"
    else:
        return path


def finalize_args(parser):
    args = parser.parse_args()

    args.save_dir = add_trailing_slash(args.save_dir)
    args.logs_dir = add_trailing_slash(args.logs_dir)

    return args


def common_args(parser):
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

    parser.add_argument(
        "--algo",
        default="ppo2",
        choices=["a2c", "ppo2"],
        help="algorithm to use: a2c | ppo2",
    )

    parser.add_argument(
        "--policy",
        default="cnn",
        choices=["cnn", "cnnlstm"],
        help="algorithm to use: cnn | cnnlstm",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="how many training CPU processes to use (default: 4)",
    )
    
    parser.add_argument(
        "--hyper-opt",
        action="store_true",
        default=False,
        help="set it to use the optimal hyperparameters",
    )
    
    parser.add_argument(
        "--no-short-life",
        action="store_true",
        default=False,
        help="whether or not to use ShortLife wrapper"
    )


def get_test_args():
    parser = argparse.ArgumentParser(
        description="Sonic's reinforcement learning testing suite"
    )

    common_args(parser)

    parser.add_argument("test_id", help="test id (used for the logs' name)")

    return finalize_args(parser)


def get_train_args():
    parser = argparse.ArgumentParser(description="Sonic's reinforcement learning")

    common_args(parser)

    parser.add_argument(
        "--game",
        default="SonicTheHedgehog-Genesis",
        help="game to train on (default: SonicTheHedgehog-Genesis)",
    )

    parser.add_argument(
        "--level",
        default="GreenHillZone.Act1",
        help="lebel to train on (default: GreenHillZone.Act1)",
    )

    parser.add_argument(
        "--joint",
        action="store_true",
        default=False,
        help="train on the full train set",
    )
    parser.add_argument("train_id", help="training id (used for the logs' name)")

    return finalize_args(parser)
