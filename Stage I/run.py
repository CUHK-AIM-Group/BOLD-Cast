import argparse
import os
import random

import numpy as np
import torch

from models.GDA import GDA


DATASET_CONFIG = {
    "UKB": {"time_len": 81},
    "HCP-YA": {"time_len": 83},
    "HCP-D": {"time_len": 85},
    "HCP-A": {"time_len": 75},
    "ABIDE": {"time_len": 30},
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="BOLD-Cast Stage I: common/private latent learning")

    # Dataset / private paths / atlas settings.
    parser.add_argument("--dataset", type=str, choices=DATASET_CONFIG.keys(), default="UKB")
    parser.add_argument("--root_path", type=str, default=None, help="Path to dataset/<dataset>_input/ts containing train/val/test folders.")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--atlas", type=str, default="CC200")
    parser.add_argument("--num_rois", type=int, default=190)
    parser.add_argument("--time_len", type=int,default=None, help="Stage-I self-reconstruction length. Defaults to 30 for ABIDE and 80 otherwise.",)
    parser.add_argument("--normalize", type=str2bool, default=True)

    # Runtime.
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)

    # Original Stage-I model/loss hyperparameters.
    parser.add_argument("--sparse", type=str2bool, default=False)
    parser.add_argument("--isBias", type=str2bool, default=False)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--feature_drop", type=float, default=0.3)
    parser.add_argument("--c_dim", type=int, default=100)
    parser.add_argument("--p_dim", type=int, default=28)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lammbda", type=float, default=1.0)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--inner_epochs", type=int, default=10)
    parser.add_argument("--phi_num_layers", type=int, default=3)
    parser.add_argument("--phi_hidden_size", type=int, default=256)
    parser.add_argument("--hid_units", type=int, default=256)
    parser.add_argument("--decolayer", type=int, default=3)
    parser.add_argument("--sample_neighbor", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.5)

    # Explicit validation control requested for the integrated Stage I.
    parser.add_argument("--lr_reduce_patience", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--min_delta", type=float, default=0.0)

    args = parser.parse_args()

    stage1_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.dirname(stage1_dir)
    if args.root_path is None:
        args.root_path = os.path.join(repo_root, "dataset", f"{args.dataset}_input", "ts")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(stage1_dir, "checkpoints")
    if args.time_len is None:
        args.time_len = DATASET_CONFIG[args.dataset]["time_len"]

    args.checkpoint_path = os.path.join(
        os.path.abspath(args.checkpoint_dir), f"{args.dataset}_best_model.pth"
    )
    # Compatibility with the original Stage-I naming used inside GDA.
    args.save_root = os.path.abspath(args.checkpoint_dir)
    args.model_name = "GDA"
    args.custom_key = "Node"
    args.neighbor_num = args.num_rois

    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config(args):
    print("Stage-I configuration:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print_config(args)

    stage1 = GDA(args)
    stage1.training()
    stage1.extract_latents()


if __name__ == "__main__":
    main()
