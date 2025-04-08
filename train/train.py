from pathlib import Path
import torch, os

from tqdm import tqdm
import pickle
import argparse
import logging, datetime
import torch.distributed as dist
from config import MyParser
from steps import trainer
from voicestar.scripts.copy_codebase import copy_codebase


def world_info_from_env():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, global_rank, world_size


if __name__ == "__main__":
    formatter = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    torch.cuda.empty_cache()
    args = MyParser().parse_args()
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"exp_dir: {str(exp_dir)}")

    if args.resume and (
        os.path.exists("%s/bundle.pth" % args.exp_dir)
        or os.path.exists("%s/bundle_prev.pth" % args.exp_dir)
    ):
        if not os.path.exists("%s/bundle.pth" % args.exp_dir):
            os.system(f"cp {args.exp_dir}/bundle_prev.pth {args.exp_dir}/bundle.pth")
        resume = args.resume
        assert bool(args.exp_dir)
        with open("%s/args.pkl" % args.exp_dir, "rb") as f:
            old_args = pickle.load(f)
        new_args = vars(args)
        old_args = vars(old_args)
        for key in new_args:
            if key not in old_args or old_args[key] != new_args[key]:
                old_args[key] = new_args[key]
        args = argparse.Namespace(**old_args)
        args.resume = resume
    else:
        args.resume = False
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)

    # make timeout longer (for generation)
    timeout = datetime.timedelta(seconds=7200)  # 60 minutes

    if args.multinodes:
        _local_rank, _, _ = world_info_from_env()
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, timeout=timeout
        )
    else:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)

    if args.local_wandb:
        os.environ["WANDB_MODE"] = "offline"

    rank = dist.get_rank()
    if rank == 0:
        logging.info(args)
        logging.info(f"exp_dir: {str(exp_dir)}")
    world_size = dist.get_world_size()

    local_rank = int(_local_rank) if args.multinodes else rank
    num_devices = torch.cuda.device_count()
    logging.info(
        f"{local_rank=}, {rank=}, {world_size=}, {type(local_rank)=}, {type(rank)=}, {type(world_size)=}"
    )
    for device_idx in range(num_devices):
        device_name = torch.cuda.get_device_name(device_idx)
        logging.info(f"Device {device_idx}: {device_name}")

    torch.cuda.set_device(local_rank)
    if rank == 0:
        user_dir = os.path.expanduser("~")
        codebase_name = "VoiceStar"
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        copy_codebase(
            os.path.join(user_dir, codebase_name),
            os.path.join(exp_dir, f"{codebase_name}_{now}"),
            max_size_mb=5,
            gitignore_path=os.path.join(user_dir, codebase_name, ".gitignore"),
        )
    my_trainer = trainer.Trainer(args, world_size, rank, local_rank)
    my_trainer.train()
