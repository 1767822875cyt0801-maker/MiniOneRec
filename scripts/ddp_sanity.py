import os

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
        gpu_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        backend = "gloo"
        gpu_name = "cpu"

    dist.init_process_group(backend=backend)
    value = torch.tensor([rank], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    print(
        "DDP sanity "
        f"rank={rank} local_rank={local_rank} world_size={world_size} "
        f"cuda_device_count={torch.cuda.device_count()} "
        f"device={device} gpu_name={gpu_name} "
        f"all_reduce_rank_sum={value.item()}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
