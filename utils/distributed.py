import os
import torch
import torch.distributed as dist
from contextlib import contextmanager


def setup_distributed():
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
        dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank():
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    if is_dist_avail_and_initialized():
        return int(os.environ["LOCAL_RANK"])
    return 0


def is_main_process():
    return get_rank() == 0


def on_main_process(function):
    def wrapper(*args, **kwargs):
        if is_main_process():
            return function(*args, **kwargs)
    return wrapper


@contextmanager
def main_process_first():
    if not is_main_process():
        wait_for_everyone()
    yield
    if is_main_process():
        wait_for_everyone()


def wait_for_everyone():
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, op="avg"):
    if is_dist_avail_and_initialized():
        rt = tensor.detach().clone().contiguous()
        if op == "avg":
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= get_world_size()
        elif op == "sum":
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        elif op == "max":
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(rt, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unknown reduce op {op}")
        return rt
    return tensor


def broadcast_tensor(tensor):
    tensor = tensor.detach().clone().contiguous()
    if is_dist_avail_and_initialized():
        dist.broadcast(tensor, src=0)
    return tensor


def broadcast_object(obj):
    if is_dist_avail_and_initialized():
        objects = [obj]
        dist.broadcast_object_list(objects, src=0)
        obj = objects[0]
    return obj


def gather_tensor(tensor):
    tensor = tensor.detach().clone().contiguous()
    if is_dist_avail_and_initialized():
        tensor_list = [torch.ones_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    return [tensor]


def gather_object(obj):
    if is_dist_avail_and_initialized():
        obj_list = [None for _ in range(get_world_size())]
        dist.all_gather_object(obj_list, obj)
        return obj_list
    return [obj]
