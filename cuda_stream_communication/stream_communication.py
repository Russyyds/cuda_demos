import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable, List

def producer(i):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:12345",
        world_size=2,
        rank=0,
    )
    # produce a tensor in GPU i
    data = torch.zeros((1024 * 1024, ), device=f"cuda:{i}")
    # get the information to reconstruct the shared tensor
    func, args = torch.multiprocessing.reductions.reduce_tensor(data)
    args = list(args)
    dist.broadcast_object_list([(func, args)], src=0)
    event = torch.cuda.Event(interprocess=True)
    # launch kernels to modify the tensor
    for i in range(100):
        data += 1
    # record the event in the producer
    event.record()
    dist.broadcast_object_list([event.ipc_handle()], src=0)
    torch.cuda.synchronize()
    # make sure the producer is alive until the consumer finishes
    dist.barrier()


def consumer(j):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:12345",
        world_size=2,
        rank=1,
    )
    torch.cuda.set_device(j)
    recv = [None]
    dist.broadcast_object_list(recv, src=0)
    func: Callable
    args: List
    func, args = recv[0]  # type: ignore
    # `args[6]` is the device id
    # by default pytorch will use `i` from the producer
    # here we need to set it to `j` to test P2P access
    args[6] = j
    data = func(*args)
    recv = [None]
    dist.broadcast_object_list(recv, src=0)
    event_handle = recv[0]
    event = torch.cuda.Event.from_ipc_handle(device=j, handle=event_handle)
    # wait for the producer to finish the kernel
    event.wait()
    assert data.mean().item() == 100
    dist.barrier()

if __name__ == "__main__":
    pi = mp.Process(target=producer, args=(0,))
    pj = mp.Process(target=consumer, args=(1,))
    pi.start()
    pj.start()
    pi.join()
    pj.join()
    assert pi.exitcode == 0 and pj.exitcode == 0