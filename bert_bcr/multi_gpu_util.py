import os
import torch
import torch.distributed as dist

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank=int(os.environ['RANK'])
        args.world_size=int(os.environ['WORLD_SIZE'])
        args.present_gpu=int(os.environ['LOCAL_RANK'])
        print(args.rank)

    args.distributed=True
    torch.cuda.set_device(int(args.gpu_list[args.rank]))
    args.dist_backend='gloo'        #通信后端， nvidia使用NCCL
    dist.init_process_group(backend=args.dist_backend,init_method=args.dist_url,world_size=args.world_size,rank=args.rank)
    dist.barrier()

