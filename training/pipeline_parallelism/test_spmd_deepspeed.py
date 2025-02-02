#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist

import torchvision
from torchvision import transforms
from torchvision.models import AlexNet
from torchvision.models import vgg19

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader


try:
    import numpy as np
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    from torch_xla import runtime as xr
    import torch_xla.distributed.spmd as xs
    # Import to register the `xla://` init_method
    import torch_xla.distributed.xla_backend
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        _prepare_spmd_partition_spec,
        SpmdFullyShardedDataParallel as FSDPv2,
    )


    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices // 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)

    print("_________________________XLA is Available!")
    XLA_AVAILABLE = True
except:
    print("_________________________XLA is not installed.")
    XLA_AVAILABLE = False



def cifar_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=0,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='gloo',
                        help='distributed backend')
    parser.add_argument('--init-method',
                        type=str,
                        default='xla://',
                        help='init method for distributed')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_base(args):
    torch.manual_seed(args.seed)

    # VGG also works :-)
    #net = vgg19(num_classes=10)
    net = AlexNet(num_classes=10)

    trainset = cifar_trainset(args.local_rank)

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    rank = dist.get_rank()
    gas = engine.gradient_accumulation_steps()

    criterion = torch.nn.CrossEntropyLoss()

    total_steps = args.steps * engine.gradient_accumulation_steps()
    step = 0
    for micro_step in range(total_steps):
        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)

        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()

        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')



def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers


def train_pipe(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #

    # VGG also works :-)
    #net = vgg19(num_classes=10)
    net = AlexNet(num_classes=10)
    net = PipelineModule(layers=join_layers(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = cifar_trainset(args.local_rank)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)
    
    for step in range(args.steps):
        with torch.autograd.set_detect_anomaly(True):
            loss = engine.train_batch()



def main():
    xr.initialize_cache("/dev/shm")

    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend, init_method=args.init_method)
    args.local_rank =  0
    
    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipe(args)



if __name__ == '__main__':
    main()