import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
import os

def main(index):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"World size: {world_size}")
    device = xla.device()
    print(f"Device: {device}")
    t = torch.randn(torch.randint(1, 8, (1,)), 4, 144, 720, 1280).to(device)
    print(f"Tensor shape: {t.shape}")
    print(f"Tensor device: {t.device}")
        

if __name__ == '__main__':
        xla.launch(main, args=())