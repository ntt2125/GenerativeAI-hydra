import pyrootutils
from omegaconf import DictConfig
import hydra
from typing import Any, Dict, Tuple
from src.models.diffusion_module import DiffusionModule
import torch
import imageio

path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root"
    )
    
config_path = str(path/"configs"/"model")
    
output_path = str(path/"output")
    
print("paths", path, config_path, output_path)

def infer(cfg):
    model = DiffusionModule.load_from_checkpoint(checkpoint_path='/home/ntthong/NTT/2023_learning_skill/Diffusion_Model/diffusion-hydra/logs/diff_2/checkpoints/epoch_000.ckpt')
    
    model = model.cpu()
    gif_shape = [3, 3]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10

    # Generate samples from denoising process
    gen_samples = []
    # x = torch.randn((sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size))
    x = torch.randn((sample_batch_size,1 , 32, 32))
    # device = torch.device("cuda:0")
    # x = x.to(device=device)
    sample_steps = torch.arange(model.t_range-1, 0, -1)
    # sample_steps.to(device=device)
    for t in sample_steps:
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
    
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], 32, 32, 1)

    def stack_samples(gen_samples, stack_dim):
        gen_samples = list(torch.split(gen_samples, 1, dim=1))
        for i in range(len(gen_samples)):
            gen_samples[i] = gen_samples[i].squeeze(1)
        return torch.cat(gen_samples, dim=stack_dim)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = gen_samples.squeeze()
    imageio.mimsave(
        "pred.gif",
        list(gen_samples),
        fps=5,
)

@hydra.main(version_base="1.3", config_path=config_path, config_name="diffusion.yaml")    
def main(cfg: DictConfig):
    print(cfg)
    infer(cfg)
            
main()