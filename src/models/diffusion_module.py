from typing import Any, Dict, Tuple
import math
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.Component_Unet_Transformer import *
import imageio
import wandb

class DiffusionModule(LightningModule):
    def __init__(
        self, 
        optimizer: torch.optim.Adam,
        # scheduler: torch.optim.lr_scheduler,
        compile: bool,
        net: torch.nn.Module,
        in_size: int=1024, # 32 x 32
        t_range: int = 1000,
        img_depth: int = 1, # 1 for mnist and 3 for cifar
        beta_small: float = 1e-4,
        beta_large: float = 0.02,
        path: str = "pre.gif",
        ):
        
        
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.beta_small = beta_small #Beta_1
        self.beta_large = beta_large #Beta_T
        self.t_range = t_range
        self.in_size = in_size
        self.net = net
        self.path = path

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x = x.to(self.device)
        t = t.to(self.device)
        return self.net(x, t)

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def infer(self, path: str):
        gif_shape = [3, 3]
        sample_batch_size = gif_shape[0] * gif_shape[1]
        n_hold_final = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate samples from denoising process
        gen_samples = []
        # x = torch.randn((sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size))
        x = torch.randn((sample_batch_size,1 , 32, 32), device=device)
        # device = torch.device("cuda:0")
        # x = x.to(device=device)
        sample_steps = torch.arange(self.t_range-1, 0, -1)
        sample_steps.to(device=device)
        for t in sample_steps:
            x = self.denoise_sample(x, t)
            if t % 50 == 0:
                gen_samples.append(x)
        for _ in range(n_hold_final):
            gen_samples.append(x)
        gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
        # print_min_max(gen_samples)
        
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
            path,
            list(gen_samples.cpu().numpy()),
            fps=5,
        )

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        x = x.to(self.device)

        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
                z=z.to(self.device)
            else:
                z = 0
            e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return

    def on_validation_epoch_end(self) :
        path = self.path
        self.infer(path)
        wandb.log({"sample": wandb.Image(path)})
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

if __name__=='__main__':
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root"
    )
    
    config_path = str(path/"configs"/"model")
    
    output_path = str(path/"output")
    
    print("paths", path, config_path, output_path)

    def test_module(cfg):
        module = hydra.utils.instantiate(cfg)
        input = 2*torch.rand((32,1,32,32)) - 1
        t = torch.randint(0, 1000, size=(32, 1))
        # net = module.net
        output = module(input, t)
        print(output.shape)
        
    @hydra.main(version_base="1.3", config_path=config_path, config_name="diffusion.yaml")    
    def main(cfg: DictConfig):
        print(cfg)
        test_module(cfg)
            
    main()
    
    # model = DiffusionModel()
    
    # input = 2*torch.rand((32,1,32,32)) - 1
    # t = torch.randint(0, 1000, size=(32, 1))
    # print(input.shape, t.shape)
    # #* torch.Size([32, 1, 32, 32]) torch.Size([32, 1])
    # output = model(input, t)
    # # print(model)
    # # model.eval()
    # print(output.shape)
    #! torch.Size([32, 1, 32, 32])