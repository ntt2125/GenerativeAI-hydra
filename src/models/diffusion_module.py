from typing import Any, Dict, Tuple
import math
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.Simple_Unet_Transformer import *

class DiffusionModule(LightningModule):
    def __init__(
        self, 
        optimizer: torch.optim.Adam,
        # scheduler: torch.optim.lr_scheduler,
        compile: bool,
        in_size: int=1024, # 32 x 32
        t_range: int = 10000,
        img_depth: int = 1, # 1 for mnist and 3 for cifar
        beta_small: float = 1e-4,
        beta_large: float = 0.02
        ):
        
        
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.beta_small = beta_small #Beta_1
        self.beta_large = beta_large #Beta_T
        self.t_range = t_range
        self.in_size = in_size

        
        
        bilinear = True
        self.inc = DoubleConv(img_depth, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        # print(type(t))
        # print('xxxxxxxxxxxxxxxxxx:', x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)
        return output

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

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
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
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

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