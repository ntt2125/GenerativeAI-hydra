from typing import Any, Dict, Optional, Tuple
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import transforms

class DIFF_Datamodule(LightningDataModule):
    def __init__(
        self,
        Data_choice: str = "MNIST",
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int] = (63_000, 7_000),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        Pad: int = 2
    ):
        
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.datasets = {
            "MNIST": MNIST,
            "FashionMNIST": FashionMNIST,
            "CIFAR10": CIFAR10,
        }
        
        # if self.hparams.Data_choice == "MNIST" or self.hparams.Data_choice == "FashionMNIST":
        #     self.transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(self.hparams.Pad)])
        # elif self.hparams.Data_choice == "CIFAR10":
        #     self.transform = transforms.Compose([transforms.ToTensor()])
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        self.batch_size_per_device = batch_size
        
        
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        # if datasets == "MNIST" or datasets == "Fashion":
        #     pad = transforms.Pad(2)
        #     data = pad(train_dataset.data)
        #     data = data.unsqueeze(3)
        #     self.depth = 1
        #     self.size = 32
        # elif datasets == "CIFAR":
        #     data = torch.Tensor(train_datasets.data)
        #     self.depth = 3
        #     self.size = 32
    
    def prepare_data(self) -> None:
        self.datasets[self.hparams.Data_choice](self.hparams.data_dir, train=True, download=True)
        self.datasets[self.hparams.Data_choice](self.hparams.data_dir, train=True, download=False)
        
    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = self.datasets[self.hparams.Data_choice](self.hparams.data_dir, train=True, transform=self.transform)
            testset = self.datasets[self.hparams.Data_choice](self.hparams.data_dir, train=False, transform=self.transform)
            
            dataset = ConcatDataset(datasets=[trainset, testset])
            print(type(dataset))
            
            # data, label = dataset[0]
            # print(data.shape)
            # print(type(data))
            # print(label)
            dataset = torch.cat([data for data, _ in dataset], dim=0)
            print(type(dataset))
            if self.hparams.Data_choice == "MNIST" or self.hparams.Data_choice == "FashionMNIST":
                pad = transforms.Pad(2)
                data = pad(dataset)
                print(data.shape)
                data = data.unsqueeze(3)
                print(data.shape)
                self.depth = 1
                self.size = 32
                
            elif self.hparams.Data_choice == "CIFAR10":
                dataset = torch.Tensor(dataset)
                self.depth = 3
                self.size = 32
            
            self.input_seq = ((data / 255.0) * 2.0) - 1.0
            print(f"input_seq_shape: {self.input_seq.shape}")
            self.input_seq = self.input_seq.moveaxis(3, 1)
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=data,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
        
            
    def train_dataloader(self) :
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self) :
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
            
if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import torchvision
    
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root"
    )
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    
    def imshow(img):
        img = img.permute(0, 2, 3, 1)
    
        # Make a grid of images
        grid = torchvision.utils.make_grid(img, nrow=8)  # nrow controls the number of images in each row of the grid
        
        # Display the grid
        plt.imshow(grid[:, :, 0], cmap='gray')  # Display the first channel of the grid (grayscale)
        plt.show()
    
    def test_datamodule(cfg: DictConfig):
        datamodule: DIFF_Datamodule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        
        bx= next(iter(loader))
        
        # imshow(bx)
        
        print("n_batch", len(loader), bx.shape, )
        
        for bx in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")
        
        for bx in tqdm(datamodule.val_dataloader()):
            pass
        print("val data passed")
        
        
        #* Plot batch of image
        imshow(torchvision.utils.make_grid(bx))
    
    @hydra.main(version_base = "1.3", config_path=config_path, config_name="diff_dataset.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_datamodule(cfg)
    
    main()