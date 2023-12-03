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
            
            # self.input_seq = ((data / 255.0) * 2.0) - 1.0
            self.input_seq = (data * 2.0) - 1.0 # (-1.0, 1.0)
            # min_value = torch.min(self.input_seq)
            # max_value = torch.max(self.input_seq)
            
            # print(f'min: {min_value.item()}')
            # print(f'max: {max_value.item()}')
            
            print(f"input_seq_shape: {self.input_seq.shape}")
            self.input_seq = self.input_seq.moveaxis(3, 1)
            print(f"input_seq_shape after: {self.input_seq.shape}")
            self.data_train, self.data_val, _ = random_split(
                dataset=self.input_seq,
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
    
    def print_min_max(batch_tensor):
        min_value = torch.min(batch_tensor)
        max_value = torch.max(batch_tensor)
            
        print(f'min: {min_value.item()}')
        print(f'max: {max_value.item()}')
        
    
    def get_rand_tensor_with_range(size: tuple, lower_bound: float, upper_bound: float):
        import torch

        # Generate random numbers between 0 and 1
        random_numbers = torch.rand(size=size)

        # Specify the range you want (e.g., between 2 and 7)
        lower_bound = 2
        upper_bound = 7

        # Scale and shift the random numbers to the desired range
        random_numbers_in_range = (upper_bound - lower_bound) * random_numbers + lower_bound
        
        return random_numbers_in_range



    
    def imshow(img):
        
        batch_size = img.shape[0]
        plt.figure(figsize=(8, 8))
        for i in range(batch_size):
            plt.subplot(4, 8, i + 1)
            plt.imshow(img[i].squeeze().numpy(), cmap='gray')
            plt.axis('off')

        plt.show()
    
    def test_datamodule(cfg: DictConfig):
        datamodule: DIFF_Datamodule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        # print(type(loader))
        
        bx= next(iter(loader))
        
        print(f'type bx: {type(bx)}')
        
        print_min_max(bx)
        
        
        imshow(bx)
        
        print("n_batch", len(loader), bx.shape, )
        
        for bx in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")
        
        for bx in tqdm(datamodule.val_dataloader()):
            pass
        print("val data passed")
        
        
        #* Plot batch of image
        # imshow(torchvision.utils.make_grid(bx))
    
    @hydra.main(version_base = "1.3", config_path=config_path, config_name="diff_dataset.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_datamodule(cfg)
        # x=get_rand_tensor_with_range((32,1,32,32), 300, 301)
        # imshow(x)
    
    main()