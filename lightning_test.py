import os
import torch
import torchvision
import time

# pytorch - lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.cifar_train = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.cifar_val = CIFAR10(root='./data', train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=True)
    
class EpochTimeCallback(Callback):
    def __init__(self):
        self.epoch_times = []
        self.init_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {trainer.current_epoch + 1} time: {epoch_time:.2f}s")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_time = self.epoch_times[-1] if self.epoch_times else 0
        epoch_str = f"Epoch {trainer.current_epoch + 1} time: {epoch_time:.2f}s\n"
        
        train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(0)).item()
        val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(0)).item()
        val_accuracy = trainer.callback_metrics.get('val_accuracy', torch.tensor(0)).item()
        result_str = f"Epoch {trainer.current_epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%\n"
        
        with open('lightning_test.txt', 'a') as file:
            file.write(result_str)
            file.write(epoch_str)
    
    def on_train_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.init_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        time_str = f"\n\nTotal training time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s\n"
        with open('lightning_test.txt', 'a') as file:
            file.write(time_str)

class CIFAR10Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 10)  # CIFAR10의 클래스 수는 10개입니다.

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = (outputs.argmax(1) == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return {'val_loss' : loss, 'val_accuracy' : accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    model = CIFAR10Model()
    data_module = CIFAR10DataModule()
    epoch_time_callback = EpochTimeCallback()
    trainer = pl.Trainer(
        max_epochs=20, 
        devices=1, 
        accelerator='gpu', 
        callbacks=[ModelCheckpoint(dirpath='./', filename='best-checkpoint'),
                   epoch_time_callback])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
