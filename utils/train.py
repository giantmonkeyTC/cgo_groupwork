import sys
from pathlib import Path
from warnings import warn
from abc import ABCMeta, abstractmethod
sys.path.append(
    str(Path(__file__).parent)
)

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from early_stop import EarlyStopper, PseudoEarlyStopper

class Trainer(metaclass=ABCMeta):
    def __init__(
        self,
        device="cuda:0",
    ) -> None:
        
        if not torch.cuda.is_available() and device != "cpu":
            warn(f"cuda is not available. Train on CPU")
            self.device = "cpu"
        else:
            self.device = device

    def train(
        self,
        model: nn.Module, 
        optimizer: Optimizer, 
        criterion: nn.Module, 
        train_dataset: Dataset,
        epoch: int,
        batch_size: int,
        shuffle=True,
        early_stopper=None,
        save_path=None,
        verbose=False,
        **kwargs
    ) -> list:
        r"""train method arguments
        - model (torch.nn.Module): model to train
        - optimizer (torch.optim.Optimizer): optimizer
        - criterion (torch.nn.Modlue): loss function
        - train_dataset (torch.utils.data.Dataset): dataset for training
        - epoch (int): training epochs
        - batch_size (int): size of mini batch
        - shuffle (bool): shuffling dataset or not (defalut=True)
        - early_stopper (EarlyStopper): early stopper objects. if None, not using early stopping (defalut=None)
        - save_path (path like object): save path of model's checkpoint when not using early stopping. if None, save model current directory (defalut=None)
        - verbose (bool): show training process or not (defalut=False)
        - **kwargs (dict): arguments of Dataloader
        """
        model.to(self.device)
        model.train()
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        pbar = tqdm(initial=0, total=epoch, desc=f"training")

        if early_stopper is None:
            if save_path is None:
                save_path = Path.cwd() / "checkpoint_model.pt"
            early_stopper = PseudoEarlyStopper(verbose=verbose, save_path=save_path)

        losses = []
        # エポックごとに学習
        for e in range(epoch):
            running_loss = 0.0

            # ミニバッチ学習
            for n, batch in enumerate(train_dataloader):
                loss = self.train_batch(batch, model, optimizer, criterion)
                running_loss += loss.item()
            
            running_loss /= n
            losses.append(running_loss)
            if verbose:
                print("---")
                print(f"Epoch {e + 1}: current loss: {running_loss}")

            early_stopper(running_loss, model)
            if early_stopper.early_stop:
                print(f"Early stop")
                break

            pbar.update(1)

        return losses

    @abstractmethod
    def train_batch(self, batch: torch.Tensor, model: nn.Module, optimizer: Optimizer, criterion: nn.Module) -> torch.Tensor:
        pass

if __name__ == "__main__":
    cwd = Path.cwd()
    file = Path(__file__)

    print(cwd)
    print(file)