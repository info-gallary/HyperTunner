"""
Trainer — Trains a CNN config and returns validation accuracy.
This is the "objective function" used by all soft computing algorithms.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path

from .models.cnn_builder import FlexCNN, build_optimizer

logger = logging.getLogger(__name__)


class CNNTrainer:
    """
    Trains a FlexCNN with a given hyperparameter config.
    Returns validation accuracy as the objective score.
    """

    def __init__(
        self,
        dataset_name: str = "cifar10",           # "cifar10", "mnist", "custom"
        data_root: str = "./data",
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 10,
        n_epochs: int = 5,                        # Short training per evaluation
        val_split: float = 0.2,
        device: Optional[str] = None,
        custom_train_loader: Optional[DataLoader] = None,
        custom_val_loader: Optional[DataLoader] = None,
        early_stopping_patience: int = 3,
    ):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_epochs = n_epochs
        self.val_split = val_split
        self.early_stopping_patience = early_stopping_patience

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self._train_loader = custom_train_loader
        self._val_loader = custom_val_loader

        if custom_train_loader is None:
            self._load_dataset()

        logger.info(f"Trainer ready | dataset={dataset_name} device={self.device} epochs={n_epochs}")

    def _load_dataset(self):
        """Load standard datasets."""
        root = self.data_root
        if self.dataset_name == "cifar10":
            transform_train = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_val = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            full_train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
            val_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_val)
            self.input_shape = (3, 32, 32)
            self.num_classes = 10

        elif self.dataset_name == "mnist":
            transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
            full_train = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
            val_set = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)
            self.input_shape = (1, 28, 28)
            self.num_classes = 10

        else:
            raise ValueError(f"Unknown dataset '{self.dataset_name}'. Use 'custom' and pass loaders.")

        # Sub-sample training set for faster tuning (use 20% of train set per eval)
        n_sub = max(1000, len(full_train) // 5)
        indices = torch.randperm(len(full_train))[:n_sub]
        sub_train = torch.utils.data.Subset(full_train, indices)

        n_val = max(500, len(val_set) // 5)
        val_indices = torch.randperm(len(val_set))[:n_val]
        sub_val = torch.utils.data.Subset(val_set, val_indices)

        self._train_loader = DataLoader(sub_train, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        self._val_loader = DataLoader(sub_val, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    def train_and_evaluate(self, config: Dict[str, Any]) -> float:
        """
        Train a CNN with the given config for n_epochs.
        Returns validation accuracy (0.0 to 1.0).
        """
        try:
            batch_size = int(config.get("batch_size", 64))
            # Re-create loaders with correct batch size
            if hasattr(self._train_loader, 'dataset'):
                train_loader = DataLoader(
                    self._train_loader.dataset, batch_size=batch_size,
                    shuffle=True, num_workers=0, pin_memory=True
                )
            else:
                train_loader = self._train_loader

            model = FlexCNN(config, self.input_shape, self.num_classes).to(self.device)
            optimizer = build_optimizer(model, config)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(self.n_epochs):
                # Training
                model.train()
                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                # Validation
                val_acc = self._validate(model)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    break

            del model
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return best_val_acc

        except Exception as e:
            logger.warning(f"Evaluation failed for config {config}: {e}")
            return 0.0

    def _validate(self, model: nn.Module) -> float:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in self._val_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return correct / total if total > 0 else 0.0

    def get_objective_fn(self) -> Callable[[Dict], float]:
        """Returns a callable objective function for optimizers."""
        return self.train_and_evaluate
