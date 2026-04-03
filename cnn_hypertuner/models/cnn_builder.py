"""
CNN Model Builder — Constructs a PyTorch CNN from a hyperparameter config dict.
Supports variable depth, filter sizes, activations, optimizers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple


ACTIVATIONS = {
    "relu":       nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.1),
    "elu":        nn.ELU(),
    "gelu":       nn.GELU(),
}


def get_activation(name: str) -> nn.Module:
    return ACTIVATIONS.get(name, nn.ReLU())


class FlexCNN(nn.Module):
    """
    Flexible CNN that builds itself from a hyperparameter config.
    
    Config keys used:
      num_conv_layers, num_filters_1/2/3, kernel_size, activation,
      dropout_rate, dense_units
    """

    def __init__(self, config: Dict[str, Any], input_shape: Tuple[int, int, int], num_classes: int):
        super().__init__()
        self.config = config
        C, H, W = input_shape
        n_layers = int(config["num_conv_layers"])
        ks = int(config["kernel_size"])
        padding = ks // 2
        dropout = float(config["dropout_rate"])
        act_name = config.get("activation", "relu")

        filter_counts = [
            int(config.get(f"num_filters_{i+1}", 64))
            for i in range(min(n_layers, 3))
        ]
        # If more than 3 layers, double last filter count
        while len(filter_counts) < n_layers:
            filter_counts.append(filter_counts[-1] * 2)

        # Build conv blocks
        layers = []
        in_channels = C
        for i in range(n_layers):
            out_channels = min(filter_counts[i], 512)
            layers += [
                nn.Conv2d(in_channels, out_channels, ks, padding=padding),
                nn.BatchNorm2d(out_channels),
                get_activation(act_name),
            ]
            if i < n_layers - 1:
                layers.append(nn.MaxPool2d(2, 2))
                H, W = H // 2, W // 2
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout * 0.5))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.features = nn.Sequential(*layers)

        flat_size = in_channels * 4 * 4
        dense_units = int(config["dense_units"])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, dense_units),
            get_activation(act_name),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Build the optimizer from config."""
    lr = float(config["learning_rate"])
    wd = float(config.get("weight_decay", 1e-4))
    name = config.get("optimizer", "adam")

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
