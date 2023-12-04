from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from absl import app, flags, logging
from ml_collections import ConfigDict

from neural_dataset import (
    ClassificationNeuralCIFAR10,
    ClassificationNeuralMNIST,
    ClassificationNeuralShapeNet,
    compute_mean_std_for_nef_dataset,
)


class MLPClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=3, num_classes=10, bn=True):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(num_layers):
            if not bn:
                layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
            else:
                layers.extend(
                    [
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ReLU(),
                    ]
                )

        layers.append(torch.nn.Linear(hidden_dim, num_classes))
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.seq(x)


def set_flags():
    cfg = ConfigDict()
    cfg.seed = 42
    cfg.num_epochs = 10

    cfg.data = ConfigDict()
    cfg.data.path = "saved_models/MNIST/SIREN"
    cfg.data.batch_size = 128
    cfg.data.num_workers = 0
    cfg.data.seed = 42
    cfg.data.split = [55000, 5000, 10000]
    cfg.data.num_classes = 10

    cfg.model = ConfigDict()
    cfg.model.name = "MLP"
    cfg.model.params = ConfigDict()
    cfg.model.params.num_layers = 4
    cfg.model.params.hidden_dim = 64

    cfg.backend = "torch"

    return configdict_to_flags(cfg)


# configdict to FLAGS
def configdict_to_flags(config: ConfigDict, prepend: str = ""):
    for key, value in config.items():
        if isinstance(value, ConfigDict):
            configdict_to_flags(value, prepend=f"{prepend}{key}.")
        else:
            flags.DEFINE_string(f"{prepend}{key}", str(value), f"{key} for {prepend}")


set_flags()


def train_torch_classifier_on_nef(config: ConfigDict):
    data_config = config.get("dataset", ConfigDict())
    metadata = compute_mean_std_for_nef_dataset(
        data_config.path, data_config.split, seed=data_config.seed
    )
    logging.info(f"Computed mean and std for dataset: {metadata}")
    train_dataset = ClassificationNeuralMNIST(data_config.path, split="train", transform=None)
    data_loaders = [
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
        ),
    ]
    train_loader, val_loader, test_loader = data_loaders

    model_config = config.get("model", ConfigDict())
    example = next(iter(train_loader))
    in_dim = example["params"][0].shape[-1]
    model = MLPClassifier(in_dim=in_dim, **model_config.get("params", {}))

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=1e-3, amsgrad=True, weight_decay=5e-4
    )

    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = config.get("num_epochs", 100)
    model = model.to("cuda")
    for epoch in range(1, 1 + num_epochs):
        model.train()
        avg_metrics = defaultdict(list)
        for batch in train_loader:
            nef_weights, labels = batch["params"], batch["labels"]
            nef_weights = torch.tensor(nef_weights, device="cuda", dtype=torch.float32)
            labels = torch.tensor(labels, device="cuda", dtype=torch.long)
            optimizer.zero_grad()
            output = model(nef_weights)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            avg_metrics["loss"].append(loss.item())
            avg_metrics["acc"].append((output.argmax(dim=-1) == labels).float().mean().item())

        for k, v in avg_metrics.items():
            avg_metrics[k] = np.mean(v)
        logging.info(
            f"[Epoch {epoch}|{num_epochs}] train loss: {avg_metrics['loss']:.3f}, train acc: {avg_metrics['acc']:.2%}"
        )

        if epoch % 10 == 0:
            model.eval()
            count = 0
            acc = 0
            for batch in val_loader:
                nef_weights, labels = batch["params"], batch["labels"]
                nef_weights = torch.tensor(nef_weights, device="cuda", dtype=torch.float32)
                labels = torch.tensor(labels, device="cuda", dtype=torch.long)
                output = model(nef_weights)
                loss = criterion(output, labels)
                batch_size = labels.shape[0]
                accuracy = (output.argmax(dim=-1) == labels).float().mean().item()
                acc += accuracy * batch_size
                count += batch_size
            val_acc = acc / count

            logging.info(f"[Epoch {epoch}|{num_epochs}] val acc: {val_acc:.2%}")

    return val_acc
