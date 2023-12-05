from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from absl import app, flags, logging
from absl.flags import FLAGS
from ml_collections import ConfigDict

from neural_dataset import (
    ClassificationNeuralCIFAR10,
    ClassificationNeuralMNIST,
    ClassificationNeuralShapeNet,
    ClassificationNeuralMicroImageNet,
    compute_mean_std_for_nef_dataset,
    Normalize,
)
from neural_dataset.classification_neural_datasets import get_mean_std
from neural_dataset.utils import torch_collate

# set config flags
flags.DEFINE_string("dataset.path", "saved_models/MNIST/SIREN", "path to dataset")
flags.DEFINE_integer("dataset.batch_size", 128, "batch size")
flags.DEFINE_integer("dataset.num_workers", 0, "number of workers")
flags.DEFINE_integer("dataset.seed", 42, "seed for dataset")
flags.DEFINE_integer("dataset.num_classes", 10, "number of classes")

flags.DEFINE_string("model.name", "MLP", "name of model")
flags.DEFINE_integer("model.params.num_layers", 4, "number of layers")
flags.DEFINE_integer("model.params.hidden_dim", 64, "hidden dimension")

flags.DEFINE_integer("num_epochs", 10, "number of epochs")
flags.DEFINE_integer("seed", 42, "seed for training")

flags.DEFINE_string("backend", "torch", "backend for dataset")



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

def flags_to_config_dict() -> ConfigDict:
    config = ConfigDict()
    for k, v in FLAGS.flag_values_dict().items():
        if "." in k:
            sub_config = config
            keys = k.split(".")
            for key in keys[:-1]:
                if key not in sub_config:
                    sub_config[key] = ConfigDict()
                sub_config = sub_config[key]
            sub_config[keys[-1]] = v
        else:
            config[k] = v
    return config

def train_torch_classifier_on_nef(_):
    config = flags_to_config_dict()

    data_config = config.get("dataset", ConfigDict())

    if "MNIST" in data_config.path:
        dset_class = ClassificationNeuralMNIST
    elif "CIFAR10" in data_config.path:
        dset_class = ClassificationNeuralCIFAR10
    elif "ShapeNet" in data_config.path:
        dset_class = ClassificationNeuralShapeNet
    elif "MicroImageNet" in data_config.path:
        dset_class = ClassificationNeuralMicroImageNet
    else:
        raise ValueError(f"Dataset {data_config.path} not supported")

    try:
        mean, std = get_mean_std(data_config.path, "train")
    except ValueError:
        metadata = compute_mean_std_for_nef_dataset(
            data_config.path,
            [end-start for (start, end) in dset_class.splits.values()],
            split_names=["train", "val", "test"],
            seed=data_config.seed,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
        )
        mean, std = np.array(metadata["train"]["mean"]), np.array(metadata["train"]["std"])

    transform = Normalize(mean, std)

    train_dataset = dset_class(data_config.path, split="train", transform=transform)
    val_dataset = dset_class(data_config.path, split="val", transform=transform)
    test_dataset = dset_class(data_config.path, split="test", transform=transform)
    data_loaders = [
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
        ),
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
        ),
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
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
            nef_weights = nef_weights.to("cuda").float()
            labels = labels.to("cuda").long()
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

        if epoch % 5 == 0:
            model.eval()
            count = 0
            acc = 0
            for batch in val_loader:
                nef_weights, labels = batch["params"], batch["labels"]
                nef_weights = nef_weights.to("cuda").float()
                labels = labels.to("cuda").long()
                output = model(nef_weights)
                loss = criterion(output, labels)
                batch_size = labels.shape[0]
                accuracy = (output.argmax(dim=-1) == labels).float().mean().item()
                acc += accuracy * batch_size
                count += batch_size
            val_acc = acc / count

            logging.info(f"[Epoch {epoch}|{num_epochs}] val acc: {val_acc:.2%}")

    return 0

if __name__ == "__main__":
    app.run(train_torch_classifier_on_nef)