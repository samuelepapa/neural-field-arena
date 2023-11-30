import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from absl import app, flags, logging

from neural_dataset import PreloadedNeFDataset
from neural_dataset.transform.params import param_list_to_vector, param_vector_to_list
from neural_dataset.utils import (
    get_param_keys,
    get_param_structure,
    numpy_collate,
    splits_to_names,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "saved_models/best_psnr/CIFAR10/SIREN", "Path to dataset")
flags.DEFINE_list("split", [40000, 10000, 10000], "splits present in the dataset")
flags.DEFINE_integer("batch_size", 128, "Batch size to use in the data loaders")
flags.DEFINE_integer("num_workers", 0, "Number of workers for each dataset")
flags.DEFINE_string(
    "norm_type", "per_layer", "Type of normalization to use, can be `per_layer` or `global`"
)


def name_to_split(split_name: str) -> Tuple[int, int]:
    start_idx, end_idx = split_name.split("_")
    return float(start_idx), float(end_idx)


def compute_mean_std_for_nef_dataset(
    path: str,
    data_split: Union[Tuple[float, float, float], Tuple[int, int, int]],
    seed=42,
    batch_size: int = 32,
    num_workers: int = 0,
    norm_type="per_layer",
):
    # set the seed for everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    split_start = 0
    loaders = []

    if data_split[0] < 1 or isinstance(data_split[0], float):
        split_type = "fractional"
    else:
        split_type = "exact"

    for split in data_split:
        split_end = split_start + split
        dset = PreloadedNeFDataset(
            path,
            start_idx=split_start,
            end_idx=split_end,
            split_type=split_type,
            data_keys=["params"],
            transform=None,
        )
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        split_start = split_end

    all_split_names = splits_to_names(data_split)
    param_keys = get_param_keys(path)
    param_structure = get_param_structure(path)

    metadata_file = Path(path) / Path("metadata.json")
    if metadata_file.exists():
        # if file is empty, do nothing
        if metadata_file.stat().st_size == 0:
            metadata = {}
        else:
            metadata = json.load(open(metadata_file))
    else:
        metadata = {}

    means = {}
    mean_squares = {}
    sizes = {}

    logging.debug(all_split_names)
    logging.debug([name_to_split(x) for x in all_split_names])

    full_means = {}
    full_stds = {}
    for loader, split_name in zip(loaders, all_split_names):
        logging.debug(f"Split: {split_name}")
        start_idx, end_idx = name_to_split(split_name)
        if split_type == "exact":
            start_idx = int(start_idx)
            end_idx = int(end_idx)

        if split_name in metadata:
            logging.info(
                f"Skipping calculation of mean and std for split `{split_name}` because it is already in metadata."
            )
            continue

        logging.debug(f"Size: {end_idx - start_idx}")
        means[split_name] = {}
        mean_squares[split_name] = {}
        sizes[split_name] = {}
        for batch in loader:
            nef_params = batch
            if norm_type == "per_layer":
                nef_params_list = numpy_collate(
                    [
                        param_vector_to_list(param=nef_param, param_structure=param_structure)
                        for nef_param in nef_params
                    ]
                )

                for param_key, param in zip(param_keys, nef_params_list):
                    if param_key not in means[split_name]:
                        means[split_name][param_key] = []
                        mean_squares[split_name][param_key] = []
                        sizes[split_name][param_key] = []

                    # create a breakpoint if any of the parameters is nan
                    if np.any(np.isnan(param)):
                        print(param_key)
                        print(np.argwhere(np.isnan(nef_params))[:, 0])

                    # if any of the values in param is nan, set it to 0
                    param = np.nan_to_num(param, nan=0.0)

                    means[split_name][param_key].append(np.mean(param))
                    mean_squares[split_name][param_key].append(np.mean(param**2))
                    sizes[split_name][param_key].append(param.shape[0])
            else:
                means[split_name].append(np.mean(nef_params, axis=(0,)))
                mean_squares[split_name].append(np.mean(nef_params**2, axis=(0,)))
                sizes[split_name].append(nef_params.shape[0])

            del nef_params
        logging.info(f"Done with split {split_name}")

        if norm_type == "per_layer":
            full_means[split_name] = []
            full_stds[split_name] = []

            for param_key, param_shape in param_structure:
                mean = np.full(
                    param_shape,
                    np.average(means[split_name][param_key], weights=sizes[split_name][param_key]),
                )
                std = np.full(
                    param_shape,
                    np.sqrt(
                        np.average(
                            mean_squares[split_name][param_key],
                            weights=sizes[split_name][param_key],
                        )
                        - mean**2
                    ),
                )
                full_means[split_name].append(mean)
                full_stds[split_name].append(std)

            metadata[split_name] = {
                "mean": param_list_to_vector(full_means[split_name]).tolist(),
                "std": param_list_to_vector(full_stds[split_name]).tolist(),
            }

        else:
            full_means[split_name] = np.average(
                means[split_name], axis=0, weights=sizes[split_name]
            )
            full_stds[split_name] = np.sqrt(
                np.average(mean_squares[split_name], axis=0, weights=sizes[split_name])
                - full_means[split_name] ** 2
            )

            metadata[split_name] = {
                "mean": full_means[split_name].tolist(),
                "std": full_stds[split_name].tolist(),
            }

    # dictionary to json
    metadata_file.write_text(json.dumps(metadata, indent=4))
    #    metadata_file.write_text(metadata.to_json())

    return metadata


def main(_):
    compute_mean_std_for_nef_dataset(
        path=FLAGS.path,
        data_split=FLAGS.split,
        seed=42,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        norm_type=FLAGS.norm_type,
    )


if __name__ == "__main__":
    app.run(main)
