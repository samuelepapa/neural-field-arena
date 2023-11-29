import json
import os
from glob import glob

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax.core.frozen_dict import FrozenDict
from ml_collections import ConfigDict

from neural_dataset import (
    ClassificationNeFDataset,
    PreloadedNeFDataset,
    build_nef_data_loader_group,
)
from neural_dataset.utils import numpy_collate, start_end_idx_from_path


class TestNeFDatapipe(absltest.TestCase):
    SAVE_PATH = "tests/test_models/"
    NUM_FILES = 10
    NUM_ELEMENTS = 100
    HIDDEN_DIM = 128
    NUM_CLASSES = 10

    def setUp(self):
        os.makedirs(TestNeFDatapipe.SAVE_PATH, exist_ok=True)
        for idx in range(TestNeFDatapipe.NUM_FILES):
            with h5py.File(
                os.path.join(
                    TestNeFDatapipe.SAVE_PATH,
                    f"train_{idx:03d}-{idx+TestNeFDatapipe.NUM_ELEMENTS:03d}.hdf5",
                ),
                "w",
            ) as f:
                f.create_dataset(
                    "params",
                    data=np.random.rand(
                        TestNeFDatapipe.NUM_ELEMENTS, TestNeFDatapipe.HIDDEN_DIM
                    ).astype(np.float32),
                )
                dt = h5py.special_dtype(vlen=str)
                data = f.create_dataset("param_config", (1,), dtype=dt)
                data[0] = json.dumps(("layer1", (TestNeFDatapipe.HIDDEN_DIM,)))
                f.create_dataset(
                    "labels",
                    data=np.random.randint(
                        0, TestNeFDatapipe.NUM_CLASSES, (TestNeFDatapipe.NUM_ELEMENTS, 1)
                    ).astype(np.int32),
                )
        files = glob(os.path.join(TestNeFDatapipe.SAVE_PATH, "*.hdf5"))
        start_end_idxs = [start_end_idx_from_path(path) for path in files]
        start_idx = 0
        end_idx = max([end_idx for _, end_idx in start_end_idxs])
        path_start_end_idxs = zip(files, start_end_idxs)

        path_start_end_list = []

        # add all the paths that contain the start and end idx
        # TODO: might be worth being done with a filter
        if end_idx is None:
            for path, (cur_start_idx, cur_end_idx) in path_start_end_idxs:
                if cur_end_idx > start_idx:
                    path_start_end_list.append(
                        (path, max(start_idx - cur_start_idx, 0)), cur_end_idx - cur_start_idx
                    )
        else:
            for path, (cur_start_idx, cur_end_idx) in path_start_end_idxs:
                if cur_end_idx > start_idx and cur_start_idx < end_idx:
                    path_start_end_list.append(
                        (
                            path,
                            max(start_idx - cur_start_idx, 0),
                            min(end_idx - cur_start_idx, cur_end_idx - cur_start_idx),
                        )
                    )

        self.path_start_end_iter = path_start_end_list

    def test_standard_datapipe(self):
        datapipe = PreloadedNeFDataset(path=TestNeFDatapipe.SAVE_PATH, data_keys=["params"])
        idx = 0
        for data_element in datapipe:
            self.assertIsInstance(data_element, dict)
            self.assertIn("params", data_element)
            self.assertEqual(len(data_element), 1)
            self.assertEqual(data_element["params"].ndim, 1)
            self.assertEqual(data_element["params"].dtype, np.float32)
            idx += 1
        self.assertGreater(idx, 0)

    def test_classification_datapipe(self):
        datapipe = ClassificationNeFDataset(path=TestNeFDatapipe.SAVE_PATH)
        for data_element in datapipe:
            self.assertIsInstance(data_element, dict)
            self.assertIn("params", data_element)
            self.assertEqual(len(data_element), 2)
            self.assertEqual(data_element["params"].ndim, 1)
            self.assertEqual(data_element["params"].dtype, np.float32)
            self.assertEqual(data_element["labels"].shape, (1,))
            self.assertEqual(data_element["labels"].dtype, np.int32)

    def tearDown(self):
        for path in glob(os.path.join(TestNeFDatapipe.SAVE_PATH, "*.hdf5")):
            os.remove(path)
        os.removedirs(TestNeFDatapipe.SAVE_PATH)


class TestNeFDataLoaders(absltest.TestCase):
    SAVE_PATH = "tests/test_models/"
    NUM_TRAIN_FILES = 4
    NUM_VAL_FILES = 2
    NUM_TEST_FILES = 2
    NUM_ELEMENTS = 10
    HIDDEN_DIM = 128

    def setUp(self):
        os.makedirs(TestNeFDataLoaders.SAVE_PATH, exist_ok=True)
        global_id = 0
        for mode, num_files in zip(
            ["train", "val", "test"],
            [
                TestNeFDataLoaders.NUM_TRAIN_FILES,
                TestNeFDataLoaders.NUM_VAL_FILES,
                TestNeFDataLoaders.NUM_TEST_FILES,
            ],
        ):
            for idx in range(num_files):
                with h5py.File(
                    os.path.join(
                        TestNeFDataLoaders.SAVE_PATH,
                        f"{global_id:02d}-{global_id+TestNeFDataLoaders.NUM_ELEMENTS:02d}.hdf5",
                    ),
                    "w",
                ) as f:
                    f.create_dataset(
                        "params",
                        data=np.random.rand(
                            TestNeFDataLoaders.NUM_ELEMENTS, TestNeFDataLoaders.HIDDEN_DIM
                        ).astype(np.float32),
                    )
                    dt = h5py.special_dtype(vlen=str)
                    data = f.create_dataset("param_config", (1,), dtype=dt)
                    data[0] = json.dumps(("layer1", (TestNeFDatapipe.HIDDEN_DIM,)))
                    labels = idx * TestNeFDataLoaders.NUM_ELEMENTS + np.arange(
                        TestNeFDataLoaders.NUM_ELEMENTS
                    ).astype(np.int32)
                    labels = labels[:, None]
                    f.create_dataset("labels", data=labels.astype(np.int32))
                    global_id += TestNeFDataLoaders.NUM_ELEMENTS

    def test_zero_workers(self):
        config = {
            "path": TestNeFDataLoaders.SAVE_PATH,
            "batch_size": 4,
            "num_workers": 0,
            "preload": True,
            "persistent_workers": False,
            "data_pipe_class": "ClassificationNeFDataset",
            "split": [40, 20, 20],
            "data_prefix": "",
        }
        config = ConfigDict(config)
        modes = ["train", "val", "test"]
        loaders = build_nef_data_loader_group(config, collate_fn=numpy_collate)
        self.assertEqual(len(loaders), 3)
        for mode, loader in zip(modes, loaders):
            for batch in loader:
                self.assertTrue(isinstance(batch, dict))
                self.assertEqual(len(batch), 2)
                self.assertEqual(batch["params"].shape[1], TestNeFDataLoaders.HIDDEN_DIM)
                self.assertEqual(batch["params"].dtype, np.float32)
                self.assertEqual(batch["params"].shape[0], batch["labels"].shape[0])
                if mode == "train":
                    self.assertEqual(batch["params"].shape[0], config.batch_size)
                self.assertEqual(batch["labels"].shape[1], 1)
                self.assertEqual(batch["labels"].dtype, np.int32)

    def test_nonzero_workers(self):
        config = {
            "path": TestNeFDataLoaders.SAVE_PATH,
            "batch_size": 4,
            "num_workers": 3,
            "preload": True,
            "persistent_workers": False,
            "data_pipe_class": "ClassificationNeFDataset",
            "shard_over_files": True,
            "split": [40, 20, 20],
            "data_prefix": "",
        }
        config = ConfigDict(config)
        loaders = build_nef_data_loader_group(config, collate_fn=numpy_collate)
        self.assertEqual(len(loaders), 3)
        modes = ["train", "val", "test"]
        for mode, loader in zip(modes, loaders):
            all_labels = []
            for batch in loader:
                self.assertTrue(isinstance(batch, dict))
                all_labels.append(batch["labels"])

            # reshape to account for batch size in the dataloader
            all_labels = np.concatenate(all_labels, axis=0)
            if mode == "train":
                pass
            else:
                orig_data = []
                all_files = sorted(glob(os.path.join(TestNeFDataLoaders.SAVE_PATH, "*.hdf5")))
                for data_file in all_files:
                    with h5py.File(data_file, "r") as f:
                        orig_data.append(f["labels"][:])
                orig_data = np.concatenate(orig_data, axis=0)
                if mode == "val":
                    orig_data = orig_data[40:60]
                else:
                    orig_data = orig_data[60:]
                self.assertEqual(all_labels.shape[0], orig_data.shape[0])
                self.assertTrue((np.sort(all_labels, axis=0) == orig_data).all())

    def tearDown(self):
        for path in glob(os.path.join(TestNeFDataLoaders.SAVE_PATH, "*.hdf5")):
            os.remove(path)
        os.removedirs(TestNeFDataLoaders.SAVE_PATH)
