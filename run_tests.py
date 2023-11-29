from absl.testing import absltest

from tests.augmentations.geometric import AugmentationGeometricTest
from tests.augmentations.network import AugmentationNetworkTest
from tests.augmentations.permutation import AugmentationPermutationTest
from tests.datapipes import TestNeFDataLoaders, TestNeFDatapipe

if __name__ == "__main__":
    absltest.main()
