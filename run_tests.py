from absl.testing import absltest

from tests.transformations.geometric import TransformationGeometricTest
from tests.transformations.network import TransformationNetworkTest
from tests.transformations.permutation import TransformationPermutationTest
from tests.datapipes import TestNeFDataLoaders, TestNeFDatapipe

if __name__ == "__main__":
    absltest.main()
