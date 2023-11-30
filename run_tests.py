from absl.testing import absltest

from tests.datapipes import TestNeFDataLoaders, TestNeFDatapipe
from tests.transformations.geometric import TransformationGeometricTest
from tests.transformations.network import TransformationNetworkTest
from tests.transformations.permutation import TransformationPermutationTest

if __name__ == "__main__":
    absltest.main()
