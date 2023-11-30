from neural_dataset.transform.core import Transform, Compose, Identity
from neural_dataset.transform.geometric import (
    RandomRotate,
    RandomScale,
    RandomTranslateMFN,
    RandomTranslateSIREN,
)
from neural_dataset.transform.network import (
    RandomDropout,
    RandomGaussianNoise,
    RandomQuantileWeightDropout,
)
from neural_dataset.transform.params import (
    ListToParameters,
    Normalize,
    ParametersToListMFN,
    ParametersToListSIREN,
    ToTensor,
    UnNormalize,
)
from neural_dataset.transform.permutation import (
    RandomFourierNetWeightPermutation,
    RandomMLPWeightPermutation,
)

AVAILABLE_TRANSFORMATIONS = [
    "Identity",
    "Compose",
    # geometric
    "RandomTranslateMFN",
    "RandomTranslateSIREN",
    "RandomRotate",
    "RandomScale",
    # params
    "ParametersToListMFN",
    "ParametersToListSIREN",
    "ListToParameters",
    "Normalize",
    "UnNormalize",
    # network
    "RandomDropout",
    "RandomQuantileWeightDropout",
    "RandomGaussianNoise",
    # permutation
    "RandomFourierNetWeightPermutation",
    "RandomMLPWeightPermutation",
]
