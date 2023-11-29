from neural_dataset.augmentations.core import Augmentation, Compose, Identity
from neural_dataset.augmentations.geometric import (
    RandomRotate,
    RandomScale,
    RandomTranslateMFN,
    RandomTranslateSIREN,
)
from neural_dataset.augmentations.network import (
    RandomDropout,
    RandomGaussianNoise,
    RandomQuantileWeightDropout,
)
from neural_dataset.augmentations.params import (
    ListToParameters,
    Normalize,
    ParametersToListMFN,
    ParametersToListSIREN,
    ToTensor,
    UnNormalize,
)
from neural_dataset.augmentations.permutation import (
    RandomFourierNetWeightPermutation,
    RandomMLPWeightPermutation,
)

AVAILABLE_AUGMENTATIONS = [
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
