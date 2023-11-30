import setuptools

setuptools.setup(
    name="neural_dataset",
    version="0.1",
    author="Samuele Papa",
    author_email="samuele.papa@gmail.com",
    description="A package to load and transform neural datasets",
    url="https://github.com/samuelepapa/neural-field-arena",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "h5py>=3.0.0",
        "numpy>=1.19.5",
        "absl-py>=0.12.0",
    ],
)
