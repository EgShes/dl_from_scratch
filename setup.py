from setuptools import find_packages, setup

setup(
    name="dl_from_scratch",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["numpy", "pydantic", "torchvision", "scikit-learn"],
)
