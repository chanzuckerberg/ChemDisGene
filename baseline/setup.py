"""
Scripts to generate graphs, train and evaluate graph representations
"""
import fastentrypoints
from setuptools import find_packages, setup

setup(
    name="iesl-re",
    version="0.1",
    package_dir={"": "src"},
    description="",
    install_requires=[
        "Click>=7.1.2",
        "scipy",
        "numpy",
        "xopen",
        "toml",
        "absl-py",
        "torch",
        "transformers",
        "torch",
        "requests",
        "tqdm",
        "scikit-learn",
    ],
    extras_require={
        "wandb": ["wandb"],
    },
)
