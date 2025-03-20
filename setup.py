from setuptools import setup, find_packages

setup(
    name="prot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "h5py",
        "pandas",
        "numpy",
        "serpentTools"
    ],
    author="Daan Houben"
)
