# setup.py
from setuptools import setup, find_packages

setup(
    name="esn_controller",
    version="0.1.0",
    description="Echo State Network-based feedforward-feedback controller for nonlinear dynamic systems",
    author="Kayode Olalere",
    author_email="kayode.olalere@example.com",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0'
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        '': [
            'docs/figures/*.png',
            'docs/figures/*.yaml',
            'benchmarks/results/*.json',
            'benchmarks/results/*.npy',
            'trained_models/*.npz'
        ]
    }
)