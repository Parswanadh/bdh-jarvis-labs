"""BDH — Bio-Distilled Hebbian Network.

Brain-inspired linear attention with multi-scale Hebbian memory.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bdh",
    version="0.2.0",
    author="BDH Team",
    description="Brain-inspired linear attention with multi-scale Hebbian memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/bdh",
    packages=find_packages(include=["bdh", "bdh.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
