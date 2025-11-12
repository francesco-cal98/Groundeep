#!/usr/bin/env python3
"""Setup script for GROUNDEEP."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = """
GROUNDEEP: Grounded Deep Learning for Numerosity Processing

A modular analysis pipeline for studying numerosity representations
in deep belief networks trained on visual stimuli.
"""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="groundeep",
    version="1.0.0",
    description="Grounded Deep Learning for Numerosity Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src", "src.*", "pipeline_refactored", "pipeline_refactored.*"]),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "groundeep-analyze=src.main_scripts.analyze_modular:main",
            "groundeep-train=src.main_scripts.train:main",
        ],
    },
)
