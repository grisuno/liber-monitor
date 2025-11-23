"""
setup.py v1.0.0
================
Instalador profesional para liber-monitor.
Formato estándar pip install compatible.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="liber-monitor",
    version="1.0.0",
    author="RESMA Project",
    author_email="contact@resma.ai",
    description="Geometric diagnostic tool for neural networks - Detect overfitting 2-3 epochs early",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/grisuno/liber-monitor",
    project_urls={
        "Bug Tracker": "https://github.com/grisuno/liber-monitor/issues",
        "Source": "https://github.com/grisuno/liber-monitor",
        "Research": "https://github.com/grisuno/resma",
        "Documentation": "https://github.com/grisuno/liber-monitor#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.3.0",  # Para visualización
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "experiments": [
            "torchvision>=0.10.0",
            "networkx>=2.6",  # Para topología en versiones futuras
        ]
    },
    keywords="neural-networks, overfitting, monitoring, geometry, entropy, singular-values, early-stopping",
    license="GPL-3.0",
    zip_safe=False,
    include_package_data=True,
)
