from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cnn-hypertuner",
    version="1.0.0",
    author="CNN HyperTuner",
    description="CNN Hyperparameter Tuning via Soft Computing (GA, PSO, SA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "plotly>=5.0.0",
        "streamlit>=1.20.0",
        "pyyaml",
    ],
    extras_require={
        "viz": ["matplotlib", "seaborn"],
        "dev": ["pytest", "pytest-cov", "twine"],
    },
    entry_points={
        "console_scripts": [
            "cnn-hypertuner=cnn_hypertuner.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/info-gallary/HyperTunner/issues",
        "Source Code": "https://github.com/info-gallary/HyperTunner",
    },
)
