import setuptools

setuptools.setup(
    name="janus_physics_discovery",
    version="0.1.0",
    author="Example Author",
    author_email="author@example.com",
    description="A framework for physics discovery using Janus.",
    long_description="""A comprehensive framework for running, tracking, and analyzing physics discovery experiments, supporting various phases of validation protocols with statistical rigor.""",
    long_description_content_type="text/markdown",
    url="https://github.com/example/janus_physics_discovery", # Replace with actual URL
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, adjust if necessary
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8', # Adjust as per project requirements
    install_requires=[
        # List core dependencies here, e.g.:
        # "numpy>=1.20",
        # "pandas>=1.3",
        # "torch>=1.9",
        # "sympy>=1.8",
        # "scipy>=1.7",
        # "matplotlib>=3.4",
        # "seaborn>=0.11",
        # "wandb", # If used directly by the library part, not just scripts
        # "tqdm",
        # Add other specific versions if necessary
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.9",
            "mypy>=0.900",
            "black>=21.0",
            "isort>=5.0",
        ],
    },
    entry_points={
        'janus.experiments': [
            'physics_discovery_example = experiment_runner:PhysicsDiscoveryExperiment',
        ],
        'console_scripts': [
            # Example: 'janus-cli=janus_cli.main:app', # If you have a CLI
        ],
    },
    include_package_data=True, # If you have non-code files in packages (e.g. MANIFEST.in)
)
