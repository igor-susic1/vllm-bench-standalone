from setuptools import setup, find_packages

setup(
    name="vllm-bench",
    version="0.1.0",
    description="Benchmarking tool for vLLM inference servers",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "regex",
        "aiohttp",
        "numpy",
        "tqdm",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "vllm-bench=vllm.entrypoints.cli.main:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)