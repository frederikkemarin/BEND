import os
from setuptools import setup, find_packages


with open(os.path.join("README.md"), "r") as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="bend",
    version="0.1.0",
    description="Benchmark of DNA Language Models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Anonymous",
    packages=find_packages(),
    python_requires=">=3.6, <=3.11",
    install_requires=requirements,
)