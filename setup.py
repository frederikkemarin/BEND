import os
from setuptools import setup


with open(os.path.join("README.md"), "r") as f:
    readme = f.read()


setup(
    name="bend",
    version="0.1.0",
    description="Benchmark of DNA Language Models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="F. Marin and F. Teufel and M. Horlacher",
    packages=['bend'],
    python_requires=">=3.6, <4",
)