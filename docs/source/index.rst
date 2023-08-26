.. BEND documentation master file, created by
   sphinx-quickstart on Sat Aug 26 12:57:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BEND's documentation!
================================

`BEND <https://github.com/frederikkemarin/BEND/>`_ is a benchmark collection for evaluating the performance of DNA language models (LMs).
The BEND codebase serves three purposes:

* Providing a unified interface for computing embeddings from pretrained DNA LMs.
* Extracting sequences from reference genomes using coordinates listed in bed files, and computing embeddings for these sequences for training and evaluating DNA LMs.
* Training lightweight supervised CNN models that use DNA LM embeddings as input, and evaluating their performance on a variety of tasks.

.. automodule:: bend.models
    :members:

.. automodule:: bend.utils
    :members:
    :no-index: embedders



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   hydra
   bend.utils.embedders
   bend.models





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
