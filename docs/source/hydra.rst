Configuring BEND using hydra
============================

BEND's embedding generation, model training and evaluation workflow is configured
using `Hydra <https://hydra.cc/>`_ .

If you want to extend BEND to either use a different model for embedding generation, or train
supervised models on new tasks and datasets, you can do so by creating new Hydra configuration files.

Running new embedders
*********************

To run a new embedder, you should extend the `conf/embedding/embed.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/embedding/embed.yaml>`_ file. This file



