Configuring BEND using hydra
============================

BEND's embedding generation, model training and evaluation workflow is configured
using `hydra <https://hydra.cc/>`_ .

If you want to extend BEND to either use a different model for embedding generation, or train
supervised models on new tasks and datasets, you can do so by creating new Hydra configuration files.

Running new embedders
*********************

First, an embedder needs to be implemented as laid out in the tutorial on adding new embedders. To run a new embedder on tasks, you should extend the `conf/embedding/embed.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/embedding/embed.yaml>`_ file. 
In order to add the new embedder, and embedder class must be used. 
The configuration for the new embedder class is added to the conf/embedding/embed.yaml file following this example (the values can of course be modified):

.. code-block::

    embedder_name:
    _target_ : bend.utils.embedders.NewEmbedderClass
    arg_1 : value_1
    arg_2 : value_2
 
The ``embedder_name`` will be the name under which the embeddings are saved in the output directory (``data/task_name/embedder_name/``).
The `arg_1` and `arg_2` are arguments that are passed to the ``NewEmbedderClass`` when it is initialized, these arguments and their values are dependent on the ``load_model`` method of
the embedding class. E.g. typically, ``arg_1`` will be a name or path of the model. ``arg_2`` could be e.g. a configuration argument of the tokenizer. Most embedders only need a name/path.

The ``embedder_name`` must also be added in the `conf/embedding/embed.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/embedding/embed.yaml>`_ under ``models``:

.. code-block::

    models:
    - resnetlm
    - awdlstm
    - nt_transformer_ms
    - embedder_name
    - other_embedder_name

Running new datasets
*********************

For running a new dataset, we need to configure both the embedding generation and the model training pipelines that are called 
by ``precompute_embeddings.py`` and ``train_on_task.py`` respectively.

To run a new dataset, you should extend the 
`conf/data/dataset.yaml  <https://github.com/frederikkemarin/BEND/tree/main/conf/data/dataset.yaml>`_ file.
Like the existing BEND datasets, data must be formatted as ``bed`` files. Label values must either be in the ``bed`` file or in a separate hdf5 file.

The ``bed`` file should contain a header row and the following columns: ``chrom``, ``start``, ``end``, ``strand``, ``split`` 
(and optionally ``label``).
If the ``label`` column is used, it should contain comma-separated labels of the given region (e.g. ``1,2,5``) in the multilabel case. 

The ``split`` column is used to split the data into ``train``, ``validation`` and ``test`` sets. Alternatively, the 
column can also indicate folds for cross-validation, as seen in BEND's Enhancer annotation task.

The `conf/embedding/embed.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/embedding/embed.yaml>`_ file configures for which 
``splits`` embeddings are generated when running ``precompute_embeddings.py``:

.. code-block::

  splits : 
    - train
    - valid 
    - test

If ``splits`` is set to ``null``, all splits in the ``split`` column will be generated. 

Add the name of the new dataset/task under:

.. code-block::

  tasks : 
    - new_task_name 
    - enhancer_annotation

  new_task_name:
    reference_fasta : path_to_reference_fasta
    hdf5_file : path_to_label_file # optional, labels must either be contained here or in the bed file
    bed : path_to_bed_file
    read_strand : True # if True read the strand from the bed file and get the reverse complimentary of the DNA sequence if the strand is negative
    label_depth : 1 # number of possible labels in a multilabel situation (optional, only add if labels are contained in the bed file)`

