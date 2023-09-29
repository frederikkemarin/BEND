Configuring BEND using hydra
============================

BEND's embedding generation, model training and evaluation workflow is configured
using `hydra <https://hydra.cc/>`_.

If you want to extend BEND to either use a different model for embedding generation, or train
supervised models on new tasks and datasets, you can do so by creating new Hydra configuration files.

A Note 
*********************
Please be consistent in the naming of new tasks an embedders across the different configuration files. 
This is required for the code to function correclty.

Running new embedders
*********************

First, an embedder needs to be implemented as laid out in the tutorial on adding new embedders. To run a new embedder on tasks, you should extend the `conf/embedding/embed.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/embedding/embed.yaml>`_ file following the example below.
This config file is used by the ``precompute_embeddings.py`` script to generate embeddings for the different tasks, as shown in the GitHub README.

.. code-block::

    embedder_name:
    _target_ : bend.utils.embedders.NewEmbedderClass
    arg_1 : value_1
    arg_2 : value_2
 
The ``embedder_name`` will be the name under which the embeddings are saved in the output directory (``data/task_name/embedder_name/``).
The ``arg_1`` and ``arg_2`` are arguments that are passed to the ``NewEmbedderClass`` when it is initialized, these arguments and their 
values are dependent on the ``load_model`` method of
the embedding class. Only ``_target_`` is required by the config file, any other arguments are optional to specify.
E.g. typically, ``arg_1`` will be a name or path of the model. ``arg_2`` could be e.g. a configuration argument of the tokenizer. Most embedders only need a name/path.

The ``embedder_name`` must also be added in the `conf/embedding/embed.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/embedding/embed.yaml>`_ under ``models``:

.. code-block::

    models:
    - resnetlm
    - awdlstm
    - nt_transformer_ms
    - embedder_name
    - other_embedder_name

Add the hidden state size of the new embedder to the 
`conf/datadims/embedding_dims.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/datadims/embedding_dims.yaml>`_ file. 
This will be needed for training models on the new embeddings.

Embedding new datasets
**********************

For running a new dataset, we need to configure both the embedding generation and the model training pipelines that are called 
by ``precompute_embeddings.py`` and ``train_on_task.py`` respectively.

To add a new dataset, you should extend the 
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

In this file, add the name of the new dataset/task under ``tasks``, and append a new config entry indicating the files and how to process them:

.. code-block::

  tasks : 
    - new_task_name 
    - enhancer_annotation

  # at the end of the file
  new_task_name:
    reference_fasta : path_to_reference_fasta
    hdf5_file : path_to_label_file # optional, labels must either be contained here or in the bed file
    bed : path_to_bed_file
    read_strand : True # if True, read the strand from the bed file and get the reverse complement of the DNA sequence if the strand is negative
    label_depth : 1 # number of possible labels in a multilabel situation (optional, only add if labels are contained in the bed file)


Add the label dimension of the new task to the `conf/datadims/label_dims.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/datadims/label_dims.yaml>`_ file.
This will be needed for training models on the new task.

Now you can run ``precompute_embeddings.py`` as indicated in the GitHub README!

Running new downstream tasks
****************************


To train models on the new task, you should add a ``new_task`` directory to 
`conf/supervised_tasks <https://github.com/frederikkemarin/BEND/tree/main/conf/supervised_tasks>`_. 
This directory needs to be populated with a config file for each model that should be trained on the task.
Below is an example of one such config file.

.. code-block::

  defaults:
    - datadims : [label_dims,embedding_dims]
    - _self_
  hydra : 
    searchpath:
      - file://conf 
  task : gene_finding # name of the task (should be same as the name of the folder in data_dir that was generated by precompute_embeddings.py)
  embedder : onehot # the name of the embedding model to evaluate
  output_dir: ./downstream_tasks/${task}/${embedder}/ # output directory
  model: # configurations for the downstream model to be used 
    _target_: bend.models.downstream.CNN # train the 2-layer CNN model
    input_size: ${datadims.${embedder}} # we have added this information to the config earlier in the tutorial.
    output_size: ${datadims.${task}} # we have added this information to the config earlier in the tutorial.
    hidden_size: 64
    kernel_size: 3
    upsample_factor: null
  optimizer : 
    _target_ : torch.optim.AdamW 
    lr : 0.003
    weight_decay: 0.01
  data: # data arguments. 
    _target_: bend.utils.data_downstream.get_data
    cross_validation : false
    batch_size : 64
    num_workers : 0
    padding_value : -100
    shuffle : 5000
    data_dir : ./data/${task}/${embedder}/ # directory where the tf reoc
    train_data : [train.tfrecord] # list of tfrecords to be used for training
    valid_data : [valid.tfrecord] # list of tfrecords to be used for validation
    test_data :  [test.tfrecord] # list of tfrecords to be used for testing
    # cross_validation : 1 # which number fold to run for Cross validation (use either this or the above train/test/valid options)
  params: # training arguments
    epochs: 100
    load_checkpoint: false
    mode: train
    gradient_accumulation_steps: 1
    criterion: cross_entropy
    class_weights: null
    metric : mcc #adjust this to the metric you want to use for evaluation. Currenly, AUC, MCC, and AUPRC are implemented.
    activation : none
  wandb:
    mode : disabled 

After having run ``precompute_embeddings.py``, you can run ``train_on_task.py`` as indicated in the GitHub README!
