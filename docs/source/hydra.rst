Configuring BEND using hydra
============================

BEND's embedding generation, model training and evaluation workflow is configured
using `Hydra <https://hydra.cc/>`_ .

If you want to extend BEND to either use a different model for embedding generation, or train
supervised models on new tasks and datasets, you can do so by creating new Hydra configuration files.

Running new embedders
*********************

To run a new embedder, you should extend the `conf/embedding/embed.yaml <https://github.com/frederikkemarin/BEND/tree/main/conf/embedding/embed.yaml>`_ file. 
In order to add the new embedder, and embedder class must be used. 
The configuration for the new embedder class is added to the conf/embedding/embed.yaml file following this example (the values can of course be modified):

`embedder_name:
  _target_ : bend.utils.embedders.NewEmbedderClass
  arg_1 : value_1
  arg_2 : value_2`
 
The `embedder_name` will be the name under which the embeddings are saved in the output directory (`data/task_name/embedder_name/`).
The `arg_1` and `arg_2` are arguments that are passed to the `NewEmbedderClass` when it is initialized, these arguments and their values are dependent on the embedding class. 

The `embedder_name` must also be added in the `conf/embedding/embed.yaml` under models:
`models:
  - embedder_name
  - other_embedder_name`

Running new dataset
*********************
To run a new dataset, you should extend the `conf/data/dataset.yaml`
The data must consist of bed file. Label values must either be in the bed file or in a separate hdf5 file.

The bed file should contain a header and the following columns: `chrom`, `start`, `end`, `strand`, `split` (and optionally `label`).
If the `label` column is used it should contain comma-seperated values for the given region (e.g. 1,2,3).

The `split` column is used to split the data into train, validation and test sets (or other desired splits).

If the to only generate labels for a certain split, specify this in the `conf/embedding/embed.yaml` file under `splits`:
`splits : 
  - train
  - valid 
  - test`

If `splits` is set to `null`, all splits in the `split` column will be generated. 

Add the name of the new dataset/task under:

`tasks : 
  - new_task_name 
  - enhancer_annotation`

`new_task_name:
  reference_fasta : path_to_reference_fasta
  hdf5_file : path_to_label_file # optional, labels must either be contained here or in the bed file
  bed : path_to_bed_file
  read_strand : True # if True read the strand from the bed file and get the reverse complimentary of the DNA sequence if the strand is negative
  label_depth : 1 # number of possible labels in a multilabel situation (optional, only add if labels are contained in the bed file)`

