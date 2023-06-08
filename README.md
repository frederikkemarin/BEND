# BEND
Benchmarking DNA Language Models on Biologically Meaningful Tasks


Data is available at https://sid.erda.dk/cgi-sid/ls.py?share_id=eXAmVvbRSW


## Tutorial

### Loading data and making embeddings

The data for each task is stored as a `bed` file. This file includes the genomic coordinates for each sample, as well as its split membership and potentially a label. Together with a reference genome, the file is used to extract the DNA sequences for training. Labels that are too complex to be stored in a column in the text-based `bed` file are stored in a `hdf5` file. The two files share their index, so that sample `i` in the `bed` file matches record `i` in the `hdf5` file.


For training downstream models, it is practical to precompute and save the embeddings.
```
# Gene finding with DNABERT
python3 scripts/precompute_embeddings.py data/gene_finding.bed temp/dnabert_gene_finding/ dnabert checkpoints/dnabert data/GRCh38.primary_assembly.genome.fa --kmer 6

# Variant effects with ResNet-LM
python3 scripts/precompute_embeddings.py data/variant_effects.bed temp/resnetlm_variant_effects checkpoints/resnetlm data/GRCh38.primary_assembly.genome.fa checkpoints/tokenizer_bare --extra_context 256
```
#### TODO 
- add exhaustive examples or make bash scripts.
- work out how to actually store the embeddings. script incomplete
- put tokenizers in the repo 


### Evaluating models

#### TODO
- add stuff to bend that is needed to have a train loop running in `scripts/train_on_task.py`