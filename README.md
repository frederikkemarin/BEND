# 🧬 BEND  - **Ben**chmarking **D**NA Language Models on Biologically Meaningful Tasks


Data is available at https://sid.erda.dk/cgi-sid/ls.py?share_id=eXAmVvbRSW


## Tutorial

### 1. Data format

The data for each task is stored as a `bed` file. This file includes the genomic coordinates for each sample, as well as its split membership and potentially a label. Together with a reference genome, the file is used to extract the DNA sequences for training. Labels that are too complex to be stored in a column in the text-based `bed` file are stored in a `hdf5` file. The two files share their index, so that sample `i` in the `bed` file matches record `i` in the `hdf5` file.


`bed` is a tab-separated format that can be read like a regular table. All our task files include a column `split`, and optionally `label`. If `label` is missing, the labels are found in the `hdf5` file of the same name.
```
chromosome	start	end     split	label
chr1	    1055037	1055849	train	1
chr3	    1070026	1070436	valid	0
```


### 2. Computing embeddings

For training downstream models, it is practical to precompute and save the embeddings to avoid recomputing them at each epoch. As embeddings can grow large when working with genomes, we use TFRecords as the format.
```sh
# Gene finding with DNABERT
python3 scripts/precompute_embeddings.py data/gene_finding.bed temp/dnabert_gene_finding/ dnabert checkpoints/dnabert data/GRCh38.primary_assembly.genome.fa --kmer 6

# Enhancer annotation with ResNet-LM
python3 scripts/precompute_embeddings.py data/enhancers.bed temp/resnetlm_enhancers checkpoints/resnetlm data/GRCh38.primary_assembly.genome.fa checkpoints/tokenizer_bare 
```
#### TODO 
- add exhaustive examples or make bash scripts.
- work out how to actually store the embeddings. script incomplete


If you need to make embeddings for other purposes than preparing downstream task data, [`bend.embedders`](bend/utils/embedders.py) contains wrapper classes around the individual models.


### 3. Evaluating models

#### TODO
- add stuff to bend that is needed to have a train loop running in `scripts/train_on_task.py`
- Trainer
- Dataset class to consume embeddings from prev step
- Metrics