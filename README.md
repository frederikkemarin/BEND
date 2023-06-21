# 🧬 BEND  - **Ben**chmarking **D**NA Language Models on Biologically Meaningful Tasks

![Stars](https://img.shields.io/github/stars/frederikkemarin/BEND?logo=GitHub&color=yellow)

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

If you need to make embeddings for other purposes than preparing downstream task data, [`bend.embedders`](bend/utils/embedders.py) contains wrapper classes around the individual models.


### 3. Evaluating models

#### Training supervised models

It is first required that the above step (computing the embeddings) is completed.
The embeddings should afterwards be located in `BEND/data/{task_name}/{embedder}/*tfrecords`

To run a runstream task run (from ```BEND/```):
```
python scripts/train_on_task.py --config-path conf/supervised_tasks/{task_name}/ --config-name {embedder}
```
E.g. to run gene finding on the convnet embeddings the commandline is then:
```sh
python3 scripts/train_on_task.py --config-path conf/supervised_tasks/gene_finding/ --config-name convnet
```
The full list of current task names are : 

- `gene_finding`
- `enhancer_annotation`
- `variant_effects`
- `histone_modification`
- `chromatin_accesibility`

And the list of available embedders/models used for training on the tasks are : 

- `awdlstm`
- `convnet`
- `nt_transformer_ms`
- `nt_transformer_human_ref`
- `dnabert6` 
- `convnet_supervised`
- `onehot`
- `nt_transformer_1000g`

The results of a run can be found at :
```
BEND/downstream_tasks/{task_name}/{embedder}/
```
If desired, the config files can be modified to change parameters, output/input directory etc.

#### Unsupervised tasks

For unsupervised prediction of variant effects, embeddings don't have to be precomputed and stored. Embeddings are generated and directly evaluated using

```bash
python3 scripts/predict_variant_effects.py variant_effects.bed {output_file_name}.csv {model_type} {path_to_checkpoint} {path_to_reference_genome_fasta} --embedding_idx {position_of_embedding}
```

A notebook with an example of how to run the script and evaluate the results can be found in [examples/unsupervised_variant_effects.ipynb](notebooks/variant_effects.ipynb).

------------
## Extending BEND

### Adding a new embedder

All embedders are defined in [bend/utils/embedders.py](bend/utils/embedders.py) and inherit `BaseEmbedder`. A new embedder needs to implement `load_model`, which should set up all required attributes of the class and handle loading the model checkpoint into memory. It also needs to implement `embed`, which takes a list of sequences, and returns a list of embedding matrices formatted as numpy arrays. The `embed` method should be able to handle sequences of different lengths.

### Adding a new task
As the first step, the data for a new task needs to be formatted in the [bed-based format](#1-data-format). If necessary, a `split` and `label` column should be included. The next step is to add new config files to `conf/supervised_tasks`. You should create a new directory named after the task, and add a config file for each embedder you want to evaluate. The config files should be named after the embedder.


-------------

## Citation Guidelines

The datasets included in BEND were collected from a variety of sources. When you use any of the datasets, please ensure to correctly cite the respective original publications describing each dataset.

### Gene finding ([GENCODE](https://www.gencodegenes.org/))

    @article{frankish_gencode_2021,
	title = {{GENCODE} 2021},
	volume = {49},
	issn = {0305-1048},
	url = {https://doi.org/10.1093/nar/gkaa1087},
	doi = {10.1093/nar/gkaa1087},
	number = {D1},
	urldate = {2022-09-26},
	journal = {Nucleic Acids Research},
	author = {Frankish, Adam and Diekhans, Mark and Jungreis, Irwin and Lagarde, Julien and Loveland, Jane E and Mudge, Jonathan M and Sisu, Cristina and Wright, James C and Armstrong, Joel and Barnes, If and Berry, Andrew and Bignell, Alexandra and Boix, Carles and Carbonell Sala, Silvia and Cunningham, Fiona and Di Domenico, Tomás and Donaldson, Sarah and Fiddes, Ian T and García Girón, Carlos and Gonzalez, Jose Manuel and Grego, Tiago and Hardy, Matthew and Hourlier, Thibaut and Howe, Kevin L and Hunt, Toby and Izuogu, Osagie G and Johnson, Rory and Martin, Fergal J and Martínez, Laura and Mohanan, Shamika and Muir, Paul and Navarro, Fabio C P and Parker, Anne and Pei, Baikang and Pozo, Fernando and Riera, Ferriol Calvet and Ruffier, Magali and Schmitt, Bianca M and Stapleton, Eloise and Suner, Marie-Marthe and Sycheva, Irina and Uszczynska-Ratajczak, Barbara and Wolf, Maxim Y and Xu, Jinuri and Yang, Yucheng T and Yates, Andrew and Zerbino, Daniel and Zhang, Yan and Choudhary, Jyoti S and Gerstein, Mark and Guigó, Roderic and Hubbard, Tim J P and Kellis, Manolis and Paten, Benedict and Tress, Michael L and Flicek, Paul},
	month = jan,
	year = {2021},
	pages = {D916--D923},
    }

### Chromatin accessibility ([ENCODE](https://www.encodeproject.org/))
### Histone modification ([ENCODE](https://www.encodeproject.org/))

    @article{noauthor_integrated_2012,
	title = {An {Integrated} {Encyclopedia} of {DNA} {Elements} in the {Human} {Genome}},
	volume = {489},
	issn = {0028-0836},
	url = {https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3439153/},
	doi = {10.1038/nature11247},
	number = {7414},
	urldate = {2023-05-23},
	journal = {Nature},
	month = sep,
	year = {2012},
	pmid = {22955616},
	pmcid = {PMC3439153},
	pages = {57--74},
    }


### Enhancer annotation ([Fulco et al.](https://www.nature.com/articles/s41588-019-0538-0), [Gasperini et al.](https://www.sciencedirect.com/science/article/pii/S009286741831554X), [Avsec et al.](https://www.nature.com/articles/s41592-021-01252-x) )

**Enhancers**

    @article{fulco_activity-by-contact_2019,
    title = {Activity-by-contact model of enhancer–promoter regulation from thousands of {CRISPR} perturbations},
    volume = {51},
    copyright = {2019 The Author(s), under exclusive licence to Springer Nature America, Inc.},
    issn = {1546-1718},
    url = {https://www.nature.com/articles/s41588-019-0538-0},
    doi = {10.1038/s41588-019-0538-0},
    language = {en},
    number = {12},
    urldate = {2023-05-23},
    journal = {Nature Genetics},
    author = {Fulco, Charles P. and Nasser, Joseph and Jones, Thouis R. and Munson, Glen and Bergman, Drew T. and Subramanian, Vidya and Grossman, Sharon R. and Anyoha, Rockwell and Doughty, Benjamin R. and Patwardhan, Tejal A. and Nguyen, Tung H. and Kane, Michael and Perez, Elizabeth M. and Durand, Neva C. and Lareau, Caleb A. and Stamenova, Elena K. and Aiden, Erez Lieberman and Lander, Eric S. and Engreitz, Jesse M.},
    month = dec,
    year = {2019},
    note = {Number: 12
    Publisher: Nature Publishing Group},
    keywords = {Epigenetics, Epigenomics, Functional genomics, Gene expression, Gene regulation},
    pages = {1664--1669},
    }

**Enhancers**

    @article{gasperini_genome-wide_2019,
    title = {A {Genome}-wide {Framework} for {Mapping} {Gene} {Regulation} via {Cellular} {Genetic} {Screens}},
    volume = {176},
    issn = {0092-8674},
    url = {https://www.sciencedirect.com/science/article/pii/S009286741831554X},
    doi = {10.1016/j.cell.2018.11.029},
    language = {en},
    number = {1},
    urldate = {2023-05-23},
    journal = {Cell},
    author = {Gasperini, Molly and Hill, Andrew J. and McFaline-Figueroa, José L. and Martin, Beth and Kim, Seungsoo and Zhang, Melissa D. and Jackson, Dana and Leith, Anh and Schreiber, Jacob and Noble, William S. and Trapnell, Cole and Ahituv, Nadav and Shendure, Jay},
    month = jan,
    year = {2019},
    keywords = {CRISPR, CRISPRi, RNA-seq, crisprQTL, eQTL, enhancer, gene regulation, genetic screen, human genetics, single cell},
    pages = {377--390.e19},
    }


**Transcription start sites**

    @article{avsec_effective_2021,
    title = {Effective gene expression prediction from sequence by integrating long-range interactions},
    volume = {18},
    copyright = {2021 The Author(s)},
    issn = {1548-7105},
    url = {https://www.nature.com/articles/s41592-021-01252-x},
    doi = {10.1038/s41592-021-01252-x},
    language = {en},
    number = {10},
    urldate = {2023-05-23},
    journal = {Nature Methods},
    author = {Avsec, Žiga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R. and Grabska-Barwinska, Agnieszka and Taylor, Kyle R. and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R.},
    month = oct,
    year = {2021},
    note = {Number: 10
    Publisher: Nature Publishing Group},
    keywords = {Gene expression, Machine learning, Software, Transcriptomics},
    pages = {1196--1203},
    }


### Noncoding Variant Effects ([DeepSEA](https://www.nature.com/articles/nmeth.3547))
DeepSEA's data was sourced from [GRASP](https://grasp.nhlbi.nih.gov/Overview.aspx) and the [1000 Genomes Project](https://www.internationalgenome.org/), which should also be attributed accordingly.

    @article{zhou_predicting_2015,
	title = {Predicting effects of noncoding variants with deep learning–based sequence model},
	url = {https://www.nature.com/articles/nmeth.3547},
	doi = {10.1038/nmeth.3547},
	language = {en},
	number = {10},
	urldate = {2023-06-07},
	journal = {Nature Methods},
	author = {Zhou, Jian and Troyanskaya, Olga G},
	year = {2015},
    }
