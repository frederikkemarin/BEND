'''
This script is used to precompute the embeddings for a task.
In the following step, the embeddings will be used to train a model.
'''
import argparse
from bend import embedders
from bend.utils import Annotation
from tqdm.auto import tqdm




def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('out_dir', type=str, help='Path to the output directory')
    # model can be any of the ones supported by bend.utils.embedders
    parser.add_argument('model', choices=['nt', 'dnabert', 'awdlstm', 'gpn', 'convnet'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('--extra_context', type=int, default=0, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=3, help = 'Kmer size for the DNABERT model')
    

    args = parser.parse_args()

    # get the embedder
    if args.model == 'nt':
        embedder = embedders.NucleotideTransformerEmbedder(args.checkpoint)
    elif args.model == 'dnabert':
        embedder = embedders.DNABERTEmbedder(args.checkpoint, kmer = args.kmer)
    elif args.model == 'awdlstm':
        embedder = embedders.AWDLSTMEmbedder(args.checkpoint)
    elif args.model == 'gpn':
        embedder = embedders.GPNEmbedder(args.checkpoint)
    elif args.model == 'convnet':
        embedder = embedders.ConvNetEmbedder(args.checkpoint)
    else:
        raise ValueError('Model not supported')
    

    # load the bed file
    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)

    # extend the segments if necessary
    if args.extra_context > 0:
        genome_annotation.extend_segments(args.extra_context)


    # TODO split train, val, test and save separately


    for index, row in tqdm(genome_annotation.annotation.iterrows()):

        dna = genome_annotation.get_dna_segment(index = index)

        # compute the embedding
        embedding = embedder.embed([dna])[0]

