'''
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
'''
import argparse
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
from scipy import spatial


def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('out_file', type=str, help='Path to the output file')
    # model can be any of the ones supported by bend.utils.embedders
    parser.add_argument('model', choices=['nt', 'dnabert', 'awdlstm', 'gpn', 'convnet'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('--extra_context', type=int, default=256, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=3, help = 'Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=0, help = 'Index of the embedding to use for computing the distance')
    # 512: 43 for NT, 254 for DNABERT, 256 for AWDLSTM and ConvNet

    args = parser.parse_args()

    # get the embedder
    if args.model == 'nt':
         embedder = embedders.NucleotideTransformerEmbedder(args.checkpoint)
    elif args.model == 'dnabert':
        embedder = embedders.DNABertEmbedder(args.checkpoint, kmer = args.kmer)
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

    genome_annotation.annotation['distance'] = None

    for index, row in genome_annotation.annotation.iterrows():


        # middle_point = row['start'] + 256


        # index the right embedding with dna[len(dna)//2]
        dna = genome_annotation.get_dna_segment(index = index)

        embedding_wt = embedder.embed([dna])[0]

        dna_alt = [x for x in dna]
        dna_alt[len(dna_alt)//2] = row['alt']
        dna_alt = ''.join(dna_alt)

        embedding_alt = embedder.embed([dna_alt])[0]


        d = spatial.distance.cosine(embedding_alt[0,args.embedding_idx], embedding_wt[0,args.embedding_idx])
        genome_annotation.annotation.loc[index, 'distance'] = d


    genome_annotation.annotation.to_csv(args.out_file)




