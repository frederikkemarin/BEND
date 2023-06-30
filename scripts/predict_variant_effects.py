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
    parser.add_argument('model', choices=['nt', 'dnabert', 'awdlstm', 'gpn', 'convnet', 'genalm', 'hyenadna'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('--extra_context', type=int, default=256, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=3, help = 'Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=0, help = 'Index of the embedding to use for computing the distance')

    args = parser.parse_args()

    extra_context_left = args.extra_context
    extra_context_right = args.extra_context

    kwargs = {'disable_tqdm': True}
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
    elif args.model == 'genalm':
        embedder = embedders.GENALMEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'hyenadna':
        embedder = embedders.HyenaDNAEmbedder(args.checkpoint)
        # autogressive model. No use for right context.
        extra_context_left = args.extra_context
        extra_context_right = 0
    else:
        raise ValueError('Model not supported')
    

    # load the bed file
    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)

    # extend the segments if necessary
    if args.extra_context > 0:
        genome_annotation.extend_segments(extra_context_left=extra_context_left, extra_context_right=extra_context_right)

    genome_annotation.annotation['distance'] = None

    for index, row in tqdm(genome_annotation.annotation.iterrows()):


        # middle_point = row['start'] + 256
        # index the right embedding with dna[len(dna)//2]
        dna = genome_annotation.get_dna_segment(index = index)
        dna_alt = [x for x in dna]
        if extra_context_left == extra_context_right:
            dna_alt[len(dna_alt)//2] = row['alt']
        elif extra_context_right == 0:
            dna_alt[-1] = row['alt']
        elif extra_context_left == 0:
            dna_alt[0] = row['alt']
        else:
            raise ValueError('Not implemented')
        dna_alt = ''.join(dna_alt)

        embedding_wt, embedding_alt = embedder.embed([dna, dna_alt], **kwargs)
        d = spatial.distance.cosine(embedding_alt[0, args.embedding_idx], embedding_wt[0, args.embedding_idx])
        genome_annotation.annotation.loc[index, 'distance'] = d


    genome_annotation.annotation.to_csv(args.out_file)




if __name__ == '__main__':
    main()