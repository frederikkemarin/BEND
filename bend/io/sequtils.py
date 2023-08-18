import tqdm
import tensorflow as tf
import pysam
from bioio.tf.utils import multi_hot
import pandas as pd
import h5py

# %%
baseComplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

def has_header(file, nrows=20):
    df = pd.read_csv(file, header=None, nrows=nrows, sep='\t')
    df_header = pd.read_csv(file, nrows=nrows, sep='\t')
    return tuple(df.dtypes) != tuple(df_header.dtypes)

# %%
def reverse_complement(dna_string):
    """Returns the reverse-complement for a DNA string."""

    complement = [baseComplement.get(base, 'N') for base in dna_string]
    reversed_complement = reversed(complement)
    return ''.join(list(reversed_complement))

# %%
class Fasta():
    def __init__(self, fasta) -> None:
        self._fasta = pysam.FastaFile(fasta)
    
    def fetch(self, chrom, start, end, strand='+', reverse = False):
        sequence = self._fasta.fetch(chrom, start, end).upper()
        if reverse:
            sequence = sequence[::-1]
        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')
        
        return sequence

# %%
def embed_from_multilabled_bed_gen(bed, reference_fasta, embedder, label_column_idx, label_depth):
    fasta = Fasta(reference_fasta)
    with open(bed) as f:
        for line in tqdm.tqdm(f):
            # get bed row
            row = line.strip().split('\t')
            chrom, start, end, strand = row[0], int(row[1]), int(row[2]), row[5]
            labels = list(map(int, row[label_column_idx].split(',')))

            # get sequence
            sequence = fasta.fetch(chrom, start, end, strand)

            # embed sequence and multi-hot encode labels
            sequence_embed = tf.squeeze(tf.constant(embedder(sequence)))
            labels_multi_hot = multi_hot(labels, depth=label_depth)

            yield {'inputs': sequence_embed, 'outputs': labels_multi_hot}


def embed_from_bed(bed, reference_fasta, embedder, upsample_embeddings = False, 
                  hdf5_file= None, read_strand = False, 
                  read_reverse = False, label_column_idx=6, label_depth=None, split = None):
    fasta = Fasta(reference_fasta)
    # open hdf5 file 
    hdf5_file = h5py.File(hdf5_file, mode = "r") if hdf5_file else None
    header = 'infer' if has_header(bed) else None
    f = pd.read_csv(bed, header = header, sep = '\s+')
    
    label_column_idx = f.columns.get_loc('label') if 'label' in f.columns else label_column_idx
    strand_column_idx = f.columns.get_loc('strand') if 'strand' in f.columns else 3
    if split: 
        f = f[f.iloc[:, -1] == split]

    for n, line in tqdm.tqdm(f.iterrows()):
        # get bed row
        if read_strand:
            chrom, start, end, strand, reverse = line[0], int(line[1]), int(line[2]), line[strand_column_idx], False
        if read_reverse: 
            chrom, start, end, strand, reverse = line[0], int(line[1]), int(line[2]), '+', bool(line[4]) # strand will be reversed
        else:
            chrom, start, end, strand, reverse = line[0], int(line[1]), int(line[2]), '+', False # strand wil not be reversed
        if hdf5_file is not None: 
            labels = hdf5_file['labels'][n]
        else: 
            labels = list(map(int, line[label_column_idx].split(',')))
            labels = multi_hot(labels, depth=label_depth)
        # get sequence
        sequence = fasta.fetch(chrom, start, end, strand = strand, reverse = reverse) # categorical labels
        # embed sequence
        sequence_embed = tf.squeeze(tf.constant(embedder(sequence, upsample_embeddings = upsample_embeddings)))
        yield {'inputs': sequence_embed, 'outputs': labels}


def get_splits(bed):
    header = 'infer' if has_header(bed) else None
    f = pd.read_csv(bed, header = header, sep = '\s+')
    splits = f.iloc[:, -1].unique().tolist() # splits should be in last column
    return splits
        