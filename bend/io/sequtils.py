import tqdm
import tensorflow as tf
import pysam
from bioio.tf.utils import multi_hot

# %%
baseComplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

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
    
    def fetch(self, chrom, start, end, strand='+'):
        sequence = self._fasta.fetch(chrom, start, end).upper()

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
            sequence_emebd = tf.squeeze(tf.constant(embedder(sequence)))
            labels_multi_hot = multi_hot(labels, depth=label_depth)

            yield {'inputs': sequence_emebd, 'outputs': labels_multi_hot}