"""
sequtils.py
===========
Utilities for processing genome coordinate-based sequence data to embeddings.
"""
from tqdm.auto import tqdm
import pysam
import pandas as pd
import numpy as np
import webdataset as wds
import h5py

baseComplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

def multi_hot(labels, num_labels):
    """
    Convert a numpy array to a one-hot encoded numpy array.

    Parameters
    ----------
    labels : list
        The labels that are true
    num_labels : int
        The number of potential labels.

    Returns
    -------
    numpy.ndarray
        A multi-hot encoded numpy array.
    """
    encoded = np.eye(num_labels, dtype=np.int64)[labels].sum(axis=0)

    #encoded = np.zeros((num_labels), dtype=np.int64)
    #for i, row in enumerate(labels):
    #    encoded[row] = 1
    return encoded

def reverse_complement(dna_string: str):
    # """Returns the reverse-complement for a DNA string."""
    """
    Returns the reverse-complement for a DNA string.
    
    Parameters
    ----------
    dna_string : str
        DNA string to reverse-complement.
        
    Returns
    -------
    str
        Reverse-complement of the input DNA string.
    """

    complement = [baseComplement.get(base, 'N') for base in dna_string]
    reversed_complement = reversed(complement)
    return ''.join(list(reversed_complement))

# %%
class Fasta():
    """Class for fetching sequences from a reference genome fasta file."""
    def __init__(self, fasta) -> None:
        """
        Initialize a Fasta object for fetching sequences from a reference genome fasta file.
        
        Parameters
        ----------
        fasta : str
            Path to a reference genome fasta file.
        """
        
        self._fasta = pysam.FastaFile(fasta)
    
    def fetch(self, chrom: str, start: int, end: int, strand: str = '+', flank : int = 0) -> str:
        """
        Fetch a sequence from the reference genome fasta file.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        start : int
            Start coordinate.
        end : int
            End coordinate.
        strand : str, optional
            Strand. The default is '+'.
            If strand is '-', the sequence will be reverse-complemented before returning.
        flank : int, optional
            Number of bases to add to the start and end coordinates. The default is 0.
        Returns
        -------
        str
            Sequence from the reference genome fasta file.
        """
        sequence = self._fasta.fetch(str(chrom), start - flank, end + flank).upper()
        
        if strand == '+':
            pass
        elif strand == '-':
            sequence = ''.join(reverse_complement(sequence))
        else:
            raise ValueError(f'Unknown strand: {strand}')
        
        return sequence




def embed_from_bed(bed, reference_fasta, embedder, 
                    output_path,
                   hdf5_file= None,
                   chunk_size = None, chunk: int = None, 
                   upsample_embeddings = False,
                    read_strand = False, label_column_idx=6, 
                  label_depth=None, split = None, flank = 0):
    fasta = Fasta(reference_fasta)
    f = pd.read_csv(bed, header = 'infer', sep = '\t', low_memory=False)
    if split: 
        f = f[f.iloc[:, -1] == split]
    label_column_idx = f.columns.get_loc('label') if 'label' in f.columns else label_column_idx
    strand_column_idx = f.columns.get_loc('strand') if 'strand' in f.columns else 3
    # open hdf5 file 
    hdf5_file = h5py.File(hdf5_file, mode = "r") if hdf5_file else None
    
    if chunk is not None:
        # check if chunk is valid 
        if chunk * chunk_size > len(f):
            raise ValueError(f'Requested chunk {chunk}, but chunk ids range from 0-{int(len(f) / chunk_size)}')
        f = f[chunk*chunk_size:(chunk+1)*chunk_size].reset_index(drop=True)

    buffer_inputs = []
    buffer_labels = []
    start_offset = chunk*chunk_size
    buffer_size=5000

    sink = wds.TarWriter(output_path, compress=True)
    for n, line in tqdm(f.iterrows(), total=len(f), desc='Embedding sequences'):
        # get bed row
        if read_strand:
            chrom, start, end, strand = line.iloc[0], int(line.iloc[1]), int(line.iloc[2]), line.iloc[strand_column_idx]
        else:
            chrom, start, end, strand = line.iloc[0], int(line.iloc[1]), int(line.iloc[2]), '+' 
        if hdf5_file is not None: 
            labels = hdf5_file['labels'][n + (chunk*chunk_size)]
        else: 
            labels = line.iloc[label_column_idx]
            labels = list(map(int, labels.split(','))) if isinstance(labels, str) else [] # if no label for sample
            labels = multi_hot(labels, label_depth)
        # get sequence
        sequence = fasta.fetch(chrom, start, end, strand = strand, flank = flank) # categorical labels
        # embed sequence
        sequence_embed = embedder(sequence, upsample_embeddings = upsample_embeddings)
        if sequence_embed.shape[1] != len(sequence):
            print(f'Embedding length does not match sequence length ({sequence_embed.shape[1]} != {len(sequence)} : {n} {chrom}:{start}-{end}{strand})')
            print(n, chrom, start, end, strand)
            continue

        sink.write({
            "__key__": f"sample_{n + start_offset}",
            "input.npy": sequence_embed,
            "output.npy": labels
        })

    sink.close()





def get_splits(bed):
    #header = 'infer' if has_header(bed) else None
    f = pd.read_csv(bed, header = 'infer', sep = '\t')
    splits = f.iloc[:, -1].unique().tolist() # splits should be in last column
    return splits
        