from collections import Counter
import numpy as np
from Bio import SeqIO
from typing import List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from Bio.Seq import Seq
from functools import partial
import sys 

categories_4_letters_unknown = ['A', 'C', 'G', 'N', 'T']

# label dict for coding/non coding labelling
label_dict = {'labels_simple_direction_DA' : 
              {'+' : 
               {'exon' : 8, 'start_intron' : 1, 'intron' : 2, 'end_intron' : 3,
                'CDS' : np.array([0]), 'start_codon' : 0, # exon label 
                'three_prime_UTR' : 8, 'five_prime_UTR' : 8,  'stop_codon': 8, 'stop_codon_redefined_as_selenocysteine' : 8 # non_coding
                }, 
               '-' : 
                {'exon' : 8, 'start_intron' : 7, 'intron' : 6, 'end_intron' : 5, 
                'CDS' : np.array([4]), 'start_codon' : 4, # exon label 
                'three_prime_UTR' : 8, 'five_prime_UTR' : 8,  'stop_codon': 8, 'stop_codon_redefined_as_selenocysteine' : 8}, 
               'non_coding' : 8, 
               'padding' : 8
              },
              
              
              'labels_codon_direction_DA' :
              {'+' : 
               {'exon' : 24, 'start_intron' : 3 , 'intron' : 6, 'end_intron' : 9,
                'CDS' : np.array([0, 1, 2]), #'start_codon' : 0, # exon label 
                'three_prime_UTR' : 24, 'five_prime_UTR' : 24,  'stop_codon': 24, 'stop_codon_redefined_as_selenocysteine' : 24 # non_coding
                }, 
               '-' : 
               {'exon' : 24, 'start_intron' : 21, 'intron' : 18, 'end_intron' : 15,
                'CDS' : np.array([14, 13, 12]), #'start_codon' : 4, # exon label 
                'three_prime_UTR' : 24, 'five_prime_UTR' : 24,  'stop_codon': 24, 'stop_codon_redefined_as_selenocysteine' : 24}, 
               'non_coding' : 24, 
               'padding' : 24
              }, 
              
              'labels_simple_utr_DA' : {'+' : 
               {'exon' : 8, 'start_intron' : 1, 'intron' : 2, 'end_intron' : 3,
                'CDS' : np.array([0]), 'start_codon' : 0, # exon label 
                'three_prime_UTR' : 5, 'five_prime_UTR' : 4,  'stop_codon': 6, 'stop_codon_redefined_as_selenocysteine' : 8 # non_coding
                }, 
               '-' : 
                {'exon' : 8, 'start_intron' : 3, 'intron' : 2, 'end_intron' : 1, 
                'CDS' : np.array([0]), 'start_codon' : 0, # exon label 
                'three_prime_UTR' : 5, 'five_prime_UTR' : 4,  'stop_codon': 6, 'stop_codon_redefined_as_selenocysteine' : 8}, 
               'non_coding' : 6, 
               'padding' : 6
              },
              
             'labels_codon_utr_direction_DA' : 
              {'+' : 
               {'five_prime_UTR' : 0, 
                'exon' : 28, 'start_intron' : 4 , 'intron' : 7, 'end_intron' : 10,
                'CDS' : np.array([1, 2, 3]), #'start_codon' : 0, # exon label 
                'three_prime_UTR' : 13, 'stop_codon': 13, 'stop_codon_redefined_as_selenocysteine' : 13 # non_coding
                }, 
               '-' : 
               {'five_prime_UTR' : 14, 
                'exon' : 28, 'start_intron' : 24, 'intron' : 21, 'end_intron' : 18,
                'CDS' : np.array([17, 16, 15]), #'start_codon' : 4, # exon label 
                'three_prime_UTR' : 27, 'stop_codon': 27, 'stop_codon_redefined_as_selenocysteine' : 27}, 
               'non_coding' : 28, 
               'padding' : 28
              } 
             }



def count_nucleotides(fasta, destination = None):
    '''Count occurence of each nucleotide in fasta file.
    Parameters
    ----------
    fasta : str
        Path to fasta file.
    destination : str, optional
        Path to save dictionary with counts, by default None
    Returns
    -------
    counts : dict
        Dictionary with counts.
    '''

    fasta = SeqIO.parse(open(fasta),'fasta')
    counts = {}
    for record in fasta: 
        c = Counter( list(record.seq) )
        for k, v in c.items():
            if not k in counts.keys():
                counts[k] = v
            else: 
                counts[k] +=v
    if destination:  # save dictionary with counts
        np.save(destination, counts)
    return counts
        




class EncodeSequence:
    def __init__(self, nucleotide_categories = categories_4_letters_unknown):
        """
        Encode or decode sequence into integers, onehot or string.
        Parameters
        ----------
        nucleotide_categories : list
            List with nucleotide categories, by default categories_4_letters_unknown
        """
        
        self.nucleotide_categories = nucleotide_categories
        
        self.label_encoder = LabelEncoder().fit(self.nucleotide_categories)
        
    
    def transform_integer(self, sequence, return_onehot = False): # integer/onehot encode sequence
        """
        Encode string nucleotide sequence into integers or onehot.
        Parameters
        ----------
        sequence : str, list, np.ndarray
            Sequence to encode.
        return_onehot : bool, optional
            Return onehot encoded sequence, by default False
        Returns
        -------
        sequence : np.ndarray
            Encoded sequence.
        """
        if isinstance(sequence, np.ndarray):
            return sequence
        if isinstance(sequence[0], str):  # if input is str 
            sequence = np.array(list(sequence))
        
        sequence = self.label_encoder.transform(sequence)
        
        if return_onehot:
            sequence = np.eye(len(self.nucleotide_categories))[sequence]
        return sequence
    
    def inverse_transform_integer(self, sequence):
        """
        Decode integer encoded sequence into string.
        Parameters
        ----------
        sequence : np.ndarray
            Encoded sequence.
        Returns
        -------
        sequence : str
            Decoded sequence.
        """
        if isinstance(sequence, str): # if input is str
            return sequence
        sequence = EncodeSequence.reduce_last_dim(sequence) # reduce last dim
        sequence = self.label_encoder.inverse_transform(sequence)
        return ('').join(sequence)
    
    @staticmethod
    def reduce_last_dim(sequence):
        """
        Reduce last dimension of sequence.
        Parameters
        ----------
        sequence : np.ndarray
            Sequence to reduce last dimension.
        Returns
        -------
        sequence : np.ndarray
            Sequence with reduced last dimension.
        """
        if isinstance(sequence, (str, list)): # if input is str
            return sequence
        if len(sequence.shape) > 1:
            sequence = np.argmax(sequence, axis=-1)
        return sequence
    


