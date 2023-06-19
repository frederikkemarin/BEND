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
        
        self.nucleotide_categories = nucleotide_categories
        
        self.label_encoder = LabelEncoder().fit(self.nucleotide_categories)
        
    
    def transform_integer(self, sequence, return_onehot = False): # integer/onehot encode sequence
        if isinstance(sequence, np.ndarray):
            return sequence
        if isinstance(sequence[0], str):  # if input is str 
            sequence = np.array(list(sequence))
        
        sequence = self.label_encoder.transform(sequence)
        
        if return_onehot:
            sequence = np.eye(len(self.nucleotide_categories))[sequence]
        return sequence
    
    def inverse_transform_integer(self, sequence):
        if isinstance(sequence, str): # if input is str
            return sequence
        sequence = EncodeSequence.reduce_last_dim(sequence) # reduce last dim
        sequence = self.label_encoder.inverse_transform(sequence)
        return ('').join(sequence)
    
    @staticmethod
    def reduce_last_dim(sequence):
        if isinstance(sequence, (str, list)): # if input is str
            return sequence
        if len(sequence.shape) > 1:
            sequence = np.argmax(sequence, axis=-1)
        return sequence
    


class TransformDNASequence(EncodeSequence):
    def __init__(self,                
                 nucleotide_categories : List[str] = categories_4_letters_unknown,
                 label_type = 'labels_simple_direction_DA', 
                ):
        
        super().__init__(categories = nucleotide_categories)
            
        self.label_dict = label_dict[label_type]
        
    @staticmethod
    def make_mask(sequence, mask = None):
        if mask is None:
            mask = np.full(len(sequence), True)
        return mask
    @staticmethod        
    def get_strand(label=None, reverse_value = 4, strand=None):
        if label is None:
            return None
        elif reverse_value in label:
            strand = '-'
        else:
            strand = '+'
        return strand
    
    # get CDS
    def get_cds(self, sequence,
                label,
                mask = None, 
                CDS_values : Optional[List[int]] = None, 
                return_forward = True,
                return_string = False,
               ):
            
        mask = TransformDNASequence.make_mask(sequence, mask)
        
        label = self.reduce_last_dim(label)[mask] 
        
        if isinstance(sequence, str):
            sequence = self.transform_integer(sequence)
                
        sequence = sequence[mask]
            
        sequence = self.reduce_last_dim(sequence) 
        
        if CDS_values is None: 
            CDS_values = [self.label_dict[k].get('CDS') for k in self.label_dict.keys() if type(self.label_dict[k]) is dict]
            CDS_values = np.concatenate(CDS_values).tolist()
        cds_mask = np.in1d(label, CDS_values)
        sequence = sequence[cds_mask]
        
        if return_string is True:
            sequence = self.inverse_transform_integer(sequence)
            if return_forward is True and TransformDNASequence.get_strand(label) == '-':
                sequence = TransformDNASequence.get_reverse_complement(sequence)
            
        return sequence

    # translate to protein
    def translate(self, sequence, label = None, 
                  mask = None, reverse_strand = False, get_cds_first = True): 
        
        '''
        Translate DNA sequence into protein sequence
        Ff the sequence is the reverse strand then translate reverse complement.
        '''
        if get_cds_first is True and label is not None: 
            sequence = self.get_cds(sequence, label = label, mask = mask, return_forward = True, return_string=True)
        else: # TODO this is clumsy
            if isinstance(sequence, (torch.Tensor, np.ndarray)): # make string
                sequence = self.inverse_transform_integer(sequence)
        
        sequence = Seq(sequence)
        if reverse_strand:
            sequence = sequence.reverse_complement()
        sequence = sequence.translate()
    
        return str(sequence)
    
    @staticmethod
    def get_complement(sequence):
        pass
    @staticmethod 
    def get_reverse_complement(sequence):
        sequence = Seq(sequence)
        sequence = sequence.reverse_complement()
        return str(sequence)
    
    def get_cds_bulk(self, sequence, label, mask = None, 
                     CDS_values : Optional[List[int]] = None, 
                     return_forward = True,
                     return_string = False,):
    

        mapfunc = partial(self.get_cds, mask = mask, 
                          CDS_values = CDS_values, return_forward = return_forward, return_string=return_string)
        cds = list(map(mapfunc, sequence, label))
        return cds
    
    
    def translate_bulk(self, sequence, label = None, mask = None, reverse_strand = False, get_cds_first = True):
        if label is None: 
            mapfunc = partial(self.translate, label = label, mask = mask, reverse_strand= reverse_strand, get_cds_first=get_cds_first)
            protein = list(map(mapfunc, sequence))
        else: 
            mapfunc = partial(self.translate, mask = mask, reverse_strand= reverse_strand, get_cds_first=get_cds_first)
            protein = list(map(mapfunc, sequence, label))
        return protein
    
    def bulk_fn(self, fn,  *args, **kwargs):
        

        mapfunc = partial(fn, **kwargs)
        
        result = list(map(mapfunc, *args))
        
        return result