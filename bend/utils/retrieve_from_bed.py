'''
Class to extract sequences from a reference genome 
using a `bed` file.


how to use:
get_dna = Annotation(annotation = 'path/to/bed/file', reference_genome = '/path/to/genome/fasta')
get_dna.get_dna_segment(index = 0) # will return the dna segment for index 0 in the annotation file
'''
from Bio import SeqIO
import pandas as pd



class Annotation:
    def __init__(self, annotation : str = None, 
                reference_genome : str = None) -> str:

        '''
        Return a dna sequence from a bed file and a reference genome
        annotation: path to bed file
        reference_genome: path to reference genome fasta file
        '''
        if annotation is not None:
            if isinstance(annotation, str):
                annotation = pd.read_csv(annotation, sep = '\s+')

            self.annotation = annotation
        if reference_genome is not None: 
            self.genome_dict = SeqIO.to_dict(SeqIO.parse(reference_genome, "fasta"))

    
    def extend_segments(self, extra_context_left: int = None, extra_context_right: int = None, extra_context: int = None):
        '''Modify the annotation to include extra context on both sides of the segments'''

        if extra_context is not None:
            if extra_context_right is not None or extra_context_left is not None:
                raise ValueError('extra_context cannot be used with extra_context_left or extra_context_right')
            extra_context_left = extra_context
            extra_context_right = extra_context

        self.annotation.loc[:, 'start'] = self.annotation.loc[:, 'start'] - extra_context_left
        self.annotation.loc[:, 'end'] = self.annotation.loc[:, 'end'] + extra_context_right
            
    
    def get_item(self, index) :
        transcript_df = self.annotation.iloc[index] # return the row of the annotation

        return transcript_df

    def get_dna_segment(self, index = None):
        
        item = self.get_item(index)
         
        # get dna segment from genome dict  
        dna_segment =  str(self.genome_dict[item.chromosome].seq[int(item.start) : int(item.end)]) 
        
        return dna_segment 
    