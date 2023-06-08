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
                annotation = pd.read_csv(annotation)
            self.annotation = annotation
            #self.annotation.loc[:, ['start', 'end']] -= 1 # subtract 1 from all to 0-index, not neccessary in bed format
        if reference_genome is not None: 
            self.genome_dict = SeqIO.to_dict(SeqIO.parse(reference_genome, "fasta"))
            
    
    def get_item(self, index, reset_start = False, return_type = None, subtract_type = 'transcript') :
        transcript_df = self.annotation.iloc[index] # return the row of the annotation

        return transcript_df

    def get_dna_segment(self, contig=None, index = None):
        
        item = self.get_item(index)
         
        # get dna segment from genome dict  
        dna_segment =  str(self.genome_dict[item.chromosome].seq[int(item.start) : int(item.end)+1]) 
        
        return dna_segment 
    