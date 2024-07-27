# Description: This script aligns the reads to the reference genome using cas-OFFinder.
import Bio.SeqIO
import os
import pandas as pd
import subprocess


def make_input(reference, query, working_path='../alignments', genome_path='../alignments/'):
    """
    This function generate the input file for cas-OFFinder.
    """
    fasta_file_path = os.path.join(genome_path, reference)
    # Create a txt file to store the information and sequences of the query file.
    txt_file_path = os.path.join(working_path, 'input.txt')
    query_file_path = os.path.join(working_path, query)
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write('{}\n'.format(fasta_file_path))
        # Change the zeros when we can predict off-targets with pairs.
        txt_file.write('{}\n'.format("NNNNNNNNNNNNNNNNNNNNNRG 0 0"))
        for record in Bio.SeqIO.parse(query_file_path, 'fasta'):
            txt_file.write('{} 5\n'.format(record.seq))


def run_casOFFinder(reference, query, working_path='../alignments', genome_path='../genome/'):
    """
    This function runs the cas-OFFinder to find the potential off-targets.
    """
    make_input(reference, query, working_path, genome_path)
    os.chdir(working_path)
    # Change C to G when using GPU.
    # Write the full path to the cas-offinder.
    # command = "/Users/yalotein/Projects/gRNA/cas-offinder/build/cas-offinder input.txt C output.txt"
    command = 'source ~/.bashrc; cas-offinder input.txt C output.txt'
    os.system(command)


def to_csv(output="output.txt", working_path="../alignments"):
    """
    This function converts the output file to csv file.
    """
    os.chdir(working_path)
    # Use pandas to convert the output file to csv file.
    df = pd.read_csv(output, sep='\t', skiprows=1)
    df.to_csv('output.csv', index=False)
    move_csv()


def move_csv(working_path="../alignments", target_path="../tasks"):
    """
    This function moves the csv file to the target path.
    """
    new_name = os.path.join(target_path, 'off_target.csv')
    os.chdir(working_path)
    os.system('mv output.csv {}'.format(new_name))


def genome_align(reference="hg38.fa", query='on_target.fa', working_path='../alignments', genome_path='../genome/'):
    """
    This function aligns the reads to the reference genome.
    """
    run_casOFFinder(reference, query, working_path, genome_path)
    to_csv()


if __name__ == "__main__":
    genome_align()






