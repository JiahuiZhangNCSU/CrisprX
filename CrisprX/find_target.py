# This script is designed for finding all potential targets in a given long sequence.
import os
from Bio import SeqIO
import pandas as pd


def find_target(input_fa, input_path="../input/"):
    """
    This function finds all potential targets in a given long sequence.
    """
    fasta_file = os.path.join(input_path, input_fa)
    sequences = SeqIO.parse(fasta_file, "fasta")

    result = []
    for record in sequences:
        dna_seq = str(record.seq)
        dna_seq = dna_seq.upper()
        sub_sequences = [dna_seq[i:i + 23] for i in range(len(dna_seq) - 23 + 1) if
                         dna_seq[i + 23 - 2:i + 23] == "GG"]
        result.extend(sub_sequences)

    return result


def to_csv(result, output_path="../tasks/"):
    """
    This function converts the output file to csv file.
    """
    # Write a csv file to store the result.
    df = pd.DataFrame(result, columns=["Seq"])
    df.to_csv(os.path.join(output_path, "on_target.csv"), index=False)


def to_fasta(result, output_path="../alignments/"):
    """
    This function converts the output file to fasta file.
    """
    # Write a fasta file to store the result.
    with open(os.path.join(output_path, "on_target.fa"), "w") as output_handle:
        for i in range(len(result)):
            output_handle.write(">{}\n".format(i))
            output_handle.write("{}\n".format(result[i]))


def get_target(input_fa, input_path="../input/", output_path="../tasks/", output_path2="../alignments/"):
    """
    This function finds all potential targets in a given long sequence.
    """
    result = find_target(input_fa, input_path)
    to_csv(result, output_path)
    to_fasta(result, output_path2)


if __name__ == "__main__":
    get_target("sequence.fa")

