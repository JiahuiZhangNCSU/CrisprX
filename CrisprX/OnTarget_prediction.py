# This script is designed for predicting the on-target efficiency of a set of given gRNA.
from typing import List
import pandas as pd
from CrisprX.utils import Nuc_NtsTokenizer, Dimer_NtsTokenizer
import subprocess
import re
import numpy as np
from Bio.SeqUtils import MeltingTemp as mt
from torch.utils.data import Dataset, DataLoader
import torch

def get_features(csv_path="../tasks/on_target.csv", output_path="../tasks/on_target_features.csv"):
    """
    This function reads the csv file and write a new csv file with features.
    """
    df = pd.read_csv(csv_path)
    scafold = 'gttttagagctagaaatagcaagttaaaataaggctagtccgttatcaacttgaaaaagtggcaccgagtcggtgcttttt'.upper()
    df['gRNA'] = df['Seq'].str[:20] + scafold
    df['gRNA'] = df['gRNA'].str.replace('T', 'U')
    structures = []
    free_energies = []
    # Loop over the RNA sequences and calculate their secondary structures and free energies using RNAfold
    for seq in df["gRNA"]:
        command = f'echo "{seq}" | RNAfold'

        # Call LinearFold using subprocess and provide the RNA sequence as input
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        # Parse the output of LinearFold to extract the predicted secondary structure and free energy
        linearfold_output = output.decode('utf-8')
        structure, free_energy_str = linearfold_output.strip().rsplit(' ', 1)
        free_energy = float(free_energy_str.strip('()'))

        # Add the secondary structure and free energy to their respective lists
        structures.append(structure)
        free_energies.append(free_energy)
    # Add the secondary structures and free energies to new columns in the DataFrame
    df["Secondary_Structure"] = structures
    df["dG"] = free_energies

    def count_hairpin(sequence):
        stack = []
        hairpins = 0

        for i in range(len(sequence)):
            if sequence[i] == '(':
                stack.append(i)
            elif sequence[i] == ')':
                if len(stack) > 0:
                    start = stack.pop()
                    end = i
                    substr = sequence[start + 1:end]
                    if '(' not in substr and ')' not in substr:
                        hairpins += 1

        return hairpins

    df["stem"] = df["Secondary_Structure"].apply(count_hairpin)

    df['grna20'] = df['gRNA'].str[:20]
    # Define an empty list to store the free energies
    free_energies = []

    # Loop over the RNA sequences and calculate their free energies using RNAfold
    for id, seq in enumerate(df["grna20"]):
        command = f'echo "{seq}" | RNAfold'
        # Call LinearFold using subprocess and provide the RNA sequence as input
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        # Parse the output of LinearFold to extract the predicted secondary structure and free energy
        linearfold_output = output.decode('utf-8')
        free_energy_str = re.search(r'-?\d+\.\d+', linearfold_output).group(0)
        free_energy = float(free_energy_str)

        # Add the free energy to the list
        free_energies.append(free_energy)
    # Add the free energies to a new "Free_Energy" column in the DataFrame
    df["dG_binding_20"] = free_energies

    df['grna7to20'] = df['gRNA'].str[6:20]
    # Define an empty list to store the free energies
    free_energies = []

    # Loop over the RNA sequences and calculate their free energies using RNAfold
    for id, seq in enumerate(df["grna7to20"]):
        command = f'echo "{seq}" | RNAfold'
        # Call LinearFold using subprocess and provide the RNA sequence as input
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        # Parse the output of LinearFold to extract the predicted secondary structure and free energy
        linearfold_output = output.decode('utf-8')
        free_energy_str = re.search(r'-?\d+\.\d+', linearfold_output).group(0)
        free_energy = float(free_energy_str)

        # Add the free energy to the list
        free_energies.append(free_energy)
    # Add the free energies to a new "Free_Energy" column in the DataFrame
    df["dG_binding_7to20"] = free_energies

    df['CG'] = df['Seq'].str[:20].str.count('C') + df['Seq'].str[:20].str.count('G')
    df['CG>10'] = 0
    df['CG<10'] = 0
    df['CG>10'] = np.where(df['CG'] > 10, 1, 0)
    df['CG<10'] = np.where(df['CG'] <= 10, 1, 0)

    df['Tm_global'] = df['Seq'].str[:21].apply(mt.Tm_NN)
    df['Tm_5mer_end'] = df['Seq'].str[15:20].apply(mt.Tm_NN)
    df['Tm_8mer_middle'] = df['Seq'].str[6:14].apply(mt.Tm_NN)
    df['Tm_4mer_start'] = df['Seq'].str[0:4].apply(mt.Tm_NN)
    df[['Seq', 'stem', 'dG', 'dG_binding_20', 'dG_binding_7to20', 'CG>10', 'CG<10', 'CG', 'Tm_global', 'Tm_5mer_end',
        'Tm_8mer_middle', 'Tm_4mer_start']].to_csv(output_path, index=False)


def get_csv_data(csv_path="../tasks/on_target_features.csv") -> List:
    """
    This function reads the csv file and returns the data.
    """
    data = []
    df = pd.read_csv(csv_path)
    for i in range(len(df)):
        seq = df['Seq'][i]
        seq = df['Seq'][i]
        t1 = Nuc_NtsTokenizer()
        seq_result = t1.tokenize(seq)
        t2 = Dimer_NtsTokenizer()
        dimer_result = t2.tokenize(seq)
        # Contact the seq_result and dimer result and append it to data.
        seq_result.extend(dimer_result)
        stem = df['stem'][i]
        dG = df['dG'][i]
        dG_binding_20 = df['dG_binding_20'][i]
        dG_binding_7to20 = df['dG_binding_7to20'][i]
        GCmt10 = df['CG>10'][i]
        GClt10 = df['CG<10'][i]
        GC = df['CG'][i]
        Tm_global = df['Tm_global'][i]
        Tm_5mer_end = df['Tm_5mer_end'][i]
        Tm_8mer_middle = df['Tm_8mer_middle'][i]
        Tm_4mer_start = df['Tm_4mer_start'][i]
        seq_result.extend(
            [stem, dG, dG_binding_20, dG_binding_7to20, GCmt10, GClt10, GC, Tm_global, Tm_5mer_end, Tm_8mer_middle, Tm_4mer_start])
        data.append(seq_result)
    return data


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    batch_X = []
    for cur_X in batch_data:
        batch_X.append(torch.tensor(cur_X, dtype=torch.float))
    batch_X = torch.stack(batch_X)
    return batch_X


def create_dataloader(dataset, batch_size=1, shuffle=False, fn=None, drop_last=False):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=fn, drop_last=drop_last)
    return dl


def OnTarget_output(model, dataloader, device, output_path="../tasks/on_target_predictions.csv"):
    model.eval()
    pred = []
    id = []
    i = 0
    for batch in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            output = model(batch)
            pred.append(output.item())
            id.append(i)
            i += 1
    # Write the prediction to csv file.
    df = pd.read_csv("../tasks/on_target_features.csv")
    df['pred'] = pred
    df['id'] = id
    df = df[['id', 'Seq', 'pred']]
    df.to_csv(output_path, index=False)


def OnTarget_predict(model, device):
    get_features()
    get_csv_data()
    data = get_csv_data()
    dataset = SequenceDataset(data)
    dataloader = create_dataloader(dataset, batch_size=1, shuffle=False, fn=collate_fn, drop_last=False)
    OnTarget_output(model, dataloader, device)


if __name__ == "__main__":
    import json
    import argparse
    from CrisprX.models import TransCrispr

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='device number')
    args = parser.parse_args()

    on_target_json_path = "../trained_models/TransCrispr.json"
    with open(on_target_json_path, 'r') as f:
        on_target_config = json.load(f)

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    model = TransCrispr(nuc_emb_outputdim=on_target_config["emb_out_dim"],
                            conv1d_filters_num=on_target_config["conv_out_chn"],
                            conv1d_filters_size=on_target_config["k_sz"],
                            dropout=on_target_config["dp"], transformer_num_layers=on_target_config["trln"],
                            transformer_final_fn=on_target_config["trfn"],
                            transformer_ffn_1stlayer=on_target_config["trffd"],
                            dense1=on_target_config["d1"], dense2=on_target_config["d2"],
                            dense3=on_target_config["d3"], MLP_out=on_target_config["physics_width"]).to(device)

    pretrained_dict = torch.load('../trained_models/TransCrispr.pt', map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    OnTarget_predict(model, device)





