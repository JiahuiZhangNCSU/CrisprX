# This script is designed for predicting the off-target effect of a set of given gRNA and off-target DNA sequences.
from typing import List
import pandas as pd
from CrisprX.utils import encode
from torch.utils.data import Dataset, DataLoader
import torch


def get_csv_data(csv_path="../tasks/off_target.csv") -> List:
    """
    This function reads the csv file and returns a list of gRNA and a list of off-target DNA sequences.
    """
    data = []
    df = pd.read_csv(csv_path)
    for i in range(len(df)):
        grna = df.loc[i, 'crRNA']
        otdna = df.loc[i, 'DNA']
        otdna = otdna.upper()
        enc_seq = encode(grna, otdna)
        data.append(enc_seq)
        if device == torch.device("cpu"):
            print("Sequence {} finished".format(i))
    return data


class SequenceDataset(Dataset):
    """
    Construct the dataset.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    """
    Define the collate function.
    """
    batch_X = []
    for cur_X in batch_data:
        batch_X.append(torch.tensor(cur_X, dtype=torch.float))
    batch_X = torch.stack(batch_X)
    return batch_X


def create_dataloader(dataset, batch_size=128, shuffle=False, fn=None, drop_last=False):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=fn, drop_last=drop_last)
    return dl


def OffTarget_output(model, dataloader, device, threshhold=0.1, output_path="../tasks/off_target_predictions.csv"):
    model.eval()
    pred = []
    j = 0
    for batch in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            output = model(batch)[:,1]
            if device == torch.device("cpu"):
                print("batch {} finished".format(j))
            j += 1
            for i in range(len(output)):
                pred.append(1 if output[i].item()>threshhold else 0)
    # Write the prediction to csv file.
    df = pd.read_csv("../tasks/off_target.csv")
    df['pred'] = pred
    df = df[['#Id', 'crRNA', 'DNA', 'pred']]
    df.to_csv(output_path, index=False)


def OffTarget_predict(threshold, model, device):
    get_csv_data()
    data = get_csv_data()
    dataset = SequenceDataset(data)
    dataloader = create_dataloader(dataset, batch_size=1024, shuffle=False, fn=collate_fn, drop_last=False)
    OffTarget_output(model, dataloader, device, threshhold=threshold)


if __name__ == "__main__":
    import json
    import argparse
    from CrisprX.models import CrisprDNT

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='device number')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for prediction')
    args = parser.parse_args()

    off_target_json_path = "../trained_models/CrisprDNT.json"
    with open(off_target_json_path, 'r') as f:
        off_target_config = json.load(f)

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    model = CrisprDNT(channel_num=off_target_config['channel_num'],
                           pool_ks=off_target_config['pool_ks'], rnn_units=off_target_config['rnn_units'],
                           rnn_num_layers=off_target_config['rnn_num_layers'],
                           rnn_drop=off_target_config["rnn_drop"],
                           num_heads=off_target_config["num_heads"], dense1=off_target_config["dense1"],
                           dense2=off_target_config["dense2"], dense3=off_target_config["dense3"],
                           dense4=off_target_config["dense4"], dense_drop=off_target_config["dense_drop"],
                           out1=off_target_config["out1"], out2=off_target_config["out2"]).to(device)

    pretrained_dict = torch.load('../trained_models/CrisprDNT.pt', map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    OffTarget_predict(args.threshold, model, device)





