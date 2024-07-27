# Train the CrisprDNT model.
from typing import Tuple, Sequence, List, Dict, Any, Optional, Union, TypeVar
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from CrisprX.utils import encode
from CrisprX.models import CrisprDNT
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import scipy as sp
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score


def get_csv_data(csv_file) -> Tuple[List,List]:
    """
    Read the data from csv file.
    :param csv_file: the input csv file.
    :return: the lists of sequences and targets.
    """
    data = []
    targets = []
    df = pd.read_csv(csv_file)
    for i in range(len(df)):
        grna = df.loc[i, 'sgrna']
        otdna = df.loc[i, 'otdna']
        enc_seq = encode(grna, otdna)
        data.append(enc_seq)
        tgt = np.zeros(2)
        tgt[df.loc[i, 'label']] = 1
        targets.append(tgt)
    return data, targets


class SequenceDataset(Dataset):
    """
    Construct the dataset.
    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def collate_fn(batch_data):
    """
    Define the collate function.
    """
    batch_X, batch_Y = [], []
    for cur_X, cur_Y in batch_data:
        batch_X.append(torch.tensor(cur_X, dtype=torch.float))
        batch_Y.append(torch.tensor(cur_Y, dtype=torch.float))
    batch_X = torch.stack(batch_X)
    batch_Y = torch.stack(batch_Y)
    return batch_X, batch_Y


def create_dataloader(dataset, batch_size, shuffle=False, fn=None, drop_last=False):
    """
    Create the dataloader.
    """
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=fn, drop_last=drop_last)
    return dl


def train(model, train_loader, optimizer, criterion, device, scheduler, epoch):
    """
    Define the training function for a single epoch.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # # Never forget the resize the shape of the target!!!
        # target = target.unsqueeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if device == torch.device("cpu"):
            print(f"In epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item()}")
    scheduler.step()


def get_loss():
    """
    Define the loss function.
    :param alpha: the weight of the NCE.
    :param beta: the weight of the RCE.
    :return: the loss function.
    """
    # criterion = NCEandRCE(alpha=alpha, beta=beta)
    # criterion = NCEandRCE(alpha=1, beta=0.1)
    criterion = nn.CrossEntropyLoss()
    return criterion


if __name__ == "__main__":
    # Only for test.
    import argparse
    from tensorboardX import SummaryWriter
    import csv
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ep', type=int, default=200, help='number of epochs')
    parser.add_argument('--bs', type=int, default=1000, help='batch size')
    parser.add_argument('--device', type=int, default=0, help='device number')
    parser.add_argument('--channel_num', type=int, default=64, help='convolution channel number')
    parser.add_argument('--pool_ks', type=int, default=2, help='pooling kernel size')
    parser.add_argument('--rnn_units', type=int, default=32, help='rnn hidden dimension')
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='rnn layer number')
    parser.add_argument('--rnn_drop', type=float, default=0, help='rnn dropout rate')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dense1', type=int, default=512, help='dense layer 1 dimension')
    parser.add_argument('--dense2', type=int, default=64, help='dense layer 2 dimension')
    parser.add_argument('--dense3', type=int, default=512, help='dense layer 3 dimension')
    parser.add_argument('--dense4', type=int, default=64, help='dense layer 4 dimension')
    parser.add_argument('--dense_drop', type=float, default=0.1, help='dense layer dropout rate')
    parser.add_argument('--out1', type=int, default=256, help='output layer 1 dimension')
    parser.add_argument('--out2', type=int, default=64, help='output layer 2 dimension')
    parser.add_argument('--out_drop', type=float, default=0.25, help='output layer dropout rate')
    parser.add_argument('--csv_path', type=str, default='../datasets/', help='csv file path')
    parser.add_argument('--save', type=str, default='../trained_models/', help='model save path')
    parser.add_argument('--log', type=str, default='./', help='log path')
    parser.add_argument('--metrics', type=str, default='./', help='metrics path')
    parser.add_argument('--name', type=str, default='CrisprDNT', help='model name')

    args = parser.parse_args()

    # Save the hyperparameters to a json file.
    params = vars(args)
    json_file = args.name + '.json'
    json_path = os.path.join(args.save, json_file)
    with open(json_path, 'w') as json_file:
        json.dump(params, json_file)

    # Load the trainval dataset.
    trainval_data = get_csv_data(args.csv_path + "offtarget_trainval.csv")
    trainval_dataset = SequenceDataset(trainval_data[0], trainval_data[1])
    train_dataset, val_dataset = train_test_split(trainval_dataset, test_size=0.05, random_state=42)

    # Resample the train_dataset.
    xtrain = np.stack([subdataset[0] for subdataset in train_dataset], axis=0)
    ytrain = np.stack([subdataset[1] for subdataset in train_dataset], axis=0)

    pos_indices = ytrain[:, 1] == 1
    pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
    pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]

    num_pos_samples = len(pos_y)
    num_neg_samples = len(neg_y)

    replication_factor = num_neg_samples // num_pos_samples

    pos_x_balanced = np.repeat(pos_x, replication_factor, axis=0)
    pos_y_balanced = np.repeat(pos_y, replication_factor, axis=0)

    xtrain_balanced = np.concatenate([pos_x_balanced, neg_x], axis=0)
    ytrain_balanced = np.concatenate([pos_y_balanced, neg_y], axis=0)

    train_dataset = SequenceDataset(xtrain_balanced, ytrain_balanced)

    # Make the dataloader.
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    # data = []
    # dataset = []
    #
    # for i in range(6):
    #     data.append(get_csv_data(args.csv_path + "dataset" + str(i+1) + ".csv"))
    #     dataset.append(SequenceDataset(data[i][0], data[i][1]))
    #
    # # As a test, the last dataset is for test and the rest are for training.
    # all_dataset = ConcatDataset(dataset[:6])
    # train_dataset, valtest_dataset = train_test_split(all_dataset, test_size=0.1, random_state=42)
    # val_dataset, test_dataset = train_test_split(valtest_dataset, test_size=0.5, random_state=42)
    #
    # testdata1 = get_csv_data(args.csv_path+"dataset7.csv")
    # test_set1 = SequenceDataset(testdata1[0], testdata1[1])
    # testdata2 = get_csv_data(args.csv_path+"dataset8.csv")
    # test_set2 = SequenceDataset(testdata2[0], testdata2[1])
    #
    # # Resample the training dataset.
    # xtrain = np.stack([subdataset[0] for subdataset in train_dataset], axis=0)
    # ytrain = np.stack([subdataset[1] for subdataset in train_dataset], axis=0)
    #
    # pos_indices = ytrain[:, 1] == 1
    # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
    # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
    #
    # num_pos_samples = len(pos_y)
    # num_neg_samples = len(neg_y)
    #
    # replication_factor = num_neg_samples // num_pos_samples
    #
    # pos_x_balanced = np.repeat(pos_x, replication_factor, axis=0)
    # pos_y_balanced = np.repeat(pos_y, replication_factor, axis=0)
    #
    # xtrain_balanced = np.concatenate([pos_x_balanced, neg_x], axis=0)
    # ytrain_balanced = np.concatenate([pos_y_balanced, neg_y], axis=0)
    #
    # train_dataset = SequenceDataset(xtrain_balanced, ytrain_balanced)
    #
    # # Make dataloaders.
    # train_loader = create_dataloader(train_dataset, batch_size=args.bs, shuffle=True, fn=collate_fn)
    # val_loader = create_dataloader(val_dataset, batch_size=args.bs, shuffle=False, fn=collate_fn)
    # test_loader = create_dataloader(test_dataset, batch_size=args.bs, shuffle=False, fn=collate_fn)
    # test_loader1 = create_dataloader(test_set1, batch_size=args.bs, shuffle=False, fn=collate_fn)
    # test_loader2 = create_dataloader(test_set2, batch_size=args.bs, shuffle=False, fn=collate_fn)

    # Define the model.
    model = CrisprDNT(channel_num=args.channel_num, pool_ks=args.pool_ks,
                      rnn_units=args.rnn_units, rnn_num_layers=args.rnn_num_layers, rnn_drop=args.rnn_drop,
                      num_heads=args.num_heads, dense1=args.dense1, dense2=args.dense2, dense3=args.dense3,
                      dense4=args.dense4, dense_drop=args.dense_drop, out1=args.out1, out2=args.out2)

    # Define the optimizer.
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Define the loss function.
    criterion = get_loss()

    # Define the device.
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the scheduler.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # Define the test function.
    def test(model, test_loader, test_loader1, test_loader2, device, parameter_save_path):
        test_output = []
        test_target = []
        test_output_tensor = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # For the output, for the last dimension, the largest element is one and the other is zero.
            output_tensor = torch.zeros_like(output).to(device)
            max_indices = torch.argmax(output, dim=-1)
            output_tensor[torch.arange(output.size(0)), max_indices] = 1
            test_output.append(output[:,1])
            test_target.append(target[:,1])
            test_output_tensor.append(output_tensor[:,1])
        test_output = torch.cat(test_output, dim=0)
        test_target = torch.cat(test_target, dim=0)
        test_output_tensor = torch.cat(test_output_tensor, dim=0)
        test_output_cpu = test_output.cpu().detach().numpy()
        test_target_cpu = test_target.cpu().detach().numpy()
        test_output_tensor_cpu = test_output_tensor.cpu().detach().numpy()
        f1 = f1_score(test_target_cpu, test_output_tensor_cpu, average=None)
        precision = precision_score(test_target_cpu, test_output_tensor_cpu, average=None)
        recall = recall_score(test_target_cpu, test_output_tensor_cpu, average=None)
        accuracy = accuracy_score(test_target_cpu, test_output_tensor_cpu)
        roc_auc = roc_auc_score(test_target_cpu, test_output_cpu, average=None)
        pr_auc = average_precision_score(test_target_cpu, test_output_cpu, average=None)

        with open(parameter_save_path, 'w') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc'])
            writer_csv.writerow([f1, precision, recall, accuracy, roc_auc, pr_auc])

        test_output = []
        test_target = []
        test_output_tensor = []
        for data, target in test_loader1:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # For the output, for the last dimension, the largest element is one and the other is zero.
            output_tensor = torch.zeros_like(output).to(device)
            max_indices = torch.argmax(output, dim=-1)
            output_tensor[torch.arange(output.size(0)), max_indices] = 1
            test_output.append(output[:, 1])
            test_target.append(target[:, 1])
            test_output_tensor.append(output_tensor[:, 1])
        test_output = torch.cat(test_output, dim=0)
        test_target = torch.cat(test_target, dim=0)
        test_output_tensor = torch.cat(test_output_tensor, dim=0)
        test_output_cpu = test_output.cpu().detach().numpy()
        test_target_cpu = test_target.cpu().detach().numpy()
        test_output_tensor_cpu = test_output_tensor.cpu().detach().numpy()
        f1 = f1_score(test_target_cpu, test_output_tensor_cpu, average=None)
        precision = precision_score(test_target_cpu, test_output_tensor_cpu, average=None)
        recall = recall_score(test_target_cpu, test_output_tensor_cpu, average=None)
        accuracy = accuracy_score(test_target_cpu, test_output_tensor_cpu)
        roc_auc = roc_auc_score(test_target_cpu, test_output_cpu, average=None)
        pr_auc = average_precision_score(test_target_cpu, test_output_cpu, average=None)

        with open(parameter_save_path, 'a') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc'])
            writer_csv.writerow([f1, precision, recall, accuracy, roc_auc, pr_auc])

        test_output = []
        test_target = []
        test_output_tensor = []
        for data, target in test_loader2:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # For the output, for the last dimension, the largest element is one and the other is zero.
            output_tensor = torch.zeros_like(output).to(device)
            max_indices = torch.argmax(output, dim=-1)
            output_tensor[torch.arange(output.size(0)), max_indices] = 1
            test_output.append(output[:, 1])
            test_target.append(target[:, 1])
            test_output_tensor.append(output_tensor[:, 1])
        test_output = torch.cat(test_output, dim=0)
        test_target = torch.cat(test_target, dim=0)
        test_output_tensor = torch.cat(test_output_tensor, dim=0)
        test_output_cpu = test_output.cpu().detach().numpy()
        test_target_cpu = test_target.cpu().detach().numpy()
        test_output_tensor_cpu = test_output_tensor.cpu().detach().numpy()
        f1 = f1_score(test_target_cpu, test_output_tensor_cpu, average=None)
        precision = precision_score(test_target_cpu, test_output_tensor_cpu, average=None)
        recall = recall_score(test_target_cpu, test_output_tensor_cpu, average=None)
        accuracy = accuracy_score(test_target_cpu, test_output_tensor_cpu)
        roc_auc = roc_auc_score(test_target_cpu, test_output_cpu, average=None)
        pr_auc = average_precision_score(test_target_cpu, test_output_cpu, average=None)

        with open(parameter_save_path, 'a') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc'])
            writer_csv.writerow([f1, precision, recall, accuracy, roc_auc, pr_auc])


    # Define the summary writer.
    log_dir = args.name + "_log"
    logs_path = os.path.join(args.log, log_dir)
    writer = SummaryWriter(logs_path)

    # Train the model and write the loss and accuracy to tensorboard.
    def train_loop(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler, parameter_save_path):
        best_pr = 0
        for epoch in range(1, epochs + 1):
            train(model, train_loader, optimizer, criterion, device, scheduler, epoch)
            # Write the train the test loss and R to the tensorboard.
            model.eval()
            with torch.no_grad():
                train_loss = 0
                train_output = []
                train_target = []
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    train_loss += criterion(output, target).item()
                    train_output.append(output)
                    train_target.append(target)
                train_output = torch.cat(train_output, dim=0)
                train_target = torch.cat(train_target, dim=0)
                train_pr_auc = average_precision_score(train_target[:,1].cpu().detach().numpy(), train_output[:,1].cpu().detach().numpy(), average=None)
                train_loss /= len(train_loader)

                val_loss = 0
                val_output = []
                val_target = []
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    val_output.append(output)
                    val_target.append(target)
                val_output = torch.cat(val_output, dim=0)
                val_target = torch.cat(val_target, dim=0)
                val_pr_auc = average_precision_score(val_target[:,1].cpu().detach().numpy(), val_output[:,1].cpu().detach().numpy(), average=None)
                val_loss /= len(val_loader)

                if val_pr_auc > best_pr:
                    best_pr = val_pr_auc
                    # Save the model.
                    save_file = args.name + '.pt'
                    save_path = os.path.join(args.save, save_file)
                    torch.save(model.state_dict(), save_path)
                    # test(model, test_loader, test_loader1, test_loader2, device, parameter_save_path)
                # Write the train the val loss to the tensorboard on the same graph.
                writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                writer.add_scalars('PR_AUC', {'train': train_pr_auc, 'val': val_pr_auc}, epoch)
        writer.close()

    # Train the model.
    metrics_file = args.name + '_metrics.csv'
    metrics_save_path = os.path.join(args.metrics, metrics_file)

    train_loop(model, train_loader, val_loader, optimizer, criterion, args.ep, device, scheduler, metrics_save_path)

