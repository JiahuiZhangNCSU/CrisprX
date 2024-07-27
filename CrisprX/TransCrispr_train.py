# Train the TransCrispr model.
from typing import Tuple, Sequence, List, Dict, Any, Optional, Union, TypeVar
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from CrisprX.utils import Nuc_NtsTokenizer, Dimer_NtsTokenizer
from CrisprX.models import TransCrispr
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import scipy as sp


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
        seq_result.extend([stem, dG, dG_binding_20, dG_binding_7to20, GCmt10, GClt10, GC, Tm_global, Tm_5mer_end, Tm_8mer_middle, Tm_4mer_start])
        data.append(seq_result)
        targets.append(df['Efficiency'][i])
    return data, targets


# Construct the dataset.
class SequenceDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


# Define the collate function.
def collate_fn(batch_data):
    batch_X, batch_Y = [], []
    for cur_X, cur_Y in batch_data:
        batch_X.append(torch.tensor(cur_X, dtype=torch.float))
        batch_Y.append(torch.tensor(cur_Y, dtype=torch.float))
    batch_X = torch.stack(batch_X)
    batch_Y = torch.stack(batch_Y)
    return batch_X, batch_Y


# Make a dataloader.
def create_dataloader(dataset, batch_size, shuffle=False, fn=None, drop_last=True):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=fn, drop_last=drop_last)
    return dl


# Define the training function.
def train(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Never forget the resize the shape of the target!!!
        target = target.unsqueeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if device == torch.device("cpu"):
            print(batch_idx, loss.item())
    scheduler.step()


# Define the loss function.
def get_loss():
    criterion = nn.SmoothL1Loss(beta=0.25, reduction='mean')
    # criterion = nn.MSELoss(reduction='mean')
    return criterion


if __name__ == "__main__":
    import argparse
    from tensorboardX import SummaryWriter
    import csv
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--ep', type=int, default=100, help='number of epochs')
    parser.add_argument('--bs', type=int, default=20, help='batch size')
    parser.add_argument('--emb_out_dim', type=int, default=66, help='the output dimension of embedding layer')
    parser.add_argument('--conv_out_chn', type=int, default=512, help='the output channel of conv1d layer')
    parser.add_argument('--k_sz', type=int, default=7, help='the kernel size of conv1d layer')
    parser.add_argument('--dp', type=float, default=0.4, help='dropout rate')
    parser.add_argument('--trln', type=int, default=4, help='transformer layers number')
    parser.add_argument('--trfn', type=int, default=198, help='transformer final layer unit number')
    parser.add_argument('--trffd', type=int, default=111, help='transformer feed forward hidden layer unit number')
    parser.add_argument('--d1', type=int, default=176, help='dense layer 1')
    parser.add_argument('--d2', type=int, default=88, help='dense layer 2')
    parser.add_argument('--d3', type=int, default=22, help='dense layer 3')
    parser.add_argument('--L2_reg', type=float, default=0.0001, help='L2 regularization rate')
    parser.add_argument('--physics_width', type=int, default=256, help='the output dimension of the MLP')
    parser.add_argument('--device', type=int, default=0, help='device number')
    parser.add_argument('--csv_path', type=str, default='../datasets/', help='csv file path')
    parser.add_argument('--save', type=str, default='../trained_models/', help='model save path')
    parser.add_argument('--log', type=str, default='./', help='log path')
    parser.add_argument('--metrics', type=str, default='./', help='metrics path')
    parser.add_argument('--name', type=str, default='TransCrispr', help='model name')

    args = parser.parse_args()

    # Save the hyperparameters to a json file.
    params = vars(args)
    json_file = args.name + '.json'
    json_path = os.path.join(args.save, json_file)
    with open(json_path, 'w') as json_file:
        json.dump(params, json_file)

    epochs = args.ep
    batch_size = args.bs

    # Set the device to be cuda if available.
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    model = TransCrispr(
        nuc_emb_outputdim=args.emb_out_dim, conv1d_filters_num=args.conv_out_chn, conv1d_filters_size=args.k_sz,
        dropout=args.dp, transformer_num_layers=args.trln, transformer_final_fn=args.trfn,
        transformer_ffn_1stlayer=args.trffd, dense1=args.d1, dense2=args.d2, dense3=args.d3, MLP_out=args.physics_width
    ).to(device)

    # Define the weight decay.
    weight_decay = {'fc1.weight': args.L2_reg, 'fc1.bias': args.L2_reg, 'fc2.weight': args.L2_reg,
                    'fc2.bias': args.L2_reg}

    # Create an Adam optimizer with custom weight decay
    optimizer = Adam([
        {'params': [param for name, param in model.named_parameters()
                    if name not in weight_decay.keys()]},
        {'params': [param for name, param in model.named_parameters()
                    if name in weight_decay.keys()], 'weight_decay': 0}
    ], lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # Set the weight decay for the specific layers
    for name, param in model.named_parameters():
        if name in weight_decay.keys():
            param.requires_grad = True
            param.weight_decay = weight_decay[name]
        else:
            param.requires_grad = True
    criterion = get_loss()

    data_path = os.path.join(args.csv_path, "DeepHF.csv")
    data = get_csv_data(data_path)
    dataset = SequenceDataset(data[0], data[1])
    train_set, testval_set = train_test_split(dataset, test_size=0.1, random_state=42)
    test_set, val_set = train_test_split(testval_set, test_size=0.5, random_state=42)
    train_loader = create_dataloader(train_set, batch_size, shuffle=True, fn=collate_fn)
    val_loader = create_dataloader(val_set, 256, shuffle=False, fn=collate_fn)
    test_loader = create_dataloader(test_set, 256, shuffle=False, fn=collate_fn)

    path = os.path.join(args.csv_path, "CRISPRLearner.csv")
    CRISPRLearner_data = get_csv_data(path)
    CRISPRLearner_dataset = SequenceDataset(CRISPRLearner_data[0], CRISPRLearner_data[1])

    path = os.path.join(args.csv_path, "DeepSpCas9.csv")
    DeepSpCas9_data = get_csv_data(path)
    DeepSpCas9_dataset = SequenceDataset(DeepSpCas9_data[0], DeepSpCas9_data[1])

    path = os.path.join(args.csv_path, "DeepCRISPRhct116.csv")
    hct116_data = get_csv_data(path)
    hct116_dataset = SequenceDataset(hct116_data[0], hct116_data[1])

    path = os.path.join(args.csv_path, "DeepCRISPRhela.csv")
    hela_data = get_csv_data(path)
    hela_dataset = SequenceDataset(hela_data[0], hela_data[1])

    path = os.path.join(args.csv_path, "DeepCRISPRhek293t.csv")
    hek293t_data = get_csv_data(path)
    hek293t_dataset = SequenceDataset(hek293t_data[0], hek293t_data[1])

    path = os.path.join(args.csv_path, "DeepCRISPRhl60.csv")
    h160_data = get_csv_data(path)
    h160_dataset = SequenceDataset(h160_data[0], h160_data[1])

    path = os.path.join(args.csv_path, "test_Shkumatava.csv")
    Shkumatava_data = get_csv_data(path)
    Shkumatava_dataset = SequenceDataset(Shkumatava_data[0], Shkumatava_data[1])

    path = os.path.join(args.csv_path, "test_Gagnon.csv")
    Gagnon_data = get_csv_data(path)
    Gagnon_dataset = SequenceDataset(Gagnon_data[0], Gagnon_data[1])

    path = os.path.join(args.csv_path, "test_Labuhn.csv")
    Labuhn_data = get_csv_data(path)
    Labuhn_dataset = SequenceDataset(Labuhn_data[0], Labuhn_data[1])

    path = os.path.join(args.csv_path, "test_Shalem.csv")
    Shalem_data = get_csv_data(path)
    Shalem_dataset = SequenceDataset(Shalem_data[0], Shalem_data[1])

    path = os.path.join(args.csv_path, "test_KoikeYusa.csv")
    KoikeYusa_data = get_csv_data(path)
    KoikeYusa_dataset = SequenceDataset(KoikeYusa_data[0], KoikeYusa_data[1])

    path = os.path.join(args.csv_path, "test_XiXiang.csv")
    XiXiang_data = get_csv_data(path)
    XiXiang_dataset = SequenceDataset(XiXiang_data[0], XiXiang_data[1])


    #  Write the Spearman correlation coefficient of each test set to the csv file.
    def test(model, testset, device):
        model.eval()
        with torch.no_grad():
            test_loader = create_dataloader(testset, 32, shuffle=False, fn=collate_fn)
            test_corr = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                target = target.unsqueeze(1)
                output = model(data)
                # Calculate the Spearman correlation coefficient.
                corr, _ = sp.stats.spearmanr(output.cpu().detach().numpy(), target.cpu().detach().numpy())
                test_corr += corr
            test_corr /= len(test_loader)
        return test_corr

    log_dir = args.name + "_log"
    log_path = os.path.join(args.log, args.dir)
    writer = SummaryWriter(log_path)
    # Build a list for test_R.
    val_R = []

    # Train the model and write the loss and R to the tensorboard.
    def train_loop(model, train_loader, test_loader, optimizer, criterion, epochs, device, scheduler):
        for epoch in range(1, epochs + 1):
            train(model, train_loader, optimizer, criterion, device, scheduler)
            # Write the train the test loss and R to the tensorboard.
            model.eval()
            with torch.no_grad():
                train_loss = 0
                train_corr = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    target = target.unsqueeze(1)
                    output = model(data)
                    # Calculate the Spearman correlation coefficient.
                    corr, _ = sp.stats.spearmanr(output.cpu().detach().numpy(), target.cpu().detach().numpy())
                    train_corr += corr
                    train_loss += criterion(output, target).item()
                train_loss /= len(train_loader)
                train_corr /= len(train_loader)
                val_loss = 0
                val_corr = 0
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    target = target.unsqueeze(1)
                    output = model(data)
                    # Calculate the Spearman correlation coefficient.
                    corr, _ = sp.stats.spearmanr(output.cpu().detach().numpy(), target.cpu().detach().numpy())
                    val_corr += corr
                    val_loss += criterion(output, target).item()
                val_loss /= len(test_loader)
                val_corr /= len(test_loader)
                # Write the train the val loss to the tensorboard on the same graph.
                writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                # Write the train the val R to the tensorboard on the same graph.
                writer.add_scalars('R', {'train': train_corr, 'val': val_corr}, epoch)
            val_R.append(val_corr)
            # If the test R is the best, save the model.
            if val_corr == max(val_R):
                save_file = args.name + ".pt"
                save_path = os.path.join(args.save, save_file)
                torch.save(model.state_dict(), save_path)
                test_R = test(model, test_set, device)
                Learner_R = test(model, CRISPRLearner_dataset, device)
                SpCas9_R = test(model, DeepSpCas9_dataset, device)
                hct116_R = test(model, hct116_dataset, device)
                hek293t_R = test(model, hek293t_dataset, device)
                h160_R = test(model, h160_dataset, device)
                hela_R = test(model, hela_dataset, device)
                Gagnon_R = test(model, Gagnon_dataset, device)
                KoikeYusa_R = test(model, KoikeYusa_dataset, device)
                Labuhn_R = test(model, Labuhn_dataset, device)
                Shalem_R = test(model, Shalem_dataset, device)
                Shkumatava_R = test(model, Shkumatava_dataset, device)
                XiXiang_R = test(model, XiXiang_dataset, device)

                metrics_file = args.name + "_metrics.csv"
                metrics_path = os.path.join(args.save, metrics_file)
                with open(metrics_path, 'w') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow(['test', 'CRISPRLearner', 'DeepSpCas9', 'DeepCRISPRhct116',
                                         'DeepCRISPRhek293t', 'DeepCRISPRh160', 'DeepCRISPRhela',
                                         'Gagnon', 'KoikeYusa', 'Labuhn', 'Shalem', 'Shkumatava', 'XiXiang'])
                    writer_csv.writerow([test_R, Learner_R, SpCas9_R, hct116_R, hek293t_R, h160_R, hela_R,
                                         Gagnon_R, KoikeYusa_R, Labuhn_R, Shalem_R, Shkumatava_R, XiXiang_R])
        writer.close()


    train_loop(model, train_loader, test_loader, optimizer, criterion, epochs, device, scheduler)
