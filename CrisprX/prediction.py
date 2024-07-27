# This script is using the trained models to predict.

if __name__ == "__main__":
    from CrisprX.models import TransCrispr, CrisprDNT
    from CrisprX.TransCrispr_train import get_csv_data as get_on_target_csv_data
    from CrisprX.CrisprDNT_train import get_csv_data as get_off_target_csv_data
    from CrisprX.TransCrispr_train import SequenceDataset, create_dataloader
    from CrisprX.TransCrispr_train import collate_fn as on_target_collate_fn
    from CrisprX.CrisprDNT_train import collate_fn as off_target_collate_fn
    import json
    import argparse
    import os
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--on_target', type=str, default='TransCrispr', help='on-target model name')
    parser.add_argument('--off_target', type=str, default='CrisprDNT', help='off-target model name')
    parser.add_argument('--load', type=str, default='../trained_models/', help='models load path')
    parser.add_argument('--task', type=str, default='../tasks/', help='tasks path')
    parser.add_argument('--on_target_csv', type=str, default='on_target.csv', help='on-target csv file name')
    parser.add_argument('--off_target_csv', type=str, default='off_target.csv', help='off-target csv file name')
    parser.add_argument('--device', type=int, default=0, help='device number')

    args = parser.parse_args()

    on_target_json = args.on_target + '.json'
    on_target_json_path = os.path.join(args.load, on_target_json)

    off_target_json = args.off_target + '.json'
    off_target_json_path = os.path.join(args.load, off_target_json)

    with open(on_target_json_path, 'r') as f:
        on_target_config = json.load(f)

    with open(off_target_json_path, 'r') as f:
        off_target_config = json.load(f)

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    on_target = TransCrispr(nuc_emb_outputdim=on_target_config["emb_out_dim"],
                            conv1d_filters_num=on_target_config["conv_out_chn"],
                            conv1d_filters_size=on_target_config["k_sz"],
                            dropout=on_target_config["dp"], transformer_num_layers=on_target_config["trln"],
                            transformer_final_fn=on_target_config["trfn"],
                            transformer_ffn_1stlayer=on_target_config["trffd"],
                            dense1=on_target_config["d1"], dense2=on_target_config["d2"],
                            dense3=on_target_config["d3"], MLP_out=on_target_config["physics_width"]).to(device)

    off_target = CrisprDNT(channel_num=off_target_config['channel_num'],
                           pool_ks=off_target_config['pool_ks'], rnn_units=off_target_config['rnn_units'],
                           rnn_num_layers=off_target_config['rnn_num_layers'],
                           rnn_drop=off_target_config["rnn_drop"],
                           num_heads=off_target_config["num_heads"], dense1=off_target_config["dense1"],
                           dense2=off_target_config["dense2"], dense3=off_target_config["dense3"],
                           dense4=off_target_config["dense4"], dense_drop=off_target_config["dense_drop"],
                           out1=off_target_config["out1"], out2=off_target_config["out2"]).to(device)

    on_target.load_state_dict(torch.load(os.path.join(args.load, args.on_target + '.pt')))

    off_target.load_state_dict(torch.load(os.path.join(args.load, args.off_target + '.pt')))

    on_target.eval()

    off_target.eval()

    on_target_csv_path = os.path.join(args.task, args.on_target_csv)
    on_target_data = get_on_target_csv_data(on_target_csv_path)
    on_target_dataset = SequenceDataset(on_target_data[0], on_target_data[1])
    on_target_dataloader = create_dataloader(on_target_dataset, 1000, on_target_collate_fn, drop_last=False)

    off_target_csv_path = os.path.join(args.task, args.off_target_csv)
    off_target_data = get_off_target_csv_data(off_target_csv_path)
    off_target_dataset = SequenceDataset(off_target_data[0], off_target_data[1])
    off_target_dataloader = create_dataloader(off_target_dataset, 1000, off_target_collate_fn, drop_last=False)

    # on-target prediction
    on_target_pred = []
    for data, target in on_target_dataloader:
        data = data.to(device)
        pred = on_target(data)
        on_target_pred.extend(pred.detach().cpu().numpy())

    # Save on-target prediction as txt file
    on_target_pred_path = os.path.join(args.task, 'on_target_pred.txt')
    with open(on_target_pred_path, 'w') as f:
        for item in on_target_pred:
            f.write("%s\n" % item)

    # off-target prediction
    off_target_pred = []
    for data, target in off_target_dataloader:
        data = data.to(device)
        pred = off_target(data)
        off_target_pred.extend(pred.detach().cpu().numpy())

    # Save off-target prediction as txt file
    off_target_pred_path = os.path.join(args.task, 'off_target_pred.txt')
    with open(off_target_pred_path, 'w') as f:
        for item in off_target_pred:
            f.write("%s\n" % item)
