import os
import pandas as pd


def make_prediction(gpu=0, threshold=0.1):
    command0 = "python find_target.py"
    os.system(command0)
    command1 = "python OnTarget_prediction.py --device "+ str(gpu)
    os.system(command1)
    command2 = "python genome_align.py"
    os.system(command2)
    command3 = "python OffTarget_prediction.py --device "+ str(gpu) + " --threshold " + str(threshold)
    os.system(command3)


def make_table(k=1):
    df = pd.read_csv("../tasks/off_target.csv")
    df1 = pd.read_csv("../tasks/off_target_predictions.csv")
    df['off_target_effect'] = df1['pred']
    # Only keep those with pred = 1.
    df = df[df['off_target_effect']==1]
    # For each #Id, count the number of data and make it as a new column.
    df2 = pd.read_csv("../tasks/on_target_predictions.csv")
    df = pd.merge(df, df2[['id', 'pred']], left_on='#Id', right_on='id', how='left')
    df.rename(columns={'pred': 'on_target_effect'}, inplace=True)
    df.drop('id', axis=1, inplace=True)
    df.drop('off_target_effect', axis=1, inplace=True)
    # Rank the data by 100*(1 - on_target_effect) - k * number of the data with the same #Id.
    df['score'] = 100*(1 - df['on_target_effect']) - k*df.groupby('#Id')['#Id'].transform('count')
    df['off_target_count'] = df.groupby('#Id')['#Id'].transform('count')
    # Add the #Id in the df2 to df if it is not in df.
    df3 = pd.DataFrame()
    for col in df.columns:
        if col not in df2.columns:
            df3[col] = None
    df3['crRNA'] = df2['Seq']
    df3['#Id'] = df2['id']
    df3['score'] = 100*(1 - df2['pred'])
    df3['off_target_count'] = 0
    df3_new = df3[~df3['#Id'].isin(df['#Id'])]
    df = pd.concat([df, df3_new], ignore_index=True)
    df.sort_values(by=['score'], ascending=True, inplace=True)
    df.drop('on_target_effect', axis=1, inplace=True)
    # Add the rank of #Id as a new column.
    df['rank'] = df['score'].rank(method='dense').astype(int)
    df.to_csv("../output/rank.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='device number')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for off-target prediction')
    parser.add_argument('--k', type=int, default=1, help='coefficient for off-target prediction')
    args = parser.parse_args()
    make_prediction(gpu=args.gpu, threshold=args.threshold)
    make_table(k=args.k)