# The utils for this project.
import torch.nn as nn
import torch
import numpy as np
import math

class Nuc_NtsTokenizer():
    """
    The tokenizer for nucleotide sequences.
    """
    def __init__(self):
        self.dic = [a for a in 'ATCG']
        self.c2i = {c:i for i,c in enumerate(self.dic)}
        self.i2c = {i:c for i,c in enumerate(self.dic)}

    def tokenize(self, seq):
        result = []
        for c in seq:
            result.append(self.c2i[c])
        return result

    def detokenize(self, idx):
        result = ''
        for i in idx:
            result += self.i2c[i]
        return result

class Dimer_NtsTokenizer():
    """
    The tokenizer for dimer sequences.
    """
    def __init__(self):
        self.dic = [a + b for a in 'ATCG' for b in 'ATCG']
        self.dic += [a + 'X' for a in 'ATCG']
        self.c2i = {c:i for i,c in enumerate(self.dic)}
        self.i2c = {i:c for i,c in enumerate(self.dic)}

    def tokenize(self, seq):
        result = []
        for i in range(len(seq)):
            result.append(self.c2i[(seq + 'X')[i:i+2]])
        return result

    def detokenize(self, idx):
        result = ''
        for i in idx:
            result += self.i2c[i]
        return result


class MLP(nn.Module):
    """
    The MLP model.
    """
    def __init__(self, input_dim, output_layer_activation, output_dim, output_use_bias,
                 hidden_layer_num, hidden_layer_units_num, hidden_layer_activation, dropout):
        super(MLP, self).__init__()

        if output_layer_activation == 'Sigmoid' or output_layer_activation == 'Tanh':
            hidden_layer_num -= 1

        # Define the hidden layers of the MLP
        self.hidden_layers = nn.ModuleList()
        ini_hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_units_num),
            getattr(nn, hidden_layer_activation)(),
            nn.Dropout(dropout))
        self.hidden_layers.append(ini_hidden_layer)
        for _ in range(hidden_layer_num-1):
            layer = nn.Sequential(
                nn.Linear(hidden_layer_units_num, hidden_layer_units_num),
                getattr(nn, hidden_layer_activation)(),
                nn.Dropout(dropout))
            self.hidden_layers.append(layer)
        # Define the output layer of the MLP
        if output_layer_activation == 'Sigmoid' or output_layer_activation == 'Tanh':
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_layer_units_num, hidden_layer_units_num),
                getattr(nn, hidden_layer_activation)(),
                nn.Linear(hidden_layer_units_num, output_dim, bias=output_use_bias),
                getattr(nn, output_layer_activation)())
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_layer_units_num, output_dim, bias=output_use_bias),
                getattr(nn, output_layer_activation)())

    def forward(self, x):
        # Apply the hidden layers of the MLP
        for layer in self.hidden_layers:
            x = layer(x)
        # Apply the output layer of the MLP
        x = self.output_layer(x)
        return x


# For the off-target model.
def position():
    """
    The position encoding.
    The sgRNA region is 20bp, and the PAM region is 3bp.
    """
    position = np.zeros((23, 1))
    for i in range(23):
        if i>=20:
            position[i] = 1
    return position


def one_hot(seq):
    """
    The one-hot encoding.
    """
    dic = {'A':0, 'T':1, 'C':2, 'G':3}
    result = np.zeros((len(seq), 4))
    for i in range(len(seq)):
        result[i][dic[seq[i]]] = 1
    return result


def mismatch(seq1, seq2):
    """
    The mismatch type encoding.
    """
    assert len(seq1) == len(seq2)

    dict ={'AA':0, 'GG':0, 'CC':0, 'TT':0, 'AG':1, 'GA':1, 'CT':1, 'TC':1,
           'AC':-1, 'CA':-1, 'GT':-1, 'TG':-1, 'AT':-1, 'TA':-1, 'CG':-1, 'GC':-1}
    result = np.zeros((len(seq1), 1))
    for i in range(len(seq1)):
        result[i] = dict[seq1[i]+seq2[i]]
    return result


def XOR(one_hot1, one_hot2):
    """
    The XOR operation.
    """
    assert len(one_hot1)==len(one_hot2)
    result = np.zeros((len(one_hot1), 4))
    for i in range(len(one_hot1)):
        for j in range(4):
            if one_hot1[i][j]==one_hot2[i][j]:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result


def encode(seq1, seq2):
    """
    The encoding function.
    """
    assert len(seq1) == len(seq2)
    result = np.concatenate((one_hot(seq1), one_hot(seq2),  XOR(one_hot(seq1), one_hot(seq2)),mismatch(seq1, seq2), position()), axis=1)
    return result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        """
        :param d_model: the dimension of the model
        :param dropout: the dropout rate
        :param max_len: the max length of the sequence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: the input sequence
        :return: the input sequence added the positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    seq1 = 'GTGCTTTTTTTTTTTTTTTCAGG'
    seq2 = 'CTGTTTTTTTTTTTTTTTTCAGA'
    print(encode(seq1, seq2))