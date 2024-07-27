# Define a model to predict the indel frequency.
import torch
from torch import nn
import torch.nn.functional as F
from CrisprX.transformer import Transformer, MultiHeadAttention
from CrisprX.utils import MLP, PositionalEncoding


class TransCrispr(nn.Module):
    """
    The model that does not use rnn layer but uses transformer instead.
    """
    def __init__(self, nuc_emb_outputdim=66, conv1d_filters_num=512, conv1d_filters_size=7, dropout=0.4,
                 transformer_num_layers=4,
                 transformer_final_fn=198, transformer_ffn_1stlayer=111, dense1=176, dense2=88, dense3=22, Seq_len=23,
                 Emb_dim=30, max_len=50, num_heads=8, transformer_dropout=0.1, MLP_out=256, MLP_hln=2, MLP_hun=150, MLP_dropout=0.05):
        super(TransCrispr, self).__init__()
        # Dealing with the nucleotide and dimer sequence.
        self.Seq_len = Seq_len
        self.emb_nuc = nn.Embedding(Emb_dim, nuc_emb_outputdim)
        self.emb_dimer = nn.Embedding(Emb_dim, nuc_emb_outputdim)
        self.conv1_nuc = nn.Conv1d(in_channels=nuc_emb_outputdim, out_channels=conv1d_filters_num,
                                   kernel_size=conv1d_filters_size, padding='same')
        self.conv1_dimer = nn.Conv1d(in_channels=nuc_emb_outputdim, out_channels=conv1d_filters_num,
                                     kernel_size=conv1d_filters_size, padding='same')

        self.dropout = nn.Dropout(dropout)
        # Transformer part.
        self.emb_pos = nn.Embedding(Emb_dim, conv1d_filters_num)
        self.conv2 = nn.Conv1d(in_channels=conv1d_filters_num, out_channels=conv1d_filters_num,
                               kernel_size=conv1d_filters_size, padding='same')
        self.conv3 = nn.Conv1d(in_channels=conv1d_filters_num, out_channels=conv1d_filters_num,
                               kernel_size=conv1d_filters_size, padding='same')
        self.transformer = Transformer(transformer_num_layers, transformer_final_fn, max_len, conv1d_filters_num,
                                       num_heads, transformer_ffn_1stlayer, transformer_dropout)
        # MLP part.
        self.flatten = nn.Flatten()
        self.physics = MLP(input_dim=11, output_layer_activation='Tanh', output_dim=MLP_out, output_use_bias=True,
                           hidden_layer_num=MLP_hln, hidden_layer_units_num=MLP_hun, hidden_layer_activation='ReLU',
                           dropout=MLP_dropout)
        self.fc1 = nn.Linear((transformer_final_fn + conv1d_filters_num) * Seq_len + MLP_out, dense1)
        self.fc2 = nn.Linear(dense1, dense2)
        self.fc3 = nn.Linear(dense2, dense3)
        self.output = nn.Linear(dense3, 1)

    def forward(self, x):
        # Split the input to nucleotide and dimer, and make the positional arrays.
        input_nuc = x[:, :self.Seq_len].type(torch.long)
        input_dimer = x[:, self.Seq_len:2 * self.Seq_len].type(torch.long)
        input_pos = torch.arange(0, self.Seq_len).repeat(x.shape[0], 1).to(x.device)
        # Deal with the physical numbers.
        numbers = x[:, 2 * self.Seq_len:].type(torch.float32)
        # Embedding.
        emb_nuc = self.emb_nuc(input_nuc)
        emb_dimer = self.emb_dimer(input_dimer)
        emb_nuc = emb_nuc.permute(0, 2, 1)
        emb_dimer = emb_dimer.permute(0, 2, 1)
        # Convolution.
        conv1_nuc = self.conv1_nuc(emb_nuc)
        conv1_nuc = F.leaky_relu(conv1_nuc, 1e-4)
        conv1_dimer = self.conv1_dimer(emb_dimer)
        conv1_dimer = F.leaky_relu(conv1_dimer, 1e-4)
        drop1_nuc = self.dropout(conv1_nuc)
        drop1_dimer = self.dropout(conv1_dimer)
        ori_seq = conv1_nuc + conv1_dimer
        drop_sep = drop1_nuc + drop1_dimer
        emb_pos = self.emb_pos(input_pos)
        emb_pos = emb_pos.permute(0, 2, 1)
        pos_ori = ori_seq + emb_pos
        pos_drop = drop_sep + emb_pos
        conv2 = self.conv2(pos_ori)
        conv2 = F.leaky_relu(conv2, 1e-4)
        conv3 = self.conv3(pos_drop)
        conv3 = F.leaky_relu(conv3, 1e-4)
        conv2 = conv2.permute(0, 2, 1)
        conv3 = conv3.permute(0, 2, 1)
        # Transformer.
        x = self.transformer(conv2, conv3)
        # Concatenate the original sequence and the transformed sequence.
        my_concat = lambda x: torch.cat([x[0], x[1]], dim=1)
        weight_1 = lambda x: x * 0.2
        weight_2 = lambda x: x * 0.8
        flat1 = self.flatten(pos_ori)
        flat2 = self.flatten(x)
        flat = my_concat([weight_1(flat1), weight_2(flat2)])
        features = self.physics(numbers)
        flat = torch.cat([flat, features], dim=1)
        fc1 = self.fc1(flat)
        fc1 = F.leaky_relu(fc1, 1e-4)
        fc1 = self.dropout(fc1)
        fc2 = self.fc2(fc1)
        fc2 = F.leaky_relu(fc2, 1e-4)
        fc2 = self.dropout(fc2)
        fc3 = self.fc3(fc2)
        fc3 = F.leaky_relu(fc3, 1e-4)
        fc3 = self.dropout(fc3)
        output = self.output(fc3)
        return output


class CrisprDNT(nn.Module):
    """
    An off-target model from a research paper.
    """
    def __init__(self, channel_num=64, d_enc=14, pool_ks=2, rnn_units=32, rnn_num_layers=1, rnn_drop=0, seq_len=23, num_heads=8,
                 dense1=512, dense2=64, dense3=512, dense4=64, dense_drop=0.1, out1=256, out2=64, out_drop=0.25):
        super(CrisprDNT, self).__init__()

        # Make sure the dimensions are correct.
        assert dense2 == 2*rnn_units and dense4 == 2*rnn_units

        # Convolution and pooling.
        self.channel_num = channel_num
        self.seq_len = seq_len

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channel_num,
                               kernel_size=(1, d_enc),
                               padding=0)

        self.batchnorm = nn.BatchNorm2d(num_features=channel_num)
        self.average_pooling = nn.AvgPool1d(kernel_size=pool_ks)
        self.max_pooling = nn.MaxPool1d(kernel_size=pool_ks)

        # BiLSTM.
        if rnn_num_layers == 1:
            rnn_drop = 0

        self.bilstm = nn.LSTM(input_size=d_enc+channel_num*round(2/pool_ks),
                              hidden_size=rnn_units,
                              num_layers=rnn_num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=rnn_drop)

        # Multi-head attention.
        self.layernorm = nn.LayerNorm(normalized_shape=2*rnn_units)
        self.pos_enc = PositionalEncoding(d_model=2*rnn_units, max_len=seq_len)
        self.mha1 = MultiHeadAttention(d_model=2*rnn_units, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=2*rnn_units, num_heads=num_heads)
        self.dense1 = nn.Sequential(nn.Linear(2*rnn_units, dense1), nn.ReLU(), nn.Dropout(dense_drop))
        self.dense2 = nn.Sequential(nn.Linear(dense1, dense2), nn.ReLU(), nn.Dropout(dense_drop))
        self.dense3 = nn.Sequential(nn.Linear(2*rnn_units, dense3), nn.ReLU(), nn.Dropout(dense_drop))
        self.dense4 = nn.Sequential(nn.Linear(dense3, dense4), nn.ReLU(), nn.Dropout(dense_drop))

        # Output.
        self.flatten = nn.Flatten()
        self.out1 = nn.Sequential(nn.Linear(2*seq_len*rnn_units, out1), nn.ReLU(), nn.Dropout(out_drop))
        self.out2 = nn.Sequential(nn.Linear(out1, out2), nn.ReLU(), nn.Dropout(out_drop))
        self.out3 = nn.Sequential(nn.Linear(out2, 2), nn.Softmax(dim=-1))

    def forward(self, x):
        # x: (batch_size, 23, 14), make x as if they are batch of figures.
        x = x.float()
        x1 = x
        x = x.unsqueeze(1) # (batch_size, 1, 23, 14)
        x = self.conv1(x) # (batch_size, channel_num, 23, 1)
        x = F.relu(x)
        x = self.batchnorm(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.channel_num, -1)
        x = x.permute(0, 2, 1) # (batch_size, 23, channel_num)
        avg = self.average_pooling(x) # (batch_size, 23, channel_num//2)
        max = self.max_pooling(x) # (batch_size, 23, channel_num//2)
        x = torch.concat((x1, avg, max), dim=-1)
        x, _ = self.bilstm(x) # (batch_size, 23, 2*rnn_units)
        x = self.layernorm(x)
        pe = self.pos_enc(x)
        attention = self.mha1(pe, pe, pe)
        res = attention + pe
        layer_norm = self.layernorm(res) # (batch_size, 23, 2*rnn_units)
        linear1 = self.dense1(layer_norm) # (batch_size, 23, dense1)
        linear2 = self.dense2(linear1) # (batch_size, 23, dense2)
        res = layer_norm + linear2
        layer_norm = self.layernorm(res) # (batch_size, 23, 2*rnn_units)
        attention = self.mha2(layer_norm, layer_norm, layer_norm)
        res = attention + layer_norm
        layer_norm = self.layernorm(res) # (batch_size, 23, 2*rnn_units)
        linear1 = self.dense3(layer_norm) # (batch_size, 23, dense3)
        linear2 = self.dense4(linear1) # (batch_size, 23, dense4)
        res = layer_norm + linear2 # (batch_size, 23, 2*rnn_units)
        layer_norm = self.layernorm(res) # (batch_size, 23, 2*rnn_units)
        x = self.flatten(layer_norm) # (batch_size, 23*2*rnn_units)
        x = self.out1(x) # (batch_size, out1)
        x = self.out2(x) # (batch_size, out2)
        x = self.out3(x) # (batch_size, 2)
        return x


if __name__=="__main__":
    model = TransCrispr()
    print('TransCrispr:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = CrisprDNT()
    print('CrisprDNT:',sum(p.numel() for p in model.parameters() if p.requires_grad))
