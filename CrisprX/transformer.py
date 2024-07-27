# Define the Transformer model used in this project.
import torch
from torch import nn


# First define a feedforward network.
def feed_forward_network(d_model, diff):
    return nn.Sequential(
        nn.Linear(d_model, diff),
        nn.ReLU(),
        nn.Linear(diff, d_model)
    )


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute scaled dot product attention between query (q), key (k) and value (v) tensors.

    Args:
        - q: tensor of shape (..., seq_len_q, depth)
        - k: tensor of shape (..., seq_len_k, depth)
        - v: tensor of shape (..., seq_len_v, depth_v)
          where seq_len_k and seq_len_v are expected to be same.
        - mask: tensor of shape (..., seq_len_q, seq_len_k) or None

    Returns:
        - output: tensor of shape (..., seq_len_q, depth_v)

    """
    # matmul_qk shape: (..., seq_len_q, seq_len_k)
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))

    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # attention_weights shape: (..., seq_len_q, seq_len_k)
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    # output shape: (..., seq_len_q, depth_v)
    output = torch.matmul(attention_weights, v)

    return output


# Define the Multi-Head Attention layer.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask=None):
        batch_size = q.shape[0]
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.reshape(
            batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


# Define the encoder layer.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, diff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, diff)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask):
        # x: (batch_size, seq_len, d_model)
        # attention_output: (batch_size, seq_len, d_model)
        attention_output = self.mha(x, x, x, encoder_padding_mask)

        attention_output = self.dropout1(attention_output)
        out1 = self.layer_norm1(x + attention_output)

        ffn_output = self.ffn(out1)  # (batch_size, seq_len, d_model)
        # ffn_output = self.dropout2(ffn_output)

        out2 = self.layer_norm2(out1 + ffn_output)
        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, decoder_mask, encoder_decoder_padding_mask):
        # enc_output: (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(x, x, x, decoder_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, encoder_decoder_padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        # ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


# The following section is about positoinal encoding.
def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.FloatTensor([d_model]))
    return pos * angle_rates


def get_position_embedding(sentence_length, d_model):
    # pos.shape: (sentence_length, 1)
    # i.shape: (1, d_model)
    pos = torch.arange(sentence_length).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)

    angle_rads = get_angles(pos, i, d_model)  # (sentence_length, d_model)

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    position_embedding = torch.cat([sines, cosines], axis=-1)  # (sentence_length, d_model)

    position_embedding = position_embedding.unsqueeze(0)  # (1, sentence_length, d_model)

    return position_embedding.float()


# Define the encoder.
class EncoderModel(nn.Module):
    def __init__(self, num_layers, max_length, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length
        # Here the input is the output of a convolutional layer, not sequence, so we don't need the embedding layer.
        # self.embedding = nn.Embedding(input_vocab_size, self.d_model)
        # self.position_embedding = get_position_embedding(max_length, self.d_model)

        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)])

    def forward(self, x, encoder_padding_mask):
        # x: (batch_size, input_seq_len)
        input_seq_len = x.shape[1]
        # assert input_seq_len <= self.max_length
        assert input_seq_len <= self.max_length, "input_seq_len should be less or equal to self.max_length!"
        # Do the embedding, first transpose the input x, because in nn.Embedding, the input should be (seq_len, batch_size)
        # x = x.transpose(0,1)
        # x = self.embedding(x)
        # Transpose x back to batch size first.
        # x = x.transpose(0,1)

        # Add the linear layer.
        x *= torch.sqrt(torch.FloatTensor([self.d_model]).to(x.device))
        # x += self.position_embedding[:, :input_seq_len, :]
        # x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, encoder_padding_mask)
        # x.shape: (batch_size, input_seq_len, d_model)
        return x


# Define the decoder.
class DecoderModel(nn.Module):
    def __init__(self, num_layers, max_length, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        # self.embedding = nn.Embedding(target_vocab_size, d_model)
        # self.position_embedding = get_position_embedding(max_length, d_model)

        self.dropout = nn.Dropout(rate)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)])

    def forward(self, x, encoding_outputs, decoder_mask, encoder_decoder_padding_mask):
        # x: (batch_size, output_seq_len)
        output_seq_len = x.shape[1]
        assert output_seq_len <= self.max_length, 'output_seq_len should be less or equal to self.max_length!'
        # x = self.embedding(x)  # (batch_size, output_seq_len, d_model)
        x *= torch.sqrt(torch.FloatTensor([self.d_model]).to(x.device))
        # x += self.position_embedding[:, :output_seq_len, :]
        # x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, encoding_outputs, decoder_mask,
                                                   encoder_decoder_padding_mask)

        return x


# Finally, we construct the transformer model.
class Transformer(nn.Module):
    def __init__(self, num_layers, target_vocab_size,  max_length, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.encoder_model = EncoderModel(num_layers, max_length, d_model, num_heads, dff, rate)
        self.decoder_model = DecoderModel(num_layers, max_length, d_model, num_heads, dff, rate)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, encoding_padding_mask=None, decoder_mask=None,
                encoder_decoder_padding_mask=None):
        # encoding_outputs: (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(inp, encoding_padding_mask)
        # decoding_outputs: (batch_size, output_seq_len, d_model)
        decoding_outputs = self.decoder_model(tar, encoding_outputs, decoder_mask,
                                                                 encoder_decoder_padding_mask)
        # predictions: (batch_size, output_seq_len, target_vocab_size)
        predictions = self.final_layer(decoding_outputs)
        return predictions


if __name__ == "__main__":
    transformer = Transformer(num_layers=1, target_vocab_size=10, max_length=100, d_model=256, num_heads=8, dff=512)
    inp = torch.rand((64, 100, 256))
    tar = torch.rand((64, 100, 256))
    print(transformer(inp, tar)[0].shape)