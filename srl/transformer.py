import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# reference implementation:
# https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


# Embedding snippet from https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
def Embedding(num_embeddings, embedding_dim, padding_idx=0):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


# reference implementation:
# https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_length=1024):
        super(PositionalEmbedding, self).__init__()

        print("positional dim", dim)
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
            0
        )
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_length, -1)

        self.emb = emb

    def forward(self, seq_len):
        return self.emb[:seq_len].unsqueeze(0).detach()


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        positional_dim,
        embed_dropout,
        num_layers,
        embed_dim,
        num_heads,
        attention_dropout,
        activation_dropout,
        ffn_emb_dim,
        dropout,
    ):
        super(TransformerEncoder, self).__init__()

        print("Initializing Transfomer")
        print(
            "Input dim", input_dim, "Pos dim", positional_dim, "Emb drop", embed_dropout
        )
        print("Num layers", num_layers, "hidden dim", embed_dim)
        print("Num heads", num_heads, "FFN emb", ffn_emb_dim)
        print(
            "Attn drop",
            attention_dropout,
            "Act drop",
            activation_dropout,
            "Dropout",
            dropout,
        )

        self.pos_emb = PositionalEmbedding(positional_dim)
        self.input_proj = Linear(input_dim + positional_dim, embed_dim, bias=True)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerLayer(
                    embed_dim,
                    num_heads,
                    attention_dropout,
                    activation_dropout,
                    ffn_emb_dim,
                    dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(embed_dropout)

    def forward(self, x, mask):
        batch_size, length, dim = x.shape

        pos_emb = self.pos_emb(length).to(x.device)
        x = torch.cat([x, pos_emb.expand(x.shape[0], -1, -1)], 2)
        x = self.input_proj(x)

        x = self.dropout(x)

        # B T C > T B C
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)

        return x.transpose(0, 1)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attention_dropout,
        activation_dropout,
        ffn_emb_dim,
        dropout,
    ):
        super(TransformerLayer, self).__init__()

        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, num_heads, dropout=attention_dropout
        )
        self.activation = nn.ReLU()
        self.fc1 = Linear(self.embed_dim, ffn_emb_dim)
        self.fc2 = Linear(ffn_emb_dim, self.embed_dim)
        self.layer_norm_1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(p=activation_dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        residual = x
        x = self.layer_norm_1(x)
        x = self.self_attn(x, mask)
        x = self.dropout2(x)
        x = residual + x

        residual = x
        x = self.layer_norm_2(x)
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.proj = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, mask):
        # shape of x: T B C

        length, batch_size, emb_dim = x.size()
        q, k, v = F.linear(x, self.proj, self.bias).chunk(3, dim=-1)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(length, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(length, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(length, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = attn_weights.view(batch_size, self.num_heads, length, length)
        attn_weights = attn_weights.masked_fill(
            mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
        )
        attn_weights = attn_weights.view(batch_size * self.num_heads, length, length)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        attn = (
            attn.transpose(0, 1).contiguous().view(length, batch_size, self.embed_dim)
        )
        attn = self.out_proj(attn)

        return attn
