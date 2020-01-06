import torch
import torch.nn.functional as F
from torch import nn
import math


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.proj_keys = nn.Linear(emb, heads * emb, bias=False)
        self.proj_queries = nn.Linear(emb, heads * emb, bias=False)
        self.proj_values = nn.Linear(emb, heads * emb, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b, t, emb = x.size() # batch_size, seq_length, embedding_size
        h = self.heads
        queries = self.proj_queries(x).view(b, t, h, emb) # from (b, t, h * emb) -> (b, t, h, emb)
        keys = self.proj_keys(x).view(b, t, h, emb)
        values = self.proj_values(x).view(b, t, h, emb)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, emb)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, emb)
        values = values.transpose(1, 2).contiguous().view(b * h, t, emb)


        #  get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(emb)
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, emb)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * emb)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads)
        self.layer_norm1 = nn.LayerNorm(emb)
        self.layer_norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.ReLU(),
            nn.Linear(4*emb, emb)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.layer_norm1(attended + x)
        ff = self.ff(x)
        return self.layer_norm2(ff + x)