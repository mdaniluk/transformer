import torch
import torch.nn.functional as F
from torch import nn
import math
from .modules import TransformerBlock


class CTransformer(nn.Module):
    def __init__(self, emb, heads, num_layers, seq_length, num_tokens, num_classes, device):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_emb = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        self.tblocks = nn.ModuleList([TransformerBlock(emb=emb, heads=heads) for _ in range(num_layers)])
        self.toprobs = nn.Linear(emb, num_classes)
        self.device = device

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, emb = tokens.size()
        positions = torch.arange(t, device=self.device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, emb)

        x = tokens + positions

        for block in self.tblocks:
            x = block(x)

        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
