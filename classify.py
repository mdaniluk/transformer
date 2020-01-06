import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from model.transformer import CTransformer
from torchtext import data, datasets


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return f'cuda:{np.argmax(memory_available)}'


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab-size", type=int, default=50000,
                        help="Number of words in vocabulary")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of trained epochs")
    parser.add_argument("--embedding-size", type=int, default=128,
                        help="Size of transformer input embedding")
    parser.add_argument("--heads", type=int, default=8,
                        help="Number of heads in self-attentnion module")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of heads in self-attentnion module")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length. Longer sequences are clipped")
    parser.add_argument("--lr-warmup", type=int, default=10000,
                        help="Learning rate warmup")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    return parser


def get_data(params):
    print("Preparing IMDB dataset")
    text = data.Field(lower=True, include_lengths=True, batch_first=True)
    label = data.Field(sequential=False)
    train, test = datasets.IMDB.splits(text, label)
    text.build_vocab(train, max_size = params.vocab_size-2)  # - 2 for <unk> and <pad>
    label.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=params.batch_size,
                                device=params.device)
    return train_iter, test_iter


def train(params):
    train_iter, test_iter = get_data(params)
    model = CTransformer(emb=params.embedding_size,
                            heads=params.heads,
                            num_layers=params.num_layers,
                            seq_length=params.max_seq_length,
                            num_tokens=params.vocab_size,
                            num_classes=2,
                            device=params.device).to(params.device)
    optimizer = torch.optim.Adam(lr=params.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (params.lr_warmup / params.batch_size), 1.0))

    for epoch_idx in range(params.num_epochs):
        print(f"Epoch: {epoch_idx}")
        for batch_idx, batch in enumerate(train_iter):
            optimizer.zero_grad()
            text = batch.text[0]
            label = batch.label - 1
            if text.shape[1] > params.max_seq_length:
                text = text[:, :params.max_seq_length]
            output = model(text)
            loss = F.nll_loss(output, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
                model.eval()
                total, corrrect = 0.0, 0.0
                for batch in test_iter:
                    text = batch.text[0]
                    label = batch.label - 1
                    if text.size(1) > params.max_seq_length:
                        text = text[:, :params.max_seq_length]
                    out = model(text).argmax(dim=1)

                    total += float(text.size(0))
                    corrrect += float((label == out).sum().item())

                acc = corrrect / total
                print(f'Validation accuracy {acc:.3}')
        model.train()


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    device = get_free_gpu()
    params.device = device
    print(params)
    train(params)
