import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()

        self.vitencoder = ViTEncoder(args)
        self.mlphead = MLPHead(args)
    
    def forward(self, x):
        out = self.vitencoder(x)
        out = self.mlphead(out)
        return out


class ViTEncoder(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()
        
        self.pos_enc = PositionalEncoding(args)  # includes [CLASS] token
        self.proj = nn.Linear(args.csize * args.patch_size ** 2, args.dsize, bias=False)
        self.tfenc = TransformerEncoder(args)
    
    def forward(self, x):
        out = self.pos_enc(x)
        out = self.proj(out)
        out = self.tfenc(out)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()
        
        self.pos_enc = nn.Parameter(torch.randn(1, args.img_size // args.psize, args.psize ** 2 * args.csize))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
        return x
    
    
class MLPHead(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.linear = nn.Linear(args.dsize, args.num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.softmax(out)

        return out
