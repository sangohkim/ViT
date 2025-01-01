import torch
import torch.nn as nn

from argparse import ArgumentParser


class TransformerEncoder(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()
        
        self.msa_layers = nn.Sequential(*[TransformerEncoderLayer(args) for _ in range(args.num_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.msa_layers(x)
        
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()

        self.layernorm1 = nn.LayerNorm(args.dsize)
        self.msa = MultiHeadSelfAttention(args)
        self.attn_dropout = nn.Dropout(args.attn_drop)
        self.layernorm2 = nn.LayerNorm(args.dsize)
        self.mlp = nn.Sequential(
            nn.Linear(args.dsize, args.dff),
            nn.Dropout(args.drop),
            nn.GELU(),
            nn.Linear(args.dff, args.dsize),
            nn.Dropout(args.drop),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layernorm1(x)
        out = self.msa(out) + out
        out = self.attn_dropout(out)
        out = self.layernorm2(out)
        out = self.mlp(out) + out

        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()
        
        self.num_heads = args.num_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
        return x