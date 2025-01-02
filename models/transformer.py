import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.d_k = args.dsize // args.num_heads
        self.query = nn.Linear(args.dsize, args.dsize)
        self.key = nn.Linear(args.dsize, args.dsize)
        self.value = nn.Linear(args.dsize, args.dsize)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        Q = self.query(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, -1)

        return out
    

# For test
if __name__ == '__main__':
    x = torch.randn(32, 64, 384)
    args = ArgumentParser()
    args.dsize = 384
    args.num_layers = 5
    args.num_heads = 6
    args.dff = 1536
    args.attn_drop = 0.0
    args.drop = 0.0
    args.num_classes = 200

    tfenc = TransformerEncoder(args)
    out = tfenc(x)
    print(out.shape)  # Should be torch.Size([32, 64, args.dsize])
