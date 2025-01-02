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
        out = self.mlphead(out[:, 0])
        return out


class ViTEncoder(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()
        
        self.pos_enc = PositionalEncoding(args)  # includes [CLASS] token
        self.proj = nn.Linear(args.csize * args.patch_size ** 2, args.dsize, bias=False)
        self.tfenc = TransformerEncoder(args)
    
    def forward(self, x):
        out = self.proj(x)
        out = self.pos_enc(out)
        out = self.tfenc(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, args: ArgumentParser) -> None:
        super().__init__()
        
        self.class_token = nn.Parameter(torch.randn(args.dsize))  # [CLASS] token
        self.seq_len = (args.img_size // args.patch_size) ** 2
        self.pos_enc = nn.Parameter(torch.randn(self.seq_len + 1, args.dsize))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_len != x.shape[1]:
            raise NotImplementedError('Proper interpolation is not implemented yet')
        
        prepended_x = torch.cat([self.class_token.unsqueeze(0).repeat(x.shape[0], 1).unsqueeze(1), x], dim=1)
        out = prepended_x + self.pos_enc

        return out
    
    
class MLPHead(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.linear = nn.Linear(args.dsize, args.num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.softmax(out)

        return out
    

# This is just for test
if __name__ == '__main__':
    x = torch.randn(32, 64, 3 * 8 * 8)
    args = ArgumentParser()
    args.patch_size = 8
    args.img_size = 64
    args.csize = 3
    args.dsize = 384
    args.num_layers = 5
    args.num_heads = 6
    args.dff = 1536
    args.attn_drop = 0.0
    args.drop = 0.0
    args.num_classes = 200

    vit = ViT(args)
    out = vit(x)
    print(out.shape)  # Should be torch.Size([32, args.num_classes])
