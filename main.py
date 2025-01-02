import argparse
from argparse import ArgumentParser
import torch.nn as nn
import torch

from tqdm import tqdm
from utils.dataset import PreTrainingDataset
from torch.utils.data import DataLoader
from models.vit import ViT


def train(args: ArgumentParser):
    train_dataset = PreTrainingDataset(data_path='raw-datasets/tiny-imagenet-200/train', annotation_file_name='train_annotations.txt')
    val_dataset = PreTrainingDataset(data_path='raw-datasets/tiny-imagenet-200/train_val', annotation_file_name='train_val_annotations.txt')
    test_dataset = PreTrainingDataset(data_path='raw-datasets/tiny-imagenet-200/val', annotation_file_name='val_annotations.txt')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ViT(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                 betas=(args.beta1, args.beta2), eps=args.epsilon, 
                                 weight_decay=args.weight_decay)
    
    for epoch in tqdm(range(args.num_epochs)):
        ...


def train_epoch(args: ArgumentParser, model: nn.Module, loader: DataLoader):
    ...
    

def eval(args: ArgumentParser, model: nn.Module, loader: DataLoader):
    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_type', type=str, choices=['pretraining', 'finetuning'], default='pretraining')

    # ViT-related arguments
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--csize', type=int, default=3)

    # Transformer-related arguments
    parser.add_argument('--dsize', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers in the TransformerEncoder')
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--dff', type=int, default=1536, help='Feedforward dimension of the TransformerEncoderLayer')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Dropout rate used in MultiHeadSelfAttention')
    parser.add_argument('--drop', type=float, default=0.1, help='Dropout rate used in the MLP of TransformerEncoderLayer')
    parser.add_argument('--num_classes', type=float, default=200)
    
    # Optimizer(Adam) related arguments
    parser.add_argument('--lr', type=float, default=3e-3)  # Base LR
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.3)  # Small dataset needs stronger regularization

    # LR scheduler related arguments
    # Since there is only linear warmup in official implementation, I followed it
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--decay_type', type=str, choices=['linear', 'cosine'], default='cosine')
    parser.add_argument('--linear_end', typp=float, default=1e-5, help='Final learning rate after linear decay') # In case of cosine decay, this is not used.

    # Training related arguments
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=4096)

    args = parser.parse_args()

    assert args.dsize % args.num_heads == 0, 'dsize should be divisible by num_heads'
    assert args.train_type == 'pretraining', 'Only pretraining is supported for now'

    train(args)
    # TODO: eval
