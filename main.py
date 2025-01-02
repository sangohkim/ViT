import argparse

from dataset import PreTrainingDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--csize', type=int, default=3)
    parser.add_argument('--dsize', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers in the TransformerEncoder')
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--dff', type=int, default=1536, help='Feedforward dimension of the TransformerEncoderLayer')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Dropout rate used in MultiHeadSelfAttention')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate used in the MLP of TransformerEncoderLayer')
    parser.add_argument('--num_classes', type=float, default=200)

    args = parser.parse_args()

    assert args.dsize % args.num_heads == 0, 'dsize should be divisible by num_heads'

    train_dataset = PreTrainingDataset(data_path='raw-datasets/tiny-imagenet-200/train', annotation_file_name='train_annotations.txt')
    val_dataset = PreTrainingDataset(data_path='raw-datasets/tiny-imagenet-200/train_val', annotation_file_name='train_val_annotations.txt')
    test_dataset = PreTrainingDataset(data_path='raw-datasets/tiny-imagenet-200/val', annotation_file_name='val_annotations.txt')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for images, labels in train_loader:
        print(images.shape)
        print(labels)
        break