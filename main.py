from dataset import PreTrainingDataset
from torch.utils.data import DataLoader

train_dataset = PreTrainingDataset(data_path='raw-datasets/tiny-imagenet-200/train', annotation_file_name='train_annotations.txt')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for images, labels in train_loader:
    print(images.shape)
    print(labels)
    break