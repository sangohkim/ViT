import os
import torch
import random

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple


class PreTrainingDataset(Dataset):
    def __init__(self, data_path: str, annotation_file_name: str, random_state: int = 42):
        self.data_path = data_path
        self.image_names = os.listdir(f'{data_path}/images')

        random.seed(random_state)
        random.shuffle(self.image_names)
        
        self.name_to_slabel = dict()
        with open(f'{data_path}/{annotation_file_name}', 'r') as f:
            self.annotations = f.readlines()
            for annotation in self.annotations:
                annotation_split = annotation.strip().split('\t')
                assert len(annotation_split) == 2 or len(annotation_split) == 6, 'Annotation file should have 2 or 6 columns'

                image_name, label = annotation.strip().split('\t')[:2]
                self.name_to_slabel[image_name] = label
        
        self.slabel_to_nlabel = dict()
        labels = list(set(self.name_to_slabel.values()))
        for i, label in enumerate(labels):
            self.slabel_to_nlabel[label] = i

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_name = self.image_names[idx]
        label = self.slabel_to_nlabel[self.name_to_slabel[image_name]]

        image = Image.open(f'{self.data_path}/images/{image_name}')
        image = self.transforms(image)

        return image, label
