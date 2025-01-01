import os
import shutil
import random
from tqdm import tqdm
PROJ_ROOT = '/Users/sangohkim/ksodev/ViT'
DSET_ROOT = f'{PROJ_ROOT}/raw-datasets/tiny-imagenet-200'
# Create an image directory for the train set
os.makedirs(f'{DSET_ROOT}/train/images', exist_ok=False)
# Create a directory for the validation set
os.makedirs(f'{DSET_ROOT}/train_val', exist_ok=False)
os.makedirs(f'{DSET_ROOT}/train_val/images', exist_ok=False)
labels = os.listdir(f'{DSET_ROOT}/train')

print('Moving images to the train and validation directories...')
with open(f'{DSET_ROOT}/train_val/train_val_annotations.txt', 'w') as tvaf, open(f'{DSET_ROOT}/train/train_annotations.txt', 'w') as taf:
    for label in tqdm(labels):
        if label == 'images':
            continue
        img_dir = f'{DSET_ROOT}/train/{label}/images'
        image_names = os.listdir(img_dir)
        random.shuffle(image_names)
        train_val_images = image_names[:10]
        train_images = image_names[10:]
        for image_name in train_val_images:
            tvaf.write(f'{image_name}\t{label}\n')
            shutil.move(f'{img_dir}/{image_name}', f'{DSET_ROOT}/train_val/images/{image_name}')
        for image_name in train_images:
            taf.write(f'{image_name}\t{label}\n')
            shutil.move(f'{img_dir}/{image_name}', f'{DSET_ROOT}/train/images/{image_name}')

print('Deleting the empty directories...')
for label in labels:
    if label == 'images':
        continue
    shutil.rmtree(f'{DSET_ROOT}/train/{label}')
