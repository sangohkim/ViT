# ViT

Implementation of ViT-S/8. This model is pretrained on Tiny ImageNet.
Since Tiny ImageNet is small that it doesn't need finetuning
and this repository is for my personal study, I'll only conduct pretraining with downsized hyperparameters and model.

## ViT-S/8 Model Configuration
- Driven from [here](https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py#L95)
- Transformer's latent vector size: 384
- Self-Attention Module
  - num_heads: 6
  - mlp_dim: 1536
  - attention_dropout_rate: 0.0
  - dropout_rate: 0.0
  - num_layers: 12

## Steps to Reproduce

### 1. Download Tiny ImageNet

Due to my resource limitation, I decided to conduct pretraining with Tiny ImageNet instead of full ImageNet-21k or ImageNet-1k.

The brief configuration of Tiny ImageNet is given below.
- Image shape: (64, 64, 3)
- Num classes: 200
- Training set: 500 per class (100000 in total)
- Validation set: 50 per class (10000 in total)

Since test labels are not given, I'll treat validation set as test set. 
For validation set, I'll randomly sample 10 images per class from the training set (5000 images for validation set in total).
- training: 98000
- validation: 2000
- test: 10000

To download Tiny ImageNet, execute the commands below.
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

### 2. Refine the dataset structure

For the sake of implementation, I refined the given Tiny ImageNet structure. To do that, execute the commands below.
```
cd {YOUR_PATH}/ViT/raw-datasets/tiny-imagenet-200
python train_valid_split.py
```