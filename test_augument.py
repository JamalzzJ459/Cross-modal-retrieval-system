# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:17:26 2024

@author: ZZJ
"""
from transformers import BertTokenizer, BertModel
import torch
import nltk
import re
import torchvision.transforms as transforms
import moco.loader
import moco.builder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]

image_augument = moco.loader.TwoCropsTransform_local(transforms.Compose(augmentation))
image_path = "G:/junior/毕业论文/DSPH-main/dataset/Cleared-Set/archive/mirflickr25k/mirflickr/im187.jpg"
image = Image.open(image_path).convert("RGB")
image_1,image_2 = image_augument(image)
image_1 = np.transpose(image_1.numpy(), (1, 2, 0)) 
image_2 = np.transpose(image_2.numpy(), (1, 2, 0)) 
fig, axes = plt.subplots(1, 2)

# 在第一个子图中展示第一张图片
axes[0].imshow(image_1)
axes[0].axis('off')  # 关闭坐标轴
axes[0].set_title('Image 1')

# 在第二个子图中展示第二张图片
axes[1].imshow(image_2)
axes[1].axis('off')  # 关闭坐标轴
axes[1].set_title('Image 2')

# 显示图像
plt.show()