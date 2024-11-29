# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:33:00 2024

@author: ZZJ
"""

import numpy as np
from scipy.spatial.distance import euclidean

def compute_similarity(text_fea, image_fea):
    # 计算欧氏距离
    
    distances = [euclidean(text_fea, img) for img in image_fea]
    return distances

def top_k_similar(text_fea, image_fea, k=10):
    # 计算相似度
    distance = compute_similarity(text_fea, image_fea)
    # 找出最高的K个相似度对应的索引
    top_k_indices = np.argsort(distance)[:k]
    return top_k_indices


