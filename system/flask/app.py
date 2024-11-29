# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:38:45 2024

@author: ZZJ
"""

from flask import Flask, request, jsonify, render_template
import scipy.io as scio
import torch
import json
import numpy as np
from model import build_model
import random
from simple_tokenizer import SimpleTokenizer as Tokenizer
from scipy.spatial.distance import euclidean
from utils import compute_similarity,top_k_similar
from flask_cors import CORS
"框架搭建"
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # 允许/api/下所有路由的跨域请求，来自任意来源

"模型导入"
clipPath="./ViT-B-32.pt"
tokenizer = Tokenizer()
model = torch.jit.load(clipPath, map_location="cpu").eval()
state_dict = model.state_dict()
model = build_model(state_dict)
model = model.to("cuda")
image_fea = scio.loadmat('G:/junior/毕业论文/DSPH-main/dataset/coco_fea/image_fea.mat')['image_fea']
index2address = scio.loadmat('G:/junior/毕业论文/DSPH-main/dataset/coco_fea/index_local.mat')['index']
def load_text(text):
    maxWords = 32 
    captions = text
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                          "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    words = tokenizer.tokenize(captions)
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = maxWords - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    caption = tokenizer.convert_tokens_to_ids(words)
    while len(caption) < maxWords:
        caption.append(0)
    caption = torch.tensor(caption)

    return caption
def get_address_by_index(indexs):
    address_pool = []
    for i in indexs:
        address_pool.append(index2address[i])
    return address_pool

def open_image(address_pool):
    return 
@app.route('/api/text2image', methods=['POST'])
def encoder_text():
    text = request.json.get('text')
    #print("text",text)
    text = load_text(text)
    text = text.unsqueeze(0)
    text = text.to("cuda")
    text_fea = model.encode_text(text).squeeze(0).detach().cpu().numpy()
    top_k_indices = top_k_similar(text_fea, image_fea)
    image_address = get_address_by_index(top_k_indices)
        
    # 返回 JSON 格式的响应给客户端
    return jsonify(image_address)
if __name__ == '__main__':
    app.run()
    #top_k_indices = encoder_text()
