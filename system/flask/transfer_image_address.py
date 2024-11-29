# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:26:55 2024

@author: ZZJ
"""

import scipy.io as scio
import sqlite3
# 读取MAT文件
mat = scio.loadmat('G:/junior/graduate_design/DSPH-main/dataset/coco/index.mat')
print(mat.keys())
data = mat['index']

'''
connection = sqlite3.connect('D:/soft/sqlite/coco_image.db')
cursor = connection.cursor()

for i in range(len(data)):
    index = i
    address = data[i]
    
    # 插入数据到数据库
    cursor.execute("INSERT INTO images (id, address) VALUES (?, ?)", (index, address))

# 6. 提交更改并关闭连接
connection.commit()
cursor.close()
connection.close()
'''

def get_local_index():
    for i in range(len(data)):
        data[i] = data[i][34:]
        
    scio.savemat("G:/junior/graduate_design/DSPH-main/dataset/coco_fea/index_local.mat", {'index': data})
get_local_index()