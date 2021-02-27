import os
from sklearn.preprocessing import LabelEncoder
from torchvision.utils import make_grid
from torchvision import transforms
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np

data_path=[]
data_name=[]


for root, dirs, files in os.walk('train'):
    # 变量指定目录文件列表
    for image_file in files:
        image_path = os.path.join(root, image_file)
        data_path.append(image_path)
        data_name.append(image_file.split('.')[0])


print(len(data_path), len(data_name))
le = LabelEncoder()
le.fit(['cat', 'dog'])
data_label = le.transform(data_name)

print(data_label)




