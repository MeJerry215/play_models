import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from PIL import Image

import os

# paths = os.listdir("train/dogs")
# for path in paths:
#     try:
#         img_path = os.path.join("train/dogs", path)
#         with open(img_path, 'rb') as f:
#             img = Image.open(f)
#             img_rgb = img.convert('RGB')
#     except:
#         print("remove", img_path)
#         os.remove(img_path)

def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except:
        return None

# 加载数据集

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6821, 0.6261, 0.6226), (0.3039, 0.3045, 0.2991))
])


dataset = ImageFolder(root='my_crawler/', transform=transform_train, loader=pil_loader)
print(dataset.class_to_idx)

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
# 获取所有图像的张量并将它们放入列表中
tensor_list = []
for img, label in dataset:
    tensor_list.append(img)

# 将所有张量堆叠成一个张量数组
tensor_array = torch.stack(tensor_list)

# 计算均值和标准差
mean = torch.mean(tensor_array, dim=(0, 2, 3))
std = torch.std(tensor_array, dim=(0, 2, 3))
print('Mean:', mean)
print('Std:', std)