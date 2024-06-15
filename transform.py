from torchvision import transforms
from PIL import Image

# python的用法 -> tensor数据类型
# 通过transforms.ToTensor去解决两个问题：
# 1. 通过transforms如何使用
# 2. 为什么需要Tensor数据类型

img_path = "hymenoptera_data/train/ants/6743948_2b8c096dda.jpg"
img = Image.open(img_path)
print(img)

tensor_trans = transforms.ToTensor()    # ToTensor也是一个类，要先将其实例化
tensor_img = tensor_trans(img)

print(tensor_img)

