from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/9715481_b3cb4114ff.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ant", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([8, 2, 4], [2, 5, 5])     # 一个是三个通道的均值，一个是三个通道的标准差值
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Nomalize", img_norm, 1)

# resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)

# Compose
trans_resize_2 = transforms.Resize(512)

# 相当于是一个管道，trans_resize_2转换的输出就是trans_totensor的输入；这里就是PIL -> PIL -> tensor
img_compose = transforms.Compose([trans_resize_2, trans_totensor])  # 组合几种变化

writer.close()
