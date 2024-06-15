from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "hymenoptera_data/train/bees/85112639_6e860b0469.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 2, dataformats='HWC')   # 从PIL到numpy，需要在add_image()中指定shape中每一个数字/维表示的含义
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()
