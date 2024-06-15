from torch.utils.data import Dataset
from PIL import Image
import os

class mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img = os.listdir(self.path)    # 存储所有图片路径列表

    '''
    数据集类需要重写__getitem__，__len__，这样才能使用索引返回对应的数据与标签
    '''
    def __getitem__(self, idx):
        img_name = self.img[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img)


root_dir = "hymenoptera_data/train"
ant_label_dir = "ants"
bees_label_dir = "bees"
ant_dataset = mydata(root_dir, ant_label_dir)
bees_dataset = mydata(root_dir, bees_label_dir)

train_dataset = ant_dataset + bees_dataset      # 应该是Dataset重载了+运算符，两个数据集可以组合成一个大的数据集
