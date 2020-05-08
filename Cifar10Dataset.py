import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class Cifar10Dataset(Dataset):
    '''
    用于从文件读取Cifar数据集
    '''
    def __init__(self, root_path, train=True, transform=None):
        '''
        初始化
        :param root_path: Cifar数据集的存放目录
        :param train: 是否读取训练数据
        '''
        super(Cifar10Dataset, self).__init__()
        self.transform = transform
        self.datas = None
        self.labels = []
        # 训练文件名
        train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        # 测试文件名
        test_batches = ['test_batch']
        # 根据train标志选择文件
        if train:
            batches = train_batches
        else:
            batches = test_batches
        for file_name in batches:
            with open(root_path + file_name, 'rb') as file:
                dict = pickle.load(file, encoding='latin1')
                datas = dict['data'].reshape(-1, 3, 32, 32)
                labels = dict['labels']
                if self.datas is None:
                    self.datas = datas
                else:
                    self.datas = np.append(self.datas, datas, axis=0)
                self.labels += labels
        if self.transform is not None:
            self.datas = self.datas.transpose((0, 2, 3, 1)) 
        else:
            self.datas = torch.from_numpy(self.datas).type(torch.FloatTensor)   # 变为tensor张量
        
        


    def __getitem__(self, index):
        img = self.datas[index]
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.datas)


