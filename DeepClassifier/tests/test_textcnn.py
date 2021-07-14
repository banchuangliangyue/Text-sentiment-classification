# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from deepclassifier.models import TextCNN
from deepclassifier.trainers import Trainer
from tensorboardX import SummaryWriter


class my_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]
        item_label = self.label[item]

        return item_data, item_label

class my_dataset1(Dataset):
    def __init__(self, data):
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]


        return item_data


# 训练集
batch_size = 20
train_data = np.random.randint(0, 100, (100, 60))
train_label = torch.from_numpy(np.array([int(x > 0.5) for x in np.random.randn(100)]))
my_train_data = my_dataset(train_data, train_label)
train_loader = DataLoader(my_train_data, batch_size=batch_size, shuffle=True)

# 验证集
dev_data = np.random.randint(0, 100, (100, 60))
dev_label = torch.from_numpy(np.array([int(x > 0.5) for x in np.random.randn(100)]))
my_dev_data = my_dataset(dev_data, dev_label)
dev_loader = DataLoader(my_dev_data, batch_size=batch_size, shuffle=True)

# 测试集
test_data = np.random.randint(0, 100, (100, 60))
test_label = torch.from_numpy(np.array([int(x > 0.5) for x in np.random.randn(100)]))
my_test_data = my_dataset(test_data, dev_label)
test_loader = DataLoader(my_test_data, batch_size=batch_size, shuffle=True)

my_model = TextCNN(5, 0.2, 2, 100, 60)
optimizer = optim.Adam(my_model.parameters())
loss_fn = nn.CrossEntropyLoss()
save_path = "best.ckpt"

writer = SummaryWriter("logfie/1")
my_trainer = Trainer(model_name="textcnn", model=my_model, train_loader=train_loader, dev_loader=dev_loader,
                     test_loader=test_loader, optimizer=optimizer, loss_fn=loss_fn,
                     save_path=save_path, epochs=100, writer=writer, max_norm=0.25, eval_step_interval=10, device='cpu')

print(my_trainer.device)
# 训练
my_trainer.train()
# 测试
p, r, f1 = my_trainer.test()
print(p, r, f1)
# 打印在验证集上最好的f1值
print(my_trainer.best_f1)

# 预测
pred_data = np.random.randint(0, 100, (100, 60))
pred_data=my_dataset1(pred_data)
pred_data=DataLoader(pred_data,batch_size=1)
prd_label=my_trainer.predict(pred_data)
print(prd_label.shape)
