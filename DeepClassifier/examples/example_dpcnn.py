# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function
import sys
sys.path.append("..")
import os
import argparse
import time
from time import strftime, localtime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchsummary import summary
from deepclassifier.models import TextCNN, DPCNN, HAN,RCNN
from deepclassifier.trainers import Trainer
from tensorboardX import SummaryWriter
from preprocessing import load_pretrained_embedding, texts_convert_to_ids,pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser()
parser.add_argument("--lr", dest="init_lr", type=float, metavar='<float>', default=0.001)
parser.add_argument("--lr2", dest="lr2", type=float, metavar='<float>', default=0.)
parser.add_argument("--lr_decay", dest="lr_decay", type=float, metavar='<float>', default=0.97)
parser.add_argument("--wd", dest="weights_decay", type=float, metavar='<float>', default=0)
parser.add_argument("--epochs", dest="EPOCHS", type=int, metavar='<int>', default=100)
parser.add_argument("--bs", dest="batch_size", type=int, metavar='<int>', default=128)
parser.add_argument("--eval_step", dest="eval_step", type=int, metavar='<int>', default=50)
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=200)
parser.add_argument("--gamma", dest="gamma_neg", type=float, metavar='<float>', default=4.)
parser.add_argument('--gpu', dest="GPU", type=str, default=7, help='cuda_visible_devices')
parser.add_argument("--emb", dest="freeze_emb", type=str2bool,metavar='<bool>', default=False)
parser.add_argument("--asl", dest="ASL",type=str2bool, metavar='<bool>',default=False)
parser.add_argument("--seed", dest="seeds", type=int, metavar='<int>', default=1)
parser.add_argument("--run_id", dest="id", type=int, metavar='<int>', default=0)
args = parser.parse_args()

with open('run_' + str(args.id) + '_results.txt', 'a') as f:
    f.write('\n\n==============' + __file__ + '====================\n')
    f.write(args.__str__() + '\n')

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True
torch.manual_seed(args.seeds)
torch.cuda.manual_seed(args.seeds)
torch.cuda.manual_seed_all(args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)

# 记录运行时间
since = time.time()
# 数据路径
base_path = os.getcwd()
train_data_path = base_path + "/sentiment-analysis-on-movie-reviews/train.tsv"
test_data_path = base_path + "/sentiment-analysis-on-movie-reviews/test.tsv"

# 获取数据
train_data_df = pd.read_csv(train_data_path, sep="\t")
train_data_df, dev_data_df = train_test_split(train_data_df, test_size=0.2)
test_data_df = pd.read_csv(test_data_path, sep="\t")

train_data = train_data_df.iloc[:, -2].values
train_label = train_data_df.iloc[:, -1].values
dev_data = dev_data_df.iloc[:, -2].values
dev_label = dev_data_df.iloc[:, -1].values
test_data = test_data_df.iloc[:, -1].values

# 获取词典与词向量
pretrained_embedding_file_path = base_path+"/glove/glove.6B.300d.txt"
word2idx, embedding_matrix = load_pretrained_embedding(pretrained_embedding_file_path=pretrained_embedding_file_path)
print('embedding shape:{}'.format(embedding_matrix.shape))
# 文本向量化
train_data = texts_convert_to_ids(train_data, word2idx)
dev_data = texts_convert_to_ids(dev_data, word2idx)
test_data = texts_convert_to_ids(test_data, word2idx)
print('train_data:',train_data)

train_data=torch.from_numpy(pad_sequences(train_data))
dev_data=torch.from_numpy(pad_sequences(dev_data))
test_data=torch.from_numpy(pad_sequences(test_data))
print('train_data shape:{}, dev_data_shape:{}, test_data shape:{}'.format(train_data.shape,dev_data.shape,test_data.shape ))
print('train_class:',Counter(train_label))
print('val_class:',Counter(dev_label))
# 产生batch data
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


# 权重初始化，默认xavier
def init_network(model, method='kaiming', exclude='embedding', seed=1):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

# 训练集
batch_size = args.batch_size #20
my_train_data = my_dataset(train_data, train_label)
train_loader = DataLoader(my_train_data, batch_size=batch_size, shuffle=True,drop_last=True, pin_memory=True)
# 验证集
my_dev_data = my_dataset(dev_data, dev_label)
dev_loader = DataLoader(my_dev_data, batch_size=batch_size, shuffle=False,drop_last=False)
# dev_loader = DataLoader(my_dev_data, batch_size=batch_size, shuffle=True,drop_last=True)
# 测试集
pred_data = my_dataset1(test_data)
pred_data = DataLoader(pred_data, batch_size=batch_size, shuffle=False,drop_last=False)
# pred_data = DataLoader(pred_data, batch_size=1)

# 定义模型
my_model = DPCNN(embedding_dim=embedding_matrix.shape[1], dropout_rate=0.5, num_class=5,
                   embedding_matrix=embedding_matrix, requires_grads=args.freeze_emb)
# init_network(my_model, seed=args.seeds)
print('network architecture:', my_model)
for name, param in my_model.named_parameters():
    print(name, param.requires_grad)
print('total parameters: {}'.format(sum(torch.numel(parameter) for parameter in my_model.parameters())))

# optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.Adam(parameters,lr=args.init_lr, weight_decay=args.weights_decay)
# optimizer = get_optimizer(model=my_model, lr=args.init_lr, lr2=0, weight_decay=args.weights_decay)
optimizer = optim.Adam(my_model.parameters(),lr=args.init_lr, weight_decay=args.weights_decay)
loss_fn = nn.CrossEntropyLoss()
if args.ASL:
    loss_fn = ASLSingleLabel(gamma_neg=args.gamma_neg, gamma_pos=0, eps=0.)
print('=============loss_fn:', loss_fn)

weight_path = 'weights/'
if not (os.path.exists(weight_path)):
    os.makedirs(weight_path)
save_path = weight_path +str(args.id)+"_best.ckpt"
# writer = SummaryWriter("logfie/1")
my_trainer = Trainer(model_name="dpcnn", model=my_model, train_loader=train_loader, dev_loader=dev_loader,
                     test_loader=dev_loader, optimizer=optimizer, loss_fn=loss_fn, save_path=save_path, epochs=args.EPOCHS,
                     writer=None, max_norm=0.25, eval_step_interval=args.eval_step, lr=args.init_lr, lr2=args.lr2,
                     lr_decay=args.lr_decay, weight_decay=args.weights_decay, device=device, run_id=args.id)
# my_trainer = Trainer(model_name="textcnn", model=my_model, train_loader=train_loader, dev_loader=dev_loader,
#                      test_loader=None, optimizer=optimizer, loss_fn=loss_fn, save_path=save_path, epochs=100,
#                      writer=writer, max_norm=0.25, eval_step_interval=10, device='cpu')

# 训练
my_trainer.train()

# plot loss curve
my_trainer.plot_statistic()

# 测试
p, r, f1, acc = my_trainer.test(mode='test')
print("precision:{:.4f}, recall:{:.4f}, F1:{:.4f}, acc:{:.4f}".format(p, r, f1, acc))
# 打印在验证集上最好的f1值
print(my_trainer.best_f1)

p, r, f1, acc = my_trainer.test(mode='train')


# 预测
senti_label = my_trainer.predict(pred_data)
phrase_id = np.arange(156061, 222353)
print(phrase_id.shape, senti_label.shape)

# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'PhraseId': phrase_id, 'Sentiment': senti_label})

# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("sampleSubmission_"+str(args.id)+"_.csv", index=False, sep=',')

time_elapsed = time.time() - since
print('{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

with open('run_' + str(args.id) + '_results.txt', 'a') as f:
    f.write('current_time: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()) + '\n\n')
