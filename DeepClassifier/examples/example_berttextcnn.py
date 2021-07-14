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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,RandomSampler, TensorDataset
from torch.backends import cudnn
import torch.optim as optim
from torchsummary import summary
import torch.optim as optim
from deepclassifier.models import *
from deepclassifier.trainers import Trainer
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
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
parser.add_argument("--lr_decay", dest="lr_decay", type=str2bool, metavar='<bool>',default=True)
parser.add_argument("--wd", dest="weights_decay", type=float, metavar='<float>', default=0)
parser.add_argument("--epochs", dest="EPOCHS", type=int, metavar='<int>', default=100)
parser.add_argument("--bs", dest="batch_size", type=int, metavar='<int>', default=128)
parser.add_argument("--eval_step", dest="eval_step", type=int, metavar='<int>', default=50)
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=200)
parser.add_argument('--gpu', dest="GPU", type=str, default=7, help='cuda_visible_devices')
parser.add_argument("--gamma", dest="gamma_neg", type=float, metavar='<float>', default=4.)
parser.add_argument("--nb_encoder", dest="nb_encoder", type=int, metavar='<int>', default=3)
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
bert_path = base_path + "/bert-base-uncased-ljw"

# 获取数据
train_data_df = pd.read_csv(train_data_path, sep="\t")
train_data_df, dev_data_df = train_test_split(train_data_df, test_size=0.2)
test_data_df = pd.read_csv(test_data_path, sep="\t")

train_data = train_data_df.iloc[:, -2].values
train_label = train_data_df.iloc[:, -1].values
dev_data = dev_data_df.iloc[:, -2].values
dev_label = dev_data_df.iloc[:, -1].values
test_data = test_data_df.iloc[:, -1].values



def convert_examples_to_features(texts, labels, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s.
    Args:
        examples: 表示样本集
        label_list: 标签列表
        max_seq_length: 句子最大长度
        tokenizer： 分词器
    Returns:
        features: InputFeatures, 表示样本转化后信息
    """


    features = []
    for (tx_index, text) in enumerate(texts):
        # if ex_index % 10000 == 0:
        # print("Writing example {} of {}".format(tx_index, len(texts)))

        tokens_a = tokenizer.tokenize(text)  # 分词

        # "- 2" 是因为句子中有[CLS], [SEP] 两个标识，可参见论文
        # [CLS] the dog is hairy . [SEP]
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # [CLS] 可以视作是保存句子全局向量信息
        # [SEP] 用于区分句子，使得模型能够更好的把握句子信息
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print(input_ids)
        #input_mask: 1 表示真正的 tokens， 0 表示是 padding tokens
        #Mask to avoid performing attention on padding token indices
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        if labels is None:
            label_id = tx_index
        else:
            label_id = labels[tx_index]
        features.append([input_ids, input_mask, label_id])

    return features


def convert_features_to_tensors(features, batch_size, data_type):
    """ 将 features 转化为 tensor，并塞入迭代器
    Args:
        features: 样本 features 信息
        batch_size: batch 大小
    Returns:
        dataloader: 以 batch_size 为基础的迭代器
    """

    all_input_ids = torch.tensor(
        [f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f[2] for f in features], dtype=torch.long)


    if data_type == "test":
        data = TensorDataset(all_input_ids, all_input_mask)
        # sampler = RandomSampler(data)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        # sampler = RandomSampler(data)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)

    return dataloader




# 产生batch data
class my_dataset(Dataset):
    def __init__(self, data, label, max_length, tokenizer=None):
        self.data = data
        self.label = label
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]
        item_label = [self.label[item]]

        item_data = item_data.strip().split()
        c = ["[CLS]"] + item_data + ["SEP"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        if len(input_ids) >= self.max_length:
            input_ids = input_ids[:self.max_length]
        attention_mask = [1.0] * len(input_ids)
        extra = self.max_length - len(input_ids)
        if extra > 0:
            input_ids += [0] * extra
            attention_mask += [0.0] * extra

        return torch.LongTensor(input_ids), torch.FloatTensor(attention_mask), torch.LongTensor(item_label)


class my_dataset1(Dataset):
    def __init__(self, data, max_length, tokenizer=None):
        self.data = data

        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]

        item_data = item_data.strip().split()
        c = ["[CLS]"] + item_data + ["SEP"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        if len(input_ids) >= self.max_length:
            input_ids = input_ids[:self.max_length]
        attention_mask = [1.0] * len(input_ids)
        extra = self.max_length - len(input_ids)
        if extra > 0:
            input_ids += [0] * extra
            attention_mask += [0.0] * extra

        return torch.LongTensor(input_ids), torch.FloatTensor(attention_mask)

##共28996词，包括特殊符号:('[UNK]', 100),('[PAD]', 0),('[CLS]', 101),('[SEP]', 102), ('[MASK]', 103)..
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-cased")
# inputs = tokenizer("Hello world!", return_tensors="pt")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer(vocab_file=bert_path+"/vocab.txt")
# 训练集
batch_size = args.batch_size #20
train_features = convert_examples_to_features(train_data, train_label, 50, tokenizer)
train_loader = convert_features_to_tensors(train_features, batch_size, 'train')
# my_train_data = my_dataset(train_data, train_label, 200, tokenizer)
# train_loader = DataLoader(my_train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
# 验证集
# my_dev_data = my_dataset(dev_data, dev_label, 200, tokenizer)
# dev_loader = DataLoader(my_dev_data, batch_size=batch_size, shuffle=False, drop_last=False)
dev_features = convert_examples_to_features(dev_data, dev_label, 50, tokenizer)
dev_loader = convert_features_to_tensors(dev_features, batch_size, 'dev')
# 测试集
# pred_data = my_dataset1(test_data, 200, tokenizer)
# pred_data = DataLoader(pred_data, batch_size=1)
pred_features = convert_examples_to_features(test_data, None, 50, tokenizer)
pred_data = convert_features_to_tensors(pred_features, batch_size, 'test')

# 定义模型
my_model = BertTextCNN(embedding_dim=768, dropout_rate=0.2, num_class=5, bert_layers=args.nb_encoder, bert_path=bert_path)
print('network architecture:', my_model)
# for name, param in my_model.named_parameters():
#     if name.startswith('bert'):
#         param.requires_grad = False

for name, param in my_model.named_parameters():
    print(name, param.requires_grad)

print('total parameters: {}'.format(sum(torch.numel(p) for p in my_model.parameters() if p.requires_grad)))

optimizer = optim.Adam(my_model.parameters(),lr=args.init_lr, weight_decay=args.weights_decay)
loss_fn = nn.CrossEntropyLoss()
print('=============loss_fn:', loss_fn)

weight_path = 'weights/'
if not (os.path.exists(weight_path)):
    os.makedirs(weight_path)
save_path = weight_path +str(args.id)+"_best.ckpt"

my_trainer = Trainer(model_name="berttextcnn", model=my_model, train_loader=train_loader, dev_loader=dev_loader,
                     test_loader=dev_loader, optimizer=optimizer, loss_fn=loss_fn, save_path=save_path, epochs=args.EPOCHS,
                     writer=None, max_norm=0.25, eval_step_interval=10, lr=args.init_lr, lr_decay=args.lr_decay,
                     lr2=args.lr2, weight_decay=args.weights_decay, device=device, run_id=args.id)

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
