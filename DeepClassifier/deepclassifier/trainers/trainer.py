# -*- coding:utf-8 -*-
'''
Author:
    Zichao Li,2843656167@qq.com
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Trainer(object):
    def __init__(self,
                 model_name,
                 model,
                 train_loader,
                 dev_loader,
                 test_loader,
                 optimizer,
                 loss_fn,
                 save_path,
                 epochs,
                 writer,
                 max_norm,
                 eval_step_interval,
                 lr,
                 lr2,
                 lr_decay,
                 weight_decay,
                 device="cpu",
                 run_id=None):
        super(Trainer, self).__init__()

        self.model_name = model_name.lower()
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.epochs = epochs
        self.writer = writer
        self.max_norm = max_norm
        self.step_interval = eval_step_interval
        self.lr = lr
        self.lr2 = lr2
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.run_id = str(run_id)
        # self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
        #                                             num_warmup_steps=5,
        #                                             num_training_steps=self.epochs )
        self.model.to(self.device)


    def train(self):
        self.model.train()
        self.best_f1 = 0.0
        global_steps = 1
        all_res = []


        for epoch in range(1, self.epochs + 1):
            if self.lr_decay:
                lr_t = self.lr / (1 + 10 * epoch / self.epochs) ** 0.75
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr_t, weight_decay=self.weight_decay)
            train_loss = []
            train_acc = []
            for idx, batch_data in enumerate(self.train_loader, start=1):

                if self.model_name in ["textcnn", "rcnn", "han", "dpcnn", 'transformer']:
                    input_ids, y_true = batch_data[0], batch_data[-1]
                    if y_true.shape != 1:
                        y_true = y_true.squeeze(dim=-1)
                    input_ids = input_ids.to(self.device)
                    logits = self.model(input_ids)
                    y_true = y_true.to(self.device)
                elif self.model_name in ["berttextcnn", "bertrcnn", "berthan", "bertdpcnn"]:
                    if len(batch_data) == 3:
                        input_ids, attention_mask, y_true = batch_data[0], batch_data[1], batch_data[-1]
                        if y_true.shape != 1:
                            y_true = y_true.squeeze(dim=-1)
                        logits = self.model(input_ids.to(self.device), attention_mask.to(self.device))
                        y_true = y_true.to(self.device)
                    else:
                        input_ids, y_true = batch_data[0], batch_data[-1]
                        if y_true.shape != 1:
                            y_true = y_true.squeeze(dim=-1)
                        logits = self.model(input_ids.to(self.device))
                        y_true = y_true.to(self.device)
                else:
                    raise ValueError("the number of batch_data is wrong!")


                loss = self.loss_fn(logits, y_true)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_true).float().mean().detach().cpu().numpy()

                # print("epoch:{} step:{} train_loss:{:.4f} train_acc:{:.4f}".format(epoch, idx, train_loss.item(),
                #                                                                  train_acc))

                self.optimizer.zero_grad()
                loss.backward()
                ##裁剪梯度
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

                self.optimizer.step()
                train_loss.append(loss.item())
                train_acc.append(acc)


            train_loss, train_acc = np.mean(train_loss), np.mean(train_acc)
            val_loss, p, r, f1, val_acc = self.eval()
            current_lr = self.optimizer.param_groups[0]['lr']
            # current_embedding_lr = self.optimizer.param_groups[1]['lr']

            wd = self.optimizer.param_groups[0]['weight_decay']


            print("\n====> epoch:{}/{}  lr:{:.6f}  emb_lr:{:.6f}  weight_decay:{:.5f}  F1:{:.4f}  train_loss:{:.4f}"
                "  val_loss:{:.4f}  train_acc:{:.4f}  val_acc：{:.4f}\n".format(epoch, self.epochs, current_lr,current_lr,
                                        wd, f1, train_loss, val_loss, train_acc, val_acc))
            if self.best_f1 <= f1:
                self.best_f1 = f1
                # save_path = 'weights/' + self.run_id + '_' + str(self.best_f1) + "_best.ckpt"
                torch.save({'model_state_dict': self.model.state_dict(),
                            'epoch': epoch,
                            'best val f1': self.best_f1},
                             self.save_path)



            print("========epoch:{}, best_f1:{:.4f}========".format(epoch, self.best_f1))

            all_res.append([epoch, train_loss, val_loss, train_acc, val_acc, f1, current_lr])
            df = pd.DataFrame(all_res, columns=['epoch', 'train_loss', 'val_loss', 'train_acc',
                                                'val_acc', 'val_f1', 'lr'])
            df.to_csv('loss_acc_plot_' + self.run_id + '.csv', index=False, float_format='%.5f')

            self.model.train()



    def eval(self):
        self.model.eval()
        y_preds = []
        y_trues = []
        val_loss = []

        with torch.no_grad():
            for idx, batch_data in enumerate(self.dev_loader, start=1):

                if self.model_name in ["textcnn", "rcnn", "han", "dpcnn","transformer"]:
                    input_ids, y_true = batch_data[0], batch_data[-1]
                    if y_true.shape !=1:
                        y_true=y_true.squeeze(dim=-1)
                    logits = self.model(input_ids.to(self.device))
                    y_true = y_true.to(self.device)
                elif self.model_name in ["berttextcnn", "bertrcnn", "berthan", "bertdpcnn"]:
                    if len(batch_data) == 3:
                        input_ids, attention_mask, y_true = batch_data[0], batch_data[1], batch_data[-1]
                        if y_true.shape !=1:
                            y_true=y_true.squeeze(dim=-1)
                        logits = self.model(input_ids.to(self.device), attention_mask.to(self.device))
                        y_true = y_true.to(self.device)
                    else:
                        input_ids, y_true = batch_data[0], batch_data[-1]
                        if y_true.shape !=1:
                            y_true=y_true.squeeze(dim=-1)
                        logits = self.model(input_ids.to(self.device))
                        y_true = y_true.to(self.device)
                else:
                    raise ValueError("the number of batch_data is wrong!")

                loss = self.loss_fn(logits, y_true).item()
                val_loss.append(loss)
                y_true = list(y_true.cpu().numpy())
                y_trues.extend(y_true)

                logits = logits.cpu().numpy()
                for item in logits:
                    pred = np.argmax(item)
                    y_preds.append(pred)

        val_loss = np.mean(val_loss)
        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)

        p = precision_score(y_trues, y_preds, average="macro")
        r = recall_score(y_trues, y_preds, average="macro")
        # f1 = f1_score(y_trues, y_preds, average="weighted")
        f1 = f1_score(y_trues, y_preds, average="macro")
        acc = accuracy_score(y_trues, y_preds)

        return val_loss, p, r, f1, acc

    def test(self, mode='test'):
        print('Load model...')
        checkpoint = torch.load(self.save_path)
        print('best epoch:', checkpoint['epoch'])
        print('best val f1:', checkpoint['best val f1'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        y_preds = []
        y_trues = []

        with torch.no_grad():
            if mode=='test':
                dataLoader = self.test_loader
            else:
                dataLoader = self.train_loader

            for idx, batch_data in enumerate(dataLoader, start=1):

                if self.model_name in ["textcnn", "rcnn", "han", "dpcnn","transformer"]:
                    input_ids, y_true = batch_data[0], batch_data[-1]
                    if y_true.shape !=1:
                        y_true=y_true.squeeze(dim=-1)
                    logits = self.model(input_ids.to(self.device))
                elif self.model_name in ["berttextcnn", "bertrcnn", "berthan", "bertdpcnn"]:
                    if len(batch_data) == 3:
                        input_ids, attention_mask, y_true = batch_data[0], batch_data[1], batch_data[-1]
                        if y_true.shape !=1:
                            y_true=y_true.squeeze(dim=-1)
                        logits = self.model(input_ids.to(self.device), attention_mask.to(self.device))
                    else:
                        input_ids, y_true = batch_data[0], batch_data[-1]
                        if y_true.shape !=1:
                            y_true=y_true.squeeze(dim=-1)
                        logits = self.model(input_ids.to(self.device))
                else:
                    raise ValueError("the number of batch_data is wrong!")

                y_true = list(y_true.cpu().numpy())
                y_trues.extend(y_true)

                logits = logits.cpu().numpy()
                for item in logits:
                    pred = np.argmax(item)
                    y_preds.append(pred)

        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)

        p = precision_score(y_trues, y_preds, average="macro")
        r = recall_score(y_trues, y_preds, average="macro")
        # f1 = f1_score(y_trues, y_preds, average="weighted")
        f1 = f1_score(y_trues, y_preds, average="macro")
        acc = accuracy_score(y_trues, y_preds)

        with open('run_' + self.run_id + '_results.txt', 'a') as f:
            f.write('best epoch:' + str(checkpoint['epoch']) + '  ' + 'best val f1:' + str(
                checkpoint['best val f1']) + '\n')
            f.write(classification_report(y_trues, y_preds, target_names=['0', '1', '2', '3', '4'], digits=4) + '\n')
            f.write(str(confusion_matrix(y_trues, y_preds)) + '\n')

        return p, r, f1, acc

    def predict(self, x):
        self.model.eval()
        y_preds = []
        with torch.no_grad():
            for idx, batch_data in enumerate(x, start=1):
                if self.model_name in ["textcnn", "rcnn", "han", "dpcnn", "transformer"]:
                    input_ids = batch_data
                    logits = self.model(input_ids.to(self.device))
                elif self.model_name in ["berttextcnn", "bertrcnn", "berthan", "bertdpcnn"]:
                    if len(batch_data) == 2:
                        input_ids, attention_mask = batch_data[0], batch_data[1]
                        logits = self.model(input_ids.to(self.device), attention_mask.to(self.device))
                    else:
                        input_ids=batch_data
                        logits = self.model(input_ids.to(self.device))
                else:
                    raise ValueError("the number of batch_data is wrong!")

                logits = logits.cpu()
                prob = torch.argmax(logits, dim=-1)
                # prob = F.softmax(logits, dim=-1)
                y_preds.extend(prob)

        y_preds = torch.stack(y_preds, dim=0).numpy()

        return y_preds


    def plot_statistic(self):
        ##df = pd.DataFrame(all_res, columns=['epoch', 'step', 'train_loss', 'val_loss', 'train_acc',
        # 'val_acc', 'val_f1', 'lr'])
        df = pd.read_csv('loss_acc_plot_' + self.run_id + '.csv')
        plt.figure(figsize=(30, 10))
        plt.subplot(131)
        plt.plot(df.epoch.values.tolist(), df.train_loss.values.tolist(), color='red', label='train_loss')
        plt.plot(df.epoch.values.tolist(), df.val_loss.values.tolist(), color='blue', label='val_loss')
        plt.xlabel('step')
        plt.legend(loc="best")
        plt.grid(True)

        plt.subplot(132)
        plt.plot(df.epoch.values.tolist(), df.train_acc.values.tolist(), color='red',label='train_acc')
        plt.plot(df.epoch.values.tolist(), df.val_acc.values.tolist(), color='blue',label='val_acc')
        plt.xlabel('step')
        plt.legend(loc="best")
        plt.grid(True)

        plt.subplot(133)
        plt.plot(df.epoch.values.tolist(), df.val_f1.values.tolist(), color='red', label='val_f1')
        plt.xlabel('step')
        plt.legend(loc="best")
        plt.grid(True)

        plt.savefig('loss_acc_plot_' + self.run_id + '.png', bbox_inches='tight')


    def exp_moving_average(self, loss, beta=0.6):
        avg_loss = 0
        step_num = 0
        moving_loss = []
        for l in loss:
            step_num += 1
            avg_loss = beta * avg_loss + (1 - beta) * l
            smoothed_loss = avg_loss / (1 - beta ** step_num)
            moving_loss.append(smoothed_loss)

        return moving_loss

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        '''https://github.com/chenyuntc/PyTorchText/blob/master/models/BasicModule.py'''
        ignored_params = list(map(id, self.model.embedding.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad,
                             self.model.parameters())
        if lr2 is None:
            lr2 = lr1 * 0.5

        # optimizer = optim.Adam([
        #     dict(params=base_params, weight_decay=weight_decay, lr=lr1),
        #     {'params': self.model.embedding.parameters(), 'lr': lr2}
        # ])
        optimizer = optim.Adam(self.model.parameters(),lr=lr1, weight_decay=weight_decay)
        return optimizer



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

