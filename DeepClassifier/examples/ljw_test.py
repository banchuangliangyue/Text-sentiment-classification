import sys
sys.path.append("..")
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from deepclassifier.models import TextCNN
from deepclassifier.trainers import Trainer
from preprocessing import load_pretrained_embedding, texts_convert_to_ids,pad_sequences
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import math
from itertools import repeat
import matplotlib.pyplot as plt
plt.switch_backend('agg')



torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

model_name_or_path = 'home/prev_trained_model/berta-base'
output_dir = 'ljw/outputs/'
output_dir = output_dir + '{}'.format(list(filter(None, model_name_or_path.split('/'))).pop(),)
print(output_dir)