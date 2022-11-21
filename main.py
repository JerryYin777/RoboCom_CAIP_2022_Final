# import subprocess 
# cmd1 = '/home/jovyan/.virtualenvs/basenv/bin/pip uninstall transformers'
# cmd2 = '/home/jovyan/.virtualenvs/basenv/bin/pip install transformers-4.24.0-py3-none-any.whl'
# subprocess.call(cmd1, shell=True)
# subprocess.call(cmd2, shell=True)

import numpy as np
import pandas as pd
import time
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import Adam, SGD, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, DataCollatorWithPadding
from transformers import BertTokenizer,AutoModel,AdamW,AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
# from transformers import DebertaV2ForSequenceClassification, DebertaV2Config, DebertaV2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import copy
import torch.nn as nn
import os
import json
import gc
import random
from collections import Counter
from torch.cuda.amp import GradScaler
from math import ceil
from collections import Counter

scaler = GradScaler()

class CFG:
    input_path = './'
    model_path = 'TCbert_pretrain' #  nghuyong/ernie-2.0-large-en studio-ousia/luke-large
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0
    max_input_length = 400
    epochs = 30  # 5
    encoder_lr = 20e-6
    decoder_lr = 20e-6
    min_lr = 0.5e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0
    num_fold = 5
    batch_size = 1
    seed = 1006
    OUTPUT_DIR = './'
    num_workers = 0
    device='cpu'
    print_freq = 1

    
class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in
                                   output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in
                                   output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["label"] = torch.tensor(output["target"], dtype=torch.long)

        return output

def rmSpecialCharacter(string):
    string = string.replace('\u3000', '').replace('\xa0', '').replace('\u00A0', '').replace('\u0020', '').replace('\n', '').replace('\t', '').replace('\r', '')
    res = ''
    for uchar in string:
        if (uchar >= u'\u4e00' and uchar <= u'\u9fa5') or uchar in ['，', ',', '.', '。', '?', '!', '？', '！', ';', '；'] or uchar.isalnum(): #是中文字符
             if uchar != ' ':
                    res += uchar
    return res


class TestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.sentence = df['sentence'].values
        # self.label = df['label'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        sentence = self.sentence[item]
        input_text = sentence
        inputs = self.tokenizer(input_text, truncation=True, max_length=300, padding='max_length')
        return torch.as_tensor(inputs['input_ids'], dtype=torch.long), \
               torch.as_tensor(inputs['attention_mask'], dtype=torch.long)

def infer(test_loader, model, device):
    model.to(device)
    model.eval()
    preds = []
    probs = []
    for step, batch in tqdm(enumerate(test_loader)):
        mask = batch[1].to(device)
        input_ids = batch[0].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask)
        logits = F.softmax(output.logits, dim=-1)
        prob, y_preds = logits.max(dim=-1)
        probs.append(prob.to('cpu').numpy())
        preds.append(y_preds.to('cpu').numpy())

    predictions = np.concatenate(preds)
    probs = np.concatenate(probs)
    return predictions, probs


##########################################################################################
# macbert
# config = AutoConfig.from_pretrained('./pretrain/hfl/macbert-base/')
# config.num_labels = 5
# config.output_hidden_states = True
# tokenizer = AutoTokenizer.from_pretrained('./pretrain/hfl/macbert-base/')
# model = AutoModelForSequenceClassification.from_pretrained('./pretrain/hfl/macbert-base/',config = config)
# model.load_state_dict(torch.load('./macbert-base-300_epoch3.pth', map_location='cpu')['model'])
##########################################################################################

##########################################################################################
# 20k lert base 300
# 财经 17
# 科技 12
# 社会 20
# 体育 20
# 娱乐 20
# epoch 3 : 87
# epoch 4:  87
# epoch 5 : 87
config = AutoConfig.from_pretrained('./pretrain/hfl/lert-base/')
config.num_labels = 5
config.output_hidden_states = True
tokenizer = AutoTokenizer.from_pretrained('./pretrain/hfl/lert-base/')
collate_fn = Collate(tokenizer, isTrain=True)

model1 = AutoModelForSequenceClassification.from_pretrained('./pretrain/hfl/lert-base/',config = config)
model1.load_state_dict(torch.load('./lert-base-300-A89.pth', map_location='cpu')['model'])

model2 = AutoModelForSequenceClassification.from_pretrained('./pretrain/hfl/lert-base/',config = config)
model2.load_state_dict(torch.load('./lert-base_epoch5.pth', map_location='cpu')['model'])
##########################################################################################




# ---------------------- 模型预测，注意不要修改函数的输入输出 -------------------------
def predict(text):
    """
    :param text: 中文字符串，新闻片段
    :return: 字符串格式的类型，如'科技'
    """
#     text = text.split(' ')[0]
    text = rmSpecialCharacter(text)
    text_1 = text[:30]
    text_2 = text[:300]
    
    va_data_1 = pd.DataFrame({
        'sentence': [text_1],
    })
    
    va_data_2 = pd.DataFrame({
        'sentence': [text_2],
    })
    
    test_dataset_1 = TestDataset(va_data_1, tokenizer)
    test_dataset_2 = TestDataset(va_data_2, tokenizer)
    
    test_dataloader_1 = DataLoader(test_dataset_1,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=False)
    test_dataloader_2 = DataLoader(test_dataset_2,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=False)
    
    result_1 = infer(test_dataloader_1, model1, CFG.device)
    result_2 = infer(test_dataloader_2, model2, CFG.device)
    
    # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    mapping = {0: '财经', 1: '科技', 2: '社会', 3: '体育', 4: '娱乐'}
    
    prediction1 = mapping[result_1[0][0]]
    prediction2 = mapping[result_2[0][0]]
    prediction = prediction2
    
    if prediction1 == '财经' and prediction2 == '科技':
        prediction = prediction1
    
    return prediction
