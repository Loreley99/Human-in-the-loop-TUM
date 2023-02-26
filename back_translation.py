import torch
!pip install fairseq
!pip install sacremoses
!pip install datasets
!pip install fastBPE
import fairseq
import pickle
import os
import json
from tqdm.notebook import trange, tqdm
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from pandas import DataFrame

#read the data
dataset = load_dataset('trec')
train_data = dataset['train']

#load model for translation
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')


"""
Parameters:
  start - the start index of the data will be sampled
  end - the end index
  file_name - the name of file which will save the augmented data

Returns:
	Augmented sentence will be written in the file
"""
def translate_de(start, end, file_name, temperature=0.9):
    trans_a = {}
    for idx in tqdm(range(start, end)):
        trans_a[train_idxs[idx]] = de2en.translate(en2de.translate(train_text[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        if idx % 200 == 0:
            with open("raoyuan_"+file_name, 'wb') as f:
                pickle.dump((trans_a), f)
    with open("raoyuan_"+file_name, 'wb') as f:
        pickle.dump((trans_a), f)

def translate_ru(start, end, file_name, temperature=0.9):
    trans_a = {}
    for idx in tqdm(range(start, end)):
        trans_a[train_idxs[idx]] = ru2en.translate(en2ru.translate(train_text[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        if idx % 200 == 0:
            with open("raoyuan_"+file_name, 'wb') as f:
                pickle.dump((trans_a), f)

    with open("raoyuan_"+file_name, 'wb') as f:
        pickle.dump((trans_a), f)

#augment over the whole dataset
train_labels = [v['coarse_label'] for v in train_data]
train_text = [v['text'] for v in train_data]
data_idxs = [str(i) for i,v in enumerate(train_data)]
train_idxs = data_idxs
print('train data length:', len(data_idxs))
translate_de(0,len(train_idxs),'multinli_all_de.pkl')
translate_ru(0,len(train_idxs),'multinli_all_ru.pkl')

#read the sentences just generated
with open('/content/drive/MyDrive/raoyuan_multinli_all_de.pkl', 'rb') as pickle_file:
  content_de_all = pickle.load(pickle_file)

with open('raoyuan_multinli_all_ru.pkl', 'rb') as pickle_file:
  content_ru_all = pickle.load(pickle_file)

#append the corresponding label to the augmented data
de_text_list_all = []
ru_text_list_all = []

for value in content_ru_all.values():
  ru_text_list_all.append(value)

for value in content_de_all.values():
  de_text_list_all.append(value)

ru_text_all = DataFrame(ru_text_list_all)
de_text_all = DataFrame(de_text_list_all)

ru_text_all.reset_index(drop=True, inplace=True)
de_text_all.reset_index(drop=True, inplace=True)
train_data_df = DataFrame(train_data)
train_data_df.reset_index(drop=True, inplace=True)

ru_aug_all = pd.concat([ru_text_all,train_data_df['coarse_label']],axis = 1,ignore_index=True)
ru_aug_all.columns = ['text','coarse_label']

de_aug_all = pd.concat([de_text_all,train_data_df['coarse_label']],axis = 1,ignore_index=True)
de_aug_all.columns = ['text','coarse_label']
de_aug_all.to_pickle('de_aug_all.pkl')
ru_aug_all.to_pickle('ru_aug_all.pkl')