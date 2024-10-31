import os, sys
import ast
import random
import argparse
from tqdm import tqdm


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

import torch
from torch import cuda
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

import transformers

import matplotlib.pyplot as plt 
import torch.nn.functional as F

import logging 
from transformers.utils import logging
logging.set_verbosity(40)
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
# logging.getLogger("transformers.tokenization_tapas").setLevel(logging.ERROR)


# Load and preprocess data
def load_data(filepath, rebalance, added_task=False):
    df = pd.read_csv(filepath)

    # Get <query, label> for training and rename columns
    if added_task:
        new_df = df[["Prompt", "Length", "Task"]].copy()
        new_df.rename(columns={"Query": "query", "Length": "target", "Task": "task"}, inplace=True)
    else: 
        new_df = df[["Prompt", "Length"]].copy()
        new_df.rename(columns={"Prompt": "query", "Length": "target"}, inplace=True)
    
    print("Column names:", new_df.columns.tolist())
    print("Number of rows in DataFrame:", len(new_df))
    return new_df


# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.queries = dataframe['query']
        self.targets = dataframe['target']
        self.task = None if "task" not in dataframe else dataframe["task"]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query = self.queries.iloc[index]
        tokenized_prompt = self.tokenizer(query, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > self.max_len:
            half = int(self.max_len/2)
            query = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            # if index < 32:
            #     print(repr(query))

        inputs = self.tokenizer.encode_plus(
            query,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        targets = self.targets.iloc[index] + 1
        position_ids = torch.arange(len(inputs['input_ids']))
        
        task = self.task.iloc[index] if self.task is not None else 0
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "prompt_lengths": torch.tensor(len(ids), dtype=torch.long), 
            "position_ids": position_ids, 
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "task": task, 
        }


# Define the BERT model class
class BERTClass(torch.nn.Module):
    def __init__(self, model_name, num_classes=3):
        super(BERTClass, self).__init__()
        self.model_name = model_name
        
        if 'bert' in model_name or 'long' in model_name:
            print(f"use bert")
            self.bert = transformers.AutoModel.from_pretrained(model_name)
            self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)
        elif 'llama' in model_name: 
            self.bert = transformers.AutoModel.from_pretrained(model_name)
            self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)
        else:
            print(f"use T5")
            from transformers import T5Model
            self.bert = T5Model.from_pretrained(model_name)
            self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)
        # self.tanh = torch.nn.Tanh()
        

    def forward(self, ids, mask, token_type_ids, position_ids):
        if 'llama' in self.model_name: 
            outputs = self.bert(input_ids=ids, attention_mask=mask, position_ids=position_ids)
            last_hidden_state = outputs.last_hidden_state[:, -1, :] 
            regression_output = torch.exp(self.regressor(last_hidden_state))
            return regression_output, 0
        else: 
            # print(self.bert.encoder.layer[-1].output.dense.bias.mean())
            # print("")
            # print(self.regressor.weight.mean())
            # print(self.bert.encoder.layer[-1].output.dense.bias.min())
            if self.bert.encoder.layer[-1].output.dense.bias.grad is not None: 
                pass 
                # print("grad encoder", self.bert.encoder.layer[-1].output.dense.bias.grad.mean())
                # print("grad regressor", self.regressor.bias.grad.mean())
            
            outputs = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            # import pdb; pdb.set_trace() 
            last_hidden_state = outputs.last_hidden_state.mean(1)
            # last_hidden_state = outputs.last_hidden_state[:, -1, :] 
            regression_output = torch.exp(self.regressor(last_hidden_state))
            return regression_output, 0


def rmse(outputs, targets): 
    # import pdb; pdb.set_trace() 
    # return ((outputs - targets) / (targets + 1e-3)).abs().mean()
    return abs((outputs - targets) / (targets + 1e-3))


# Validation function
def validate_and_get_class_accuracy(model, test_loader, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    fin_tasks = []
    loss = 0 
    cnt = 0
    
    predict_infos = list() 
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            position_ids = data["position_ids"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            # targets = data["targets"].to(device)
            targets = data["targets"].to(device) 
            fin_tasks.extend(data['task'])

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids, position_ids=position_ids)[0].squeeze(-1)
            # import pdb; pdb.set_trace() 
            pred_lengths = outputs * data["prompt_lengths"].to(device)
            predict_infos.append((pred_lengths.item(), targets.item()))
            # import pdb; pdb.set_trace() 
            
    return predict_infos


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_dataloder(data_path, tokenizer, max_len): 
    df = load_data(data_path, False)
    # Split data
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    test_set = CustomDataset(test_dataset, tokenizer, max_len)
    num_workers = 8
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    return test_loader

def load_model_state(model, model_ident): 
    dirname = os.path.join('1-bert-length/', model_ident)
    ckpt_path = None 
    for path in os.listdir(dirname): 
        if path.endswith('pt'): 
            ckpt_path = os.path.join(dirname, path)
            break 
    # import pdb; pdb.set_trace() 
    model_state = torch.load(ckpt_path)['model_state_dict']
    model.load_state_dict(model_state)
    
            
# Main execution
def main():
    max_len = 2048 

    seed_everything(42)
    alg2name = {
        'hf': 'model_Meta-Llama-3-8B-instruct_8192_alg_hf_allenai_longformer-base-4096_dropout_False', 
        'kivi': 'model_Meta-Llama-3-8B-instruct_8192_alg_kivi_allenai_longformer-base-4096_dropout_False', 
        'gear': 'model_Meta-Llama-3-8B-instruct_8192_alg_gear_allenai_longformer-base-4096_dropout_False', 
        'h2o': 'model_Meta-Llama-3-8B-instruct_8192_alg_h2o_allenai_longformer-base-4096_dropout_False', 
        'streamingllm': 'model_Meta-Llama-3-8B-instruct_8192_alg_streamingllm_allenai_longformer-base-4096_dropout_False'
    }
    bert_model_name = 'allenai/longformer-base-4096'
    # Tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_model_name)
    hf_model = BERTClass(bert_model_name).to('cuda')
    load_model_state(hf_model, alg2name['hf'])
    model_ident="Meta-Llama-3-8B-instruct_8192"
    hf_loader = load_dataloder(data_path=f"0_data/leninfo/model_{model_ident}_alg_hf.csv", 
                    tokenizer=tokenizer, max_len=max_len)
    hf_pred_info = validate_and_get_class_accuracy(hf_model, hf_loader, device='cuda')
    for alg in ['hf', 'kivi', 'gear', 'h2o', 'streamingllm']: 
    # for alg in ['hf']: 
        alg_loader = load_dataloder(
            data_path=f"0_data/leninfo/model_{model_ident}_alg_{alg}.csv", 
            tokenizer=tokenizer, max_len=max_len)
        alg_model = BERTClass(bert_model_name).to('cuda')
        load_model_state(alg_model, alg2name[alg])
        alg_pred_info = validate_and_get_class_accuracy(alg_model, alg_loader, device='cuda')
        ratio_list = list() 
        for idx, (hf_pred, alg_pred) in enumerate(zip(hf_pred_info, alg_pred_info)): 
            # 1 is target, 0 is predict 
            ratio_list.append(rmse(alg_pred[0], alg_pred[1]))
            
        print("-" * 10)
        print(alg, np.mean(ratio_list))
        # import pdb; pdb.set_trace() 



if __name__ == "__main__":
    main()