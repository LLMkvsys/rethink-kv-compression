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

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for text classification")
    parser.add_argument('--max_len', type=int, default=1500, help='Maximum length of input sequences')
    parser.add_argument("--data-path", type=str, default=None, required=True, help="the filename to save length information")
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-05, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
    parser.add_argument('--model_name', type=str, default='allenai/longformer-base-4096', help='Name of the model to use')
    parser.add_argument('--ga_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--added_task', default=False, type=ast.literal_eval, help='Whether we have add the dataset info')
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--save_check_pt', default=False, action='store_true', help='Whether to save checkpoints during training')
    
    parser.add_argument('--alg', type=str, default='hf', help='Name of the trained algorithm')
    parser.add_argument('--llm', type=str, default='Meta-Llama-3-8B-instruct_8192', help='Name of the trained llm')
    parser.add_argument('--fixed_dropout', type=ast.literal_eval, default=False, help='whether to fix drop layers')

    return parser.parse_args()


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
            outputs = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            last_hidden_state = outputs.last_hidden_state.mean(1)
            # last_hidden_state = outputs.last_hidden_state[:, -1, :] 
            regression_output = torch.exp(self.regressor(last_hidden_state))
            return regression_output, 0


# Define loss function
def loss_fn(outputs, targets):
    return (torch.log(outputs / targets).abs()**2).mean()

def rmse(outputs, targets): 
    return ((outputs - targets) / (targets + 1e-3)).abs().mean()


# Train function
def train_model(model, train_loader, optimizer, epoch, device, ga_steps, fixed_dropout):
    model.train()
    # import pdb; pdb.set_trace() 
    if fixed_dropout: 
        for mod in model.modules(): 
            if isinstance(mod, nn.Dropout): 
                mod.eval()
                mod.requires_grad = False 
                # import pdb; pdb.set_trace() 
            
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        ids = data["ids"].to(device)
        mask = data["mask"].to(device)
        position_ids = data["position_ids"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        targets = data["targets"].to(device) / data["prompt_lengths"].to(device)
        
        outputs, mlm_loss = model(ids=ids, mask=mask, token_type_ids=token_type_ids, position_ids=position_ids)

        loss = loss_fn(outputs.squeeze(-1), targets) 
        loss /= ga_steps
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (batch_idx + 1) % ga_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % 200 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")


def convert_to_label(targets): 
    return torch.floor(targets)



# Validation function
def validate_and_get_class_accuracy(model, test_loader, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    fin_tasks = []
    loss = 0 
    cnt = 0

    # import pdb; pdb.set_trace()
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
            if idx == 0: 
                print("\npred_lengths == ", pred_lengths.long())
                print("targets      == ", targets)
            
            loss += rmse(pred_lengths, targets).item()
            # import pdb; pdb.set_trace() 
            cnt += 1
            
    print(f"validation Loss (mse): {loss/cnt}")
    return loss/cnt 


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# Main execution
def main():
    args = parse_args()
    print(f"lr: {args.learning_rate}, wd: {args.weight_decay}, max len: {args.max_len}")

    seed_everything(42)
    data_path = args.data_path
    df = load_data(data_path, args.added_task)

    # Tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = BERTClass(args.model_name).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), betas=(0.9,0.999), eps=1e-8, lr=args.learning_rate) # , weight_decay=args.weight_decay)

    # Split data
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f"FULL Dataset: {df.shape}")
    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"TEST Dataset: {test_dataset.shape}")

    # DataLoaders
    train_set = CustomDataset(train_dataset, tokenizer, args.max_len)
    test_set = CustomDataset(test_dataset, tokenizer, args.max_len)
    num_workers = 8
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=args.valid_batch_size, shuffle=False, num_workers=num_workers)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)

    print(f"\nStart full parameter finetuning")
    train_losses, test_losses = list(), list() 
    for epoch in range(args.epochs):
        train_model(model, train_loader, optimizer, epoch, args.device, args.ga_steps, args.fixed_dropout)

        # print("@@@ W finetune, validate on train set")
        train_loss = 0 # validate_and_get_class_accuracy(model, train_loader, args.device)
        
        print("@@@ W finetune, validate on test set")
        test_loss = validate_and_get_class_accuracy(model, test_loader, args.device)

        if args.save_check_pt:
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            bert_model_name = args.model_name.replace("/", "_")
            save_dir = f'./1-bert-length/model_{args.llm}_alg_{args.alg}_{bert_model_name}_dropout_{args.fixed_dropout}'
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)
            
            plt.plot(list(range(len(train_losses))), train_losses, linewidth=4, label='train')
            plt.legend() 
            plt.savefig(f"./{save_dir}/loss_train_{args.learning_rate}.jpg")
            plt.close() 
            
            plt.plot(list(range(len(test_losses))), test_losses, linewidth=4, label='test')
            plt.legend() 
            plt.savefig(f"./{save_dir}/loss_test_{args.learning_rate}.jpg")
            plt.close() 
            
            np.save(f"./{save_dir}/loss_train.npy", train_losses)
            np.save(f"./{save_dir}/loss_test.npy", test_losses)
            if len(test_losses) <= 1 or test_loss <= min(test_losses) + 1e-5: 
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    },
                    f'./{save_dir}/finetuned_final_maxlen{args.max_len}_lr{args.learning_rate}_wd{args.weight_decay}_ga{args.ga_steps}.pt') # _balance{args.data_rebalance}
            print(save_dir, flush=True)
            
            print(f" Epoch == {epoch}\n\n")


if __name__ == "__main__":
    main()