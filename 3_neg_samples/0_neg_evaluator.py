import sys 
import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
os.environ["WANDB_DISABLED"] = "true"

from utils.process_args import process_args
from utils.load_model_tokenizer import load_model_tokenizer


from utils.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)


dataset2metric = {
    "WildChat-1M": qa_f1_score, 
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    if "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(model, tokenizer, data, max_length, _max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for idx, json_obj in tqdm(enumerate(data), total=len(data)):
        raw_prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
        prompt = raw_prompt
        max_gen = max(_max_gen, min(int(len(json_obj["answers"][0].split()) * 1.1) // 32 * 32, max_length // 4))
        if len(tokenized_prompt) > max_length:
            half = int((max_length - max_gen)/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if idx == 0: 
            print(f"@@@ context_length: {context_length}, max_gen {max_gen}")
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
            )[0]
        else:
            
            if 'Llama-3' in model_name:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            else: 
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0, 
                )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        # print(f"{pred}")
        preds.append({
            "id": idx,
            "prompt": raw_prompt,
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"]
        })
        if False: 
            pass 
            # ground_truths = preds[-1]["answers"]
            # prediction = preds[-1]["pred"]
            # all_classes = preds[-1]["all_classes"]
            
            # import pdb; pdb.set_trace() 
            # for ground_truth in ground_truths: 
            #     metric = dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes)
            #     print(f"metric is {metric}")
            # print(repr(pred))
            # import pdb; pdb.set_trace()
            # exit(0)
        
    return preds


def batch_split(prompts, prompt_format, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        prompt = prompt_format.format(**prompt)
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def prepare_input(tokenizer, prompts):
    tokenizer.padding_side = 'left'
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
    return input_tokens


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)
    model2path = json.load(open("../config/model2path.json", "r"))
    model2maxlen = json.load(open("../config/model2maxlen.json", "r"))
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("../config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("../config/dataset2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define your model
    parsed_args_dict = process_args()
    model_args = parsed_args_dict["model_args"]
    data_args = parsed_args_dict["data_args"]
    training_args = parsed_args_dict["training_args"]
    model_name = model_args.model_name_or_path.split("/")[-1]
    max_length = model2maxlen[model_name]
    dtype = torch.float16
    batch_size = data_args.batch_size
    print(f"@@@ current batch_size: {batch_size}")

    # get model and tokenizer
    model, tokenizer = load_model_tokenizer(dtype, parsed_args_dict)
    model.eval()

    # define tasks and create output folders
    datasets = ["qmsum", "2wikimqa", "gov_report",  "hotpotqa", "lcc", "multi_news", "multifieldqa_en",
                "qmsum", "passage_count", "passage_retrieval_en", "qasper", "repobench-p", "samsum", "passage_count", 
                "trec", "triviaqa"]

    if not os.path.exists("pred_neg_longbench"):
        os.makedirs("pred_neg_longbench")
    
    
    with open('benchmark_neg_data.json', 'r') as json_file:
        results = json.load(json_file)
        
    # evaluate each dataset in LongBench
    print(f"@@@ current quant method: {model_args.quant_method}")
    print(f"@@@ current eviction method: {model_args.eviction_method}")
    for dataset in datasets:
        print(f"@@@ current dataset: {dataset}")

        # load data
        out_folder = "pred_neg_longbench"
        data = results[dataset]
        # data = load_dataset('json', data_files='benchmark_neg_data.json')[dataset]
        
        # create json file
        if model_args.quant_method == "kivi":
            k_bits = parsed_args_dict["kivi_args"].k_bits
            group_size = parsed_args_dict["kivi_args"].group_size
            residual_length = parsed_args_dict["kivi_args"].residual_length
            out_folder += f"/kivi_{model_name}_{max_length}_b{batch_size}_{k_bits}bits_group{group_size}_residual{residual_length}"
        elif model_args.quant_method == "gear":
            quantize_bit = parsed_args_dict["gear_args"].quantize_bit
            out_folder += f"/gear_{model_name}_{max_length}_b{batch_size}_{quantize_bit}bits"
        elif model_args.eviction_method in ["snapkv", "pyramidkv", "h2o", "streamingllm"]:
            sparsekv_args = parsed_args_dict["sparsekv_args"]
            max_capacity_prompts = sparsekv_args.max_capacity_prompts
            window_sizes = sparsekv_args.window_sizes
            kernel_sizes = sparsekv_args.kernel_sizes
            pooling = sparsekv_args.pooling

            out_folder += f"/{model_args.eviction_method}_{model_name}_{max_length}_b{batch_size}_c{max_capacity_prompts}_w{window_sizes}_k{kernel_sizes}_{pooling}"
        else:
            out_folder += f"/hf_{model_name}_{max_length}_b{batch_size}"

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_path = f"{out_folder}/{dataset}.jsonl"
        if os.path.exists(out_path): # FIXME: skip generated file 
            continue 
        
        # generate predictions
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        # continue 
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')