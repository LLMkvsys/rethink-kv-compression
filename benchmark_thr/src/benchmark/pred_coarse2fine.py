# Reference: https://github.com/chenyushuo/Coarse-to-Fine-Evaluation-of-Inference-Efficiency/blob/main/batching_inference.py
# Data: https://github.com/chenyushuo/Coarse-to-Fine-Evaluation-of-Inference-Efficiency/tree/main/data

import os
import time
import json
import logging
import random
from utils.process_args import process_args
from utils.load_model_tokenizer import load_model_tokenizer
import utils.global_timer as g_timer


from tqdm import tqdm
import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


class StopLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, stop='\n\n\n\n'):
        self.stop_list = torch.tensor([13, 13, 13, 13]).cuda() # tokenizer.encode(stop, add_special_tokens=False, return_tensors="pt").cuda()
        self.inf = torch.tensor(1e5).cuda()
        self.stop_length = len(self.stop_list)
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.all(input_ids[:, -self.stop_length:] == self.stop_list, dim=1)
        scores[:, self.eos_token_id] = torch.where(mask, self.inf, scores[:, self.eos_token_id])
        return scores

# Replace [INST] and [/INST] with an empty string, o.w. generations keep repeating the input prompt
def remove_inst_tokens(text):
    cleaned_text = text.replace('[INST]', '').replace('[/INST]', '')
    return cleaned_text.strip()


@torch.no_grad()
def main(parsed_args_dict):
    random.seed(42)
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    # define your model
    model_args = parsed_args_dict["model_args"]
    data_args = parsed_args_dict["data_args"]
    coarse2fine_data_args = parsed_args_dict["coarse2fine_data_args"]
    training_args = parsed_args_dict["training_args"]
    model_name = model_args.model_name_or_path.split("/")[-1]
    max_length = model2maxlen[model_name]
    dtype = torch.float16
    batch_size = data_args.batch_size
    print(f"@@@ current batch size: {batch_size}")

    logging.info("loading data and model...")
    if coarse2fine_data_args.data_path is None:
        raise ValueError
    with open(coarse2fine_data_args.data_path, 'r') as f:
        eval_data = json.load(f)
    dataset_name = os.path.split(coarse2fine_data_args.data_path)[-1]
    print(f"@@@ dataset_name: {dataset_name}")

    # result_folder = "2-efficiency/"
    result_folder = "2-efficiency_flashatt/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # get model and tokenizer
    model, tokenizer = load_model_tokenizer(dtype, parsed_args_dict, perform_timing=True) # NOTE: add timing
    model.eval()
    prompts = [example['prompt'] for example in eval_data]
    prompts = [remove_inst_tokens(prompt) for prompt in prompts]
    # FIXME: XINYUZHOU, use max_tokens=4096 or limited by this ground truth upper-bound?
    # greedy search tends to generate till the max len, rather than wisely stopped at models EOS output
    max_tokens = max([example['max_tokens'] for example in eval_data])
    # s2s: 49, s2l: 996, l2s: 100, s-16k: 16k
    print(f"current max tokens {max_tokens}")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'left'
    # print(f"@@@ check prompts: {prompts[0]}")

    # create output folder and json file name
    out_folder = result_folder
    if model_args.quant_method == "kivi":
        k_bits = parsed_args_dict["kivi_args"].k_bits
        group_size = parsed_args_dict["kivi_args"].group_size
        residual_length = parsed_args_dict["kivi_args"].residual_length
        out_folder += f"kivi_{model_name}_{max_length}_b{batch_size}_{k_bits}bits_group{group_size}_residual{residual_length}"
    elif model_args.quant_method == "gear":
        quantize_bit = parsed_args_dict["gear_args"].quantize_bit
        out_folder += f"gear_{model_name}_{max_length}_b{batch_size}_{quantize_bit}bits"
    elif model_args.eviction_method in ["snapkv", "pyramidkv", "h2o", "streamingllm"]:
        sparsekv_args = parsed_args_dict["sparsekv_args"]
        max_capacity_prompts = sparsekv_args.max_capacity_prompts
        window_sizes = sparsekv_args.window_sizes
        kernel_sizes = sparsekv_args.kernel_sizes
        pooling = sparsekv_args.pooling
        attn_impl = sparsekv_args.attn_impl

        out_folder += f"{model_args.eviction_method}_{model_name}_{max_length}_b{batch_size}_c{max_capacity_prompts}_w{window_sizes}_k{kernel_sizes}_{pooling}_{attn_impl}"
    else:
        out_folder += f"hf_{model_name}_{max_length}_b{batch_size}"

    # create json file
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_path = f"{out_folder}/{dataset_name}"

    total_time = 0
    total_token_num = 0
    outputs = []
    generation_kwargs = dict(
        # max_new_tokens=coarse2fine_data_args.max_new_tokens,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    if coarse2fine_data_args.remove_row_delimiter:
        processors = LogitsProcessorList()
        processors.append(StopLogitsProcessor(tokenizer, '\n\n\n\n'))
        generation_kwargs['logits_processor'] = processors
    if coarse2fine_data_args.ignore_eos:
        generation_kwargs['eos_token_id'] = -1

    # import pdb; pdb.set_trace()
    total_prefill_tokens = 0
    total_decode_tokens = 0
    prefill_stage_duration = 0
    decode_stage_duration = 0

    start_time = time.time()

    torch.cuda.reset_peak_memory_stats()
    # st_times -> prefill_ending_times -> decode -> end_times
    g_timer.st_times.clear()
    g_timer.prefill_times.clear()
    g_timer.end_times.clear()

    for i in tqdm(range(0, len(prompts), batch_size)):
        start_time = time.time()
        inputs = tokenizer(prompts[i:i + batch_size], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].cuda()

        torch.cuda.synchronize()
        g_timer.forward_count = 0
        g_timer.st_times.append(time.time())
        output = model.generate(input_ids, **generation_kwargs)
        g_timer.end_times.append(time.time())
        torch.cuda.synchronize()

        out_str = tokenizer.batch_decode(output[:, input_ids.size(1):], skip_special_tokens=True)
        end_time = time.time()
        total_time += end_time - start_time
        token_num = int((output[:, input_ids.size(1):] != tokenizer.pad_token_id).sum().cpu())
        total_decode_tokens += token_num

        prompt_num = int((input_ids != tokenizer.pad_token_id).sum().cpu())
        total_prefill_tokens += prompt_num

        for prompt, o_str in zip(prompts[i:i + batch_size], out_str):
            outputs.append(o_str)

    # calculate the throughputs for prefill/decode stage
    prefill_stage_duration += sum([(j - i) for i, j in zip(g_timer.st_times, g_timer.prefill_times)])
    decode_stage_duration += sum([(j - i) for i, j in zip(g_timer.prefill_times, g_timer.end_times)])
    prefill_throughput = total_prefill_tokens / prefill_stage_duration
    decode_throughput = total_decode_tokens / decode_stage_duration

    # display the statistics
    print(f'prefill throughput: {prefill_throughput :.3f} tokens/s')
    print(f'decode throughput: {decode_throughput :.3f} tokens/s')
    print(f'peak mem: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB')

    result = {
        'dataset': dataset_name,
        'total_time': total_time,
        'sequence_num': len(prompts),
        'total_prefill_tokens': total_prefill_tokens,
        'total_prefill_time': prefill_stage_duration,
        'prefill_throughput': prefill_throughput,
        'total_decode_tokens': total_decode_tokens,
        'total_decode_time': decode_stage_duration,
        'decode_throughput': decode_throughput,
        'result': [],
    }
    for prompt, output in zip(prompts, outputs):
        result['result'].append({
            'prompt': prompt,
            'output': output,
        })
    json_str = json.dumps(result, ensure_ascii=False, indent=2)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)


if __name__ == '__main__':
    parsed_args_dict = process_args()
    main(parsed_args_dict)
