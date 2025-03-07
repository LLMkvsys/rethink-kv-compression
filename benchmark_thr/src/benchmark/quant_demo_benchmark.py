from lmdeploy import pipeline, TurbomindEngineConfig, PytorchEngineConfig, GenerationConfig
from datasets import load_dataset
from tqdm import tqdm
import time 
from transformers import LlamaConfig, MistralConfig, AutoTokenizer

import argparse
import copy 

from lmdeploy import g_timer
import json
import os
import torch


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--quant_bits', type=int, default=4)
    parser.add_argument('--quant_policy', type=str, default="None")
    return parser.parse_args(args)

# Replace [INST] and [/INST] with an empty string, o.w. generations keep repeating the input prompt
def remove_inst_tokens(text):
    cleaned_text = text.replace('[INST]', '').replace('[/INST]', '')
    return cleaned_text.strip()

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    quant_method = "None"
    args.quant_bits = 0
        
    engine_config = PytorchEngineConfig(tp=args.tp, quant_bits=args.quant_bits, 
                                            compression_config={
                                                "compression_method": quant_method,
                                                "quant_bits": args.quant_bits,
                                                "quant_policy": quant_method, 
                                            }
                                        )
    model_path = "meta-llama/Llama-2-13b-hf"
    pipe = pipeline(model_path, backend_config=engine_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                            use_fast=False if 'Llama-3' not in model_path else True,
                                            trust_remote_code=True,
                                            tokenizer_type='llama')
    # pipe = pipeline("meta-llama/Llama-2-7B", backend_config=engine_config)
    # pipe = pipeline("meta-llama/Meta-Llama-3.1-8B", backend_config=engine_config)
    # pipe = pipeline('openlm-research/open_llama_3b_v2', backend_config=engine_config)
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

    gen_config = GenerationConfig(
        max_new_tokens=32, 
        temperature=1.0, 
        do_sample=False
    )

    # samples = list() 
    # start = time.time() 
    # cnt = 0
    # for idx, json_obj in tqdm(enumerate(data), total=len(data)):
    #     raw_prompt = prompt_format.format(**json_obj)
    #     trim_prompt = raw_prompt
    #     tokenized_prompt = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
    #     # trim_prompt = tokenizer.decode(tokenized_prompt[:128*2+13], skip_special_tokens=True)
        
    #     samples.append(trim_prompt)
    #     # samples = [trim_prompt] + samples
    #     if (idx + 1) % batch_size  == 0: 
    #         samples = [copy.deepcopy(samples[0]) for _ in range(batch_size)]
    #         start = time.time() 
    #         import pdb; pdb.set_trace()
    #         response = pipe(samples, gen_config=gen_config)
    #         print("\n\n")
    #         print("**"*10)
    #         # print(repr(response[0].text))
    #         for j in range(batch_size): print(repr(response[j].text))
    #         print("**"*10)
    #         samples = list()
    #         print('elapsed time is {}s'.format(time.time() - start))
    #         start = time.time() 
    #         cnt += 1
    #         if cnt == 1: break 

    # data_path = "/home/wgao/PAPER/LLMServing/lmdemo/src/benchmark/coarse2fine/short2short.json"
    data_path = "/home/wgao/PAPER/LLMServing/lmdemo/src/benchmark/coarse2fine/long2short.json"
    with open(data_path, 'r') as f:
        eval_data = json.load(f)
    prompts = [example['prompt'] for example in eval_data]
    prompts = [remove_inst_tokens(prompt) for prompt in prompts]
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'left'

    dataset_name = os.path.split(data_path)[-1]
    total_time = 0
    total_token_num = 0
    outputs = []

    # import pdb; pdb.set_trace()
    total_prefill_tokens = 0
    total_decode_tokens = 0
    prefill_stage_duration = 0
    decode_stage_duration = 0

    start_time = time.time()

    torch.cuda.reset_peak_memory_stats()
    # st_times -> prefill_ending_times -> decode -> end_times
    # g_timer.st_times.clear()
    # g_timer.prefill_times.clear()
    # g_timer.end_times.clear()
    g_timer['st_times'].clear()
    g_timer['prefill_times'].clear()
    g_timer['end_times'].clear()

    debug_cnt = 0
    for i in tqdm(range(0, len(prompts), batch_size)):
        debug_cnt += 1
        # print(debug_cnt)
        if debug_cnt > 10: break

        start_time = time.time()
        inputs = tokenizer(prompts[i:i + batch_size], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].cuda()

        torch.cuda.synchronize()
        # g_timer.forward_count = 0
        # g_timer.st_times.append(time.time())
        g_timer['forward_count'] = 0
        g_timer['st_times'].append(time.time())
        response = pipe(prompts[i:i + batch_size], gen_config=gen_config)
        # output = model.generate(input_ids, **generation_kwargs) # FIXME: use lmdeploy code
        g_timer['end_times'].append(time.time())
        torch.cuda.synchronize()

        # if batch_size == 1:
        #     output = response[0].text
        #     output = tokenizer(output, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
        #     # out_str = tokenizer.decode(output[input_ids.size(1):], skip_special_tokens=True)
        # else:
        #     output = response
        #     # out_str = tokenizer.batch_decode(output[:, input_ids.size(1):], skip_special_tokens=True)
        end_time = time.time()
        total_time += end_time - start_time
        # token_num = int((output[:, input_ids.size(1):] != tokenizer.pad_token_id).sum().cpu())
        output = response[0].text
        output = tokenizer(output, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
        token_num = int((output[:,input_ids.size(1):] != tokenizer.pad_token_id).sum().cpu()) # find the EOS token
        total_decode_tokens += token_num
        print(total_token_num)

        prompt_num = int((input_ids != tokenizer.pad_token_id).sum().cpu())
        total_prefill_tokens += prompt_num

        # for prompt, o_str in zip(prompts[i:i + batch_size], out_str):
        #     outputs.append(o_str)

    # calculate the throughputs for prefill/decode stage
    prefill_stage_duration += sum([(j - i) for i, j in zip(g_timer['st_times'], g_timer['prefill_times'])])
    decode_stage_duration += sum([(j - i) for i, j in zip(g_timer['prefill_times'], g_timer['end_times'])])
    prefill_throughput = total_prefill_tokens / prefill_stage_duration
    decode_throughput = total_decode_tokens / decode_stage_duration

    # display the statistics
    print(f'prefill throughput: {prefill_throughput :.3f} tokens/s')
    print(f'decode throughput: {decode_throughput :.3f} tokens/s')
    print(f'peak mem: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB')
    # print(g_timer)

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
    # for prompt, output in zip(prompts, outputs):
    #     result['result'].append({
    #         'prompt': prompt,
    #         'output': output,
    #     })
    json_str = json.dumps(result, ensure_ascii=False, indent=2)

    out_path = f"/home/wgao/PAPER/LLMServing/lmdemo/src/benchmark/logs/bs{batch_size}_data{dataset_name}.log"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)