import os, sys 
from lmdeploy import pipeline, TurbomindEngineConfig, PytorchEngineConfig, GenerationConfig
from datasets import load_dataset
from tqdm import tqdm
import time 
from transformers import LlamaConfig, MistralConfig, AutoTokenizer

import torch 
import argparse
import copy 


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--quant_bits', type=int, default=0)
    parser.add_argument('--quant_policy', type=str, default="None")
    parser.add_argument('--sparse_policy', type=str, default=None)
    parser.add_argument('--prompt_length', type=int, default=4096)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    return parser.parse_args(args)

# cache config is  CacheConfig(max_batches=128, block_size=64, num_cpu_blocks=248, num_gpu_blocks=1679, window_size=-1, cache_max_entry_count=0.8, max_prefill_token_num=4096, enable_prefix_caching=False, quant_bits=8, quant_policy='KIVI')

def benchmark(samples, gen_config, num_iters): 
    torch.cuda.synchronize()
    start = time.time() 
    for i in range(num_iters): 
        response = pipe(samples, gen_config=gen_config)
        # print("end of benchmark")
    torch.cuda.synchronize()
    return (time.time() - start) / num_iters, response
        
def cal_block_size(base_block, quant_bits): 
    # return base_block
    if quant_bits == 4: 
        return  base_block
    if quant_bits == 8: 
        return base_block
    if quant_bits == 2: 
        return base_block
    if quant_bits == 0: 
        return base_block * 1
    
    return base_block


if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    
    quant_method = args.quant_policy

    if (quant_method == "None" and args.quant_bits not in [0]) or \
        (quant_method in ["HEAD", "KIVI"] and args.quant_bits not in [4, 8]) or \
        (quant_method in ["H2O", "StreamingLLM"] and args.quant_bits not in [0]): 
        exit(0)
    

    
    if args.sparse_policy is not None: 
        quant_method = args.sparse_policy
        engine_config = PytorchEngineConfig(tp=args.tp, quant_bits=0, block_size=cal_block_size(64,0), cache_max_entry_count=0.8,
                                                compression_config={
                                                    "compression_method": quant_method,
                                                    "quant_bits": 0,
                                                    "quant_policy": quant_method, 
                                                }
                                            )
        if args.model_path == "lmsys/longchat-7b-v1.5-32k":
            engine_config = PytorchEngineConfig(tp=args.tp, session_len=32768, max_prefill_token_num=32768,
                                                quant_bits=0, block_size=cal_block_size(64, 0), cache_max_entry_count=0.8,
                                                    compression_config={
                                                        "compression_method": quant_method,
                                                        "quant_bits": 0,
                                                        "quant_policy": quant_method, 
                                                    }
                                                )
        if args.model_path == "lmsys/longchat-13b-16k":
            engine_config = PytorchEngineConfig(tp=args.tp, session_len=16384, max_prefill_token_num=16384,
                                                quant_bits=args.quant_bits, block_size=cal_block_size(64, args.quant_bits), cache_max_entry_count=0.8,
                                                    compression_config={
                                                        "compression_method": quant_method,
                                                        "quant_bits": args.quant_bits,
                                                        "quant_policy": quant_method, 
                                                    }
                                                )
    else:
        quant_method = args.quant_policy
        engine_config = PytorchEngineConfig(tp=args.tp, quant_bits=args.quant_bits, block_size=cal_block_size(64, args.quant_bits), cache_max_entry_count=0.8,
                                                compression_config={
                                                    "compression_method": quant_method,
                                                    "quant_bits": args.quant_bits,
                                                    "quant_policy": quant_method, 
                                                }
                                            )

    model_ident = args.model_path.replace('/', '-')
    log_file = f"0_sparse_long_logs/decoding_model_{model_ident}_batch_size_{args.batch_size}_tp_{args.tp}_quant_bits_{args.quant_bits}_quant_policy_{quant_method}_prompt_length_{args.prompt_length}.log"
    with open(log_file, "w") as f: pass 
    os.environ["llama_time_log_file"] = log_file
    
    prompt_format = 'You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:'
    model_path = args.model_path
    
    
    pipe = pipeline(model_path, backend_config=engine_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                            use_fast=False if 'Llama-3' not in model_path else True,
                                            trust_remote_code=True,
                                            tokenizer_type='llama')
    

    gen_config = GenerationConfig(
        max_new_tokens=1, 
        temperature=1.0, 
        do_sample=False
    )
    samples = list() 
    start = time.time() 
    cnt = 0 
    
    datasets = data = load_dataset('THUDM/LongBench', "multi_news", split='test')
    for idx, json_obj in tqdm(enumerate(data), total=len(data)):
        raw_prompt = prompt_format.format(**json_obj) + prompt_format.format(**json_obj) + prompt_format.format(**json_obj) 
        if args.prompt_length > 10000: 
            trim_prompt = ' '.join([raw_prompt for _ in range(5)])
        tokenized_prompt = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
        # import pdb; pdb.set_trace() 
        trim_prompt = tokenizer.decode(tokenized_prompt[:args.prompt_length], skip_special_tokens=True)
        
        samples.append(trim_prompt)
        # samples = [trim_prompt] + samples
        if (idx + 1) % batch_size  == 0: 
            samples = [copy.deepcopy(samples[0]) for _ in range(batch_size)]
            
            # response = pipe(samples, gen_config=gen_config) # warm up 
            start = time.time() 
            num_iters = 3
            prefill_time_cost, _ = benchmark(samples, gen_config, num_iters=num_iters)
            DECLength = 64 + 1
            gen_config.min_new_tokens = DECLength
            gen_config.max_new_tokens = DECLength
            decoding_time_cost, response = benchmark(samples, gen_config, num_iters=num_iters)
            time_cost = decoding_time_cost - prefill_time_cost
            assert len(response[0].token_ids) == DECLength
            print("\n\n")
            
            
            with open(log_file, "a") as f: 
                f.write('elapsed time is {}s'.format(time_cost))
            
            print('elapsed time is {}s'.format(time_cost))
            start = time.time() 
            cnt += 1
            if cnt == 1: break 
            
    print(log_file)
                
        