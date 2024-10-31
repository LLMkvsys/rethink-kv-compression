
# import vidur 
import os, sys 
sys.path.insert(0, './vidur')
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.config import SimulationConfig
from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.entities.request import Request
import copy 
import pandas as pd 
from vidur.app import Application
import numpy as np 
import random 

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def init_executor(alg):

    config = SimulationConfig()
    config.cluster_config.num_replicas = 1
    config.cluster_config.replica_config.model_name = f"meta-llama/Llama-2-7b-{alg}"
    # import pdb; pdb.set_trace() 
    # config.cluster_config.replica_config.get_name = lambda : f"meta-llama/Llama-2-7b-{alg}"
    config.cluster_config.replica_config.num_pipeline_stages = 1
    config.cluster_config.replica_config.device = "a6000"
    config.cluster_config.replica_config.network_device = 'a6000_pairwise_nvlink'
    # config.execution_time_predictor_config
    execution_time_predictor = ExecutionTimePredictorRegistry.get(
        config.execution_time_predictor_config.get_type(),
        predictor_config=config.execution_time_predictor_config,
        replica_config=config.cluster_config.replica_config,
        replica_scheduler_config=config.cluster_config.replica_scheduler_config,
        metrics_config=config.metrics_config,
    )
    return execution_time_predictor


def init_req(prompt_tokens, num_decode_tokens, num_processed_tokens, prefill): 
    req = Request(
        arrived_at=0, 
        num_prefill_tokens=prompt_tokens, 
        num_decode_tokens=num_decode_tokens,
        num_processed_tokens=num_processed_tokens if not prefill else 0,
    )
    req._is_prefill_complete = not prefill 
    return req 


if __name__ == '__main__': 
    seed_everything(42)
    alg2name = {
        "None-16": "fp16", 
        "KIVI-4": "kivi", 
        "GEAR-4": "gear", 
        "StreamingLLM-16": "stream", 
        "H2O-16": "h2o", 
    }
    
    for alg_name in ['fp16', 'kivi', 'gear', 'stream', 'h2o']: 

        for key, value in alg2name.items(): 
            if value == alg_name: 
                METHOD = key 
        speed_df = pd.read_csv(f"0_exp/llama-7b-{alg_name}.csv")
        app = Application(speed_df=speed_df)
        
        
        # for kv_len in [256, 512, 1024, 2048, 3072, 4000]: 
        #     thr = app.get_throughput(1, kv_len, 'None-16', 'decoding').item() 
        #     print(kv_len, 1/thr)
        os.system("rm -rf cache/*")
        execution_time_predictor = init_executor(alg_name) 
        
        prediction_errors = list() 
        for batch_size in [1, 2, 4]: 
            for num_tokens in  [256, 512, 1024, 2048, 3072, 4000]: # [1, 2, 4, 8, 16, 32, 64, 256]: 
                for stage in ['decoding', 'prefill']: # , 'decoding']: 
                    new_batch_size = batch_size
                    
                    prefill = stage == 'prefill'
                    if stage == 'decoding': 
                        num_decode_tokens = 32 
                        prompt_tokens = num_tokens
                        if alg2name[METHOD] in ['h2o', 'stream']: 
                            prompt_tokens = min(prompt_tokens, 512)
                            req = init_req(prompt_tokens=prompt_tokens, num_decode_tokens=num_decode_tokens, num_processed_tokens=prompt_tokens, prefill=False)
                            batch = Batch(
                                replica_id=0,
                                requests=[copy.deepcopy(req) for _ in range(batch_size)], 
                                num_tokens=[1 for _ in range(batch_size)],
                            )
                        else: 
                            req = init_req(prompt_tokens=prompt_tokens, num_decode_tokens=num_decode_tokens, num_processed_tokens=prompt_tokens, prefill=False)
                            batch = Batch(
                                replica_id=0,
                                requests=[copy.deepcopy(req) for _ in range(batch_size)], 
                                num_tokens=[1 for _ in range(batch_size)],
                            )
                        execution_time = execution_time_predictor.get_execution_time(
                            batch,
                            0,
                        )
                        pred_time = execution_time.model_time
                        
                    else: 
                        
                        num_decode_tokens = 32 
                        prompt_tokens = num_tokens
                        prefill = True 
                        req = init_req(prompt_tokens=prompt_tokens, num_decode_tokens=num_decode_tokens, num_processed_tokens=prompt_tokens, prefill=True)
                        if prefill: 
                            new_batch_size = int(min(4096 / num_tokens, batch_size))
                        else: 
                            new_batch_size = batch_size
                        batch = Batch(
                            replica_id=0,
                            requests=[copy.deepcopy(req) for _ in range(new_batch_size)], 
                            num_tokens=[prompt_tokens for _ in range(new_batch_size)],
                        )
                        batch._total_num_tokens_rounded = min(batch._total_num_tokens_rounded, 4096)
                        execution_time = execution_time_predictor.get_execution_time(
                            batch,
                            0,
                        )
                        pred_time = execution_time.model_time
                        

                    thr = app.get_throughput(new_batch_size, num_tokens, METHOD, stage).item() 
                    if prefill: 
                        gt_time = batch_size * num_tokens / thr
                    else: 
                        gt_time = batch_size / thr

                    prediction_errors.append((abs(gt_time - pred_time) / gt_time).item())

        print(alg_name, (1 - np.mean(prediction_errors).item()) * 100)
