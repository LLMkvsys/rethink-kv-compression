import os, sys 
import numpy as np 
compress_ident_list = [
    ('model_Meta-Llama-3-8B-instruct_8192_alg_hf_allenai_longformer-base-4096_dropout_False', 'hf'), 
    ('model_Meta-Llama-3-8B-instruct_8192_alg_kivi_allenai_longformer-base-4096_dropout_False', 'kivi'), 
    ('model_Meta-Llama-3-8B-instruct_8192_alg_gear_allenai_longformer-base-4096_dropout_False', 'gear'), 
    ('model_Meta-Llama-3-8B-instruct_8192_alg_h2o_allenai_longformer-base-4096_dropout_False', 'h2o'), 
    ('model_Meta-Llama-3-8B-instruct_8192_alg_streamingllm_allenai_longformer-base-4096_dropout_False', 'streamingllm'), 
]

for compress_path, compress_alg in compress_ident_list: 
    filename = os.path.join("1-bert-length/", compress_path, 'loss_test.npy')
    loss_info = np.load(filename, allow_pickle=True).tolist() 
    print(compress_alg, 100 - min(loss_info) * 100)