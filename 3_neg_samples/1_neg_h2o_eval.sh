# hardware config
gpuid=3
# data processing config
batch_size=1
# model and quant config (e.g. meta-llama/Llama-2-7b-hf)
# model='mistralai/Mistral-7B-v0.1' # ~/miniconda3/envs/kivi/lib/python3.10/site-packages/transformers/models/mistral
# model='meta-llama/Llama-2-7b-hf'
model='meta-llama/Meta-Llama-3-8B-instruct'
eviction_method='h2o' # can be ["snapkv", "pyramidkv", "h2o", "streamingllm"]
# eviction parameters
max_capacity_prompts=1024
window_sizes=32
kernel_sizes=7
pooling="maxpool"
attn_impl="flash_attention_2" # support "flash_attention_2", "sdpa", "eager"
# longbench config
e=0
PYTORCH_NO_CUDA_MEMORY_CACHING=1

# export TRANSFORMERS_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_HUB_OFFLINE="1"


model=meta-llama/Meta-Llama-3-8B-instruct # meta-llama/Llama-2-7b-hf # meta-llama/Meta-Llama-3-8B-instruct meta-llama/Llama-3.1-8B-Instruct
eviction_method='h2o'
# model='mistralai/Mistral-7B-v0.1'

# set -e 

max_capacity_prompts=1024 
window_sizes=32

srun -u -p llm_s --gres=gpu:1 --job-name=h2o-long python -u 0_neg_evaluator.py \
    --batch_size $batch_size \
    --model_name_or_path $model \
    --cache_dir ./cached_models \
    --eviction_method $eviction_method \
    --max_capacity_prompts $max_capacity_prompts \
    --window_sizes $window_sizes \
    --kernel_sizes $kernel_sizes \
    --pooling $pooling \
    --attn_impl $attn_impl \
    --e $e 
