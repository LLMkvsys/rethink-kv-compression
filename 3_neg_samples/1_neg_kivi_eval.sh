# hardware config
gpuid=2
# data processing config
batch_size=1
# model and quant config (e.g. meta-llama/Llama-2-7b-hf)
model='meta-llama/Meta-Llama-3-8B-instruct'
quant_method='kivi'
# quant parameters
k_bits=4
v_bits=4
group_size=32
residual_length=128
# longbench config
e=0


# model=meta-llama/Llama-2-7b-hf # meta-llama/Meta-Llama-3-8B-instruct meta-llama/Llama-3.1-8B-Instruct
model=meta-llama/Meta-Llama-3-8B-instruct
# model='mistralai/Mistral-7B-v0.1'

for bits in 4 2 # 8
do 
    srun -u -p llm_s --gres=gpu:1 --job-name=kivi-long python -u 0_neg_evaluator.py \
        --batch_size $batch_size \
        --model_name_or_path $model \
        --cache_dir ./cached_models \
        --quant_method $quant_method \
        --k_bits $bits \
        --v_bits $bits \
        --group_size $group_size \
        --residual_length $residual_length \
        --e $e & 
done 

wait 