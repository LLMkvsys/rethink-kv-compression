# hardware config
gpuid=0
# model and quant config (e.g. meta-llama/Llama-2-7b-hf)

quant_method='gear'
# quant parameters
compress_mode='gear_batch'
quantize_bit=8
left=0.02
rank=0.02
loop=3
stream=True
streaming_gap=20
# longbench config
e=0


model=meta-llama/Meta-Llama-3-8B-instruct # meta-llama/Meta-Llama-3-8B-instruct meta-llama/Llama-3.1-8B-Instruct
# model='mistralai/Mistral-7B-v0.1'
for quantize_bit in 8 4 
do
    srun -u -p llm_s --gres=gpu:1 python 0_neg_evaluator.py \
        --model_name_or_path $model \
        --cache_dir ./cached_models \
        --quant_method $quant_method \
        --compress_mode  $compress_mode \
        --quantize_bit $quantize_bit \
        --left $left \
        --rank $rank \
        --loop $loop \
        --stream $stream \
        --streaming_gap $streaming_gap \
        --e $e         
done 
