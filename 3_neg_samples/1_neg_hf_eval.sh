model=meta-llama/Meta-Llama-3-8B-instruct # meta-llama/Meta-Llama-3-8B-instruct meta-llama/Llama-3.1-8B-Instruct


srun -u -p llm_s --gres=gpu:1 python 0_neg_evaluator.py \
    --model_name_or_path $model \
    --cache_dir ./cached_models \
    --e $e
