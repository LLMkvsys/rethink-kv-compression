model=meta-llama/Meta-Llama-3-8B-instruct # meta-llama/Meta-Llama-3-8B-instruct meta-llama/Llama-3.1-8B-Instruct

for model in $(ls ./pred_neg_longbench | grep 8B)
do 
    srun -u -p llm_s --gres=gpu:1 python -u 0_neg_scorer.py --model $model --path="./pred_neg_longbench"
done 